import json
import re
from ctypes import c_void_p, c_size_t, c_int, c_int64, POINTER, cast, c_ubyte
from typing import Sequence

from pathlib import Path

from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType
from ..libllaisys import DataType
from ..libllaisys import llaisysDataType_t, llaisysDeviceType_t
from ..libllaisys import LlaisysQwen2Meta
from ..tensor import Tensor

# ---------------------------------------------------------------------------
# Weight-name → struct-field mapping
# ---------------------------------------------------------------------------

# safetensors key → LlaisysQwen2Weights field (single tensors)
_SINGLE_MAP = {
    "model.embed_tokens.weight": "in_embed",
    "lm_head.weight":            "out_embed",
    "model.norm.weight":         "out_norm_w",
}

# safetensors suffix → LlaisysQwen2Weights field (per-layer tensors)
_LAYER_SUFFIX_MAP = {
    "input_layernorm.weight":          "attn_norm_w",
    "self_attn.q_proj.weight":         "attn_q_w",
    "self_attn.q_proj.bias":           "attn_q_b",
    "self_attn.k_proj.weight":         "attn_k_w",
    "self_attn.k_proj.bias":           "attn_k_b",
    "self_attn.v_proj.weight":         "attn_v_w",
    "self_attn.v_proj.bias":           "attn_v_b",
    "self_attn.o_proj.weight":         "attn_o_w",
    "post_attention_layernorm.weight": "mlp_norm_w",
    "mlp.gate_proj.weight":            "mlp_gate_w",
    "mlp.up_proj.weight":              "mlp_up_w",
    "mlp.down_proj.weight":            "mlp_down_w",
}

# Regex: extracts (layer_idx, suffix) from "model.layers.5.mlp.up_proj.weight"
_LAYER_RE = re.compile(r"model\.layers\.(\d+)\.(.+)")

# config.json torch_dtype string → LLAISYS DataType
_TORCH_DTYPE_MAP = {
    "float32":  DataType.F32,
    "float16":  DataType.F16,
    "bfloat16": DataType.BF16,
}

# safetensors dtype string → LLAISYS DataType
_SAFETENSORS_DTYPE_MAP = {
    "BF16": DataType.BF16,
    "F16":  DataType.F16,
    "F32":  DataType.F32,
    "F64":  DataType.F64,
    "I64":  DataType.I64,
    "I32":  DataType.I32,
    "I16":  DataType.I16,
    "I8":   DataType.I8,
    "U8":   DataType.U8,
    "BOOL": DataType.BOOL,
    "U32":  DataType.U32,
}


def _parse_safetensors(filepath: Path):
    """Parse a safetensors file manually.

    Safetensors format:
        8 bytes  — header_size (u64, little-endian)
        N bytes  — JSON header (UTF-8)
        rest     — concatenated raw tensor data

    Yields (name, dtype_str, shape, raw_bytes) tuples.
    """
    with open(filepath, "rb") as f:
        header_size = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_size))
        data_start = f.tell()

        for name, info in header.items():
            if name == "__metadata__":
                continue

            dtype_str = info["dtype"]
            shape = info["shape"]
            start, end = info["data_offsets"]

            f.seek(data_start + start)
            raw = f.read(end - start)

            yield name, dtype_str, shape, raw


# ---------------------------------------------------------------------------
# Qwen2
# ---------------------------------------------------------------------------

class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        self._device = device

        # ----------------------------------------------------------
        # 1. Read config.json & create the C++ model
        # ----------------------------------------------------------
        with open(model_path / "config.json", "r") as f:
            cfg = json.load(f)

        nlayer = int(cfg["num_hidden_layers"])
        hs     = int(cfg["hidden_size"])
        nh     = int(cfg["num_attention_heads"])
        nkvh   = int(cfg["num_key_value_heads"])
        dh     = int(cfg.get("head_dim", hs // nh))
        di     = int(cfg.get("intermediate_size", 0))
        maxseq = min(int(cfg.get("max_position_embeddings", 131072)), 2048)
        voc    = int(cfg["vocab_size"])

        epsilon   = float(cfg.get("rms_norm_eps", 1e-6))
        theta     = float(cfg.get("rope_theta", 10000.0))
        end_token = int(cfg.get("eos_token_id", -1))

        torch_dtype_str = cfg.get("torch_dtype", "bfloat16")
        model_dtype = _TORCH_DTYPE_MAP.get(torch_dtype_str, DataType.BF16)

        self._end_token = end_token

        # Populate LlaisysQwen2Meta ...
        meta = LlaisysQwen2Meta()
        meta.dtype     = llaisysDataType_t(model_dtype.value)
        meta.nlayer    = c_size_t(nlayer)
        meta.hs        = c_size_t(hs)
        meta.nh        = c_size_t(nh)
        meta.nkvh      = c_size_t(nkvh)
        meta.dh        = c_size_t(dh)
        meta.di        = c_size_t(di)
        meta.maxseq    = c_size_t(maxseq)
        meta.voc       = c_size_t(voc)
        meta.epsilon   = epsilon
        meta.theta     = theta
        meta.end_token = c_int64(end_token)

        device_ids = (c_int * 1)(0)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            meta,
            llaisysDeviceType_t(device.value),
            device_ids,
            c_int(1),
        )
        if not self._model:
            raise RuntimeError("llaisysQwen2ModelCreate returned NULL")

        # ----------------------------------------------------------
        # 2. Pre-allocate per-layer arrays in the weights struct
        # ----------------------------------------------------------
        self._weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model)

        ArrayType = c_void_p * nlayer
        self._weight_arrays = {}
        for field_name in _LAYER_SUFFIX_MAP.values():
            if field_name in self._weight_arrays:
                continue
            arr = ArrayType()
            self._weight_arrays[field_name] = arr
            setattr(
                self._weights_ptr.contents,
                field_name,
                cast(arr, POINTER(c_void_p)),
            )

        # Keep Python Tensor objects alive (so their C handles stay valid).
        self._weight_tensors = []

        # Track whether out_embed was loaded (for tied-embeddings fallback).
        out_embed_loaded = False

        # ----------------------------------------------------------
        # 3. Load safetensors weights  (preserving original loop structure)
        # ----------------------------------------------------------
        for file in sorted(model_path.glob("*.safetensors")):
            for name_, dtype_str, shape, raw in _parse_safetensors(file):
                # -- single tensors --
                if name_ in _SINGLE_MAP:
                    field = _SINGLE_MAP[name_]
                    t = self._create_weight_tensor(shape, dtype_str, raw)
                    self._weight_tensors.append(t)
                    setattr(self._weights_ptr.contents, field, t._tensor)
                    if field == "out_embed":
                        out_embed_loaded = True
                    continue

                # -- per-layer tensors --
                m = _LAYER_RE.match(name_)
                if m:
                    layer_idx = int(m.group(1))
                    suffix    = m.group(2)
                    field     = _LAYER_SUFFIX_MAP.get(suffix)
                    if field is None:
                        continue
                    t = self._create_weight_tensor(shape, dtype_str, raw)
                    self._weight_tensors.append(t)
                    self._weight_arrays[field][layer_idx] = t._tensor

        # ----------------------------------------------------------
        # 4. Tied embeddings: if lm_head.weight is absent, reuse in_embed
        # ----------------------------------------------------------
        if not out_embed_loaded:
            in_embed = self._weights_ptr.contents.in_embed
            if in_embed:
                self._weights_ptr.contents.out_embed = in_embed
            else:
                raise RuntimeError("in_embed not loaded; cannot set out_embed")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_weight_tensor(
        self, shape: list, dtype_str: str, raw: bytes
    ) -> Tensor:
        """Create a Tensor on the model's device and copy *raw* bytes into it."""
        dt = _SAFETENSORS_DTYPE_MAP[dtype_str]
        t = Tensor(tuple(shape), dtype=dt, device=self._device, device_id=0)
        buf = (c_ubyte * len(raw)).from_buffer_copy(raw)
        t.load(cast(buf, c_void_p))
        return t

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        tokens = list(inputs)
        n_input = len(tokens)
        if n_input == 0:
            return []

        # Prefill — feed all input tokens, get first predicted token.
        arr = (c_int64 * n_input)(*tokens)
        next_token = int(
            LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model, arr, c_size_t(n_input)
            )
        )

        generated = [next_token]

        # Autoregressive decode.
        limit = max_new_tokens - 1 if max_new_tokens else 128
        for _ in range(limit):
            if next_token == self._end_token:
                break
            one = (c_int64 * 1)(next_token)
            next_token = int(
                LIB_LLAISYS.llaisysQwen2ModelInfer(
                    self._model, one, c_size_t(1)
                )
            )
            generated.append(next_token)

        return tokens + generated

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def __del__(self):
        if hasattr(self, "_model") and self._model is not None:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None
