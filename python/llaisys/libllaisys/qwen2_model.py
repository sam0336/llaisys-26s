"""
ctypes wrappers for the Qwen2 model C API.

Exposes:
    - LlaisysQwen2Meta        : model hyperparameters
    - LlaisysQwen2Weights     : weight-tensor pointers (per-layer arrays)
    - llaisysQwen2ModelCreate / Destroy / Weights / Infer
"""


from ctypes import (
    c_void_p,
    c_size_t,
    c_int,
    c_float,
    c_int64,
    POINTER,
    Structure,
)

from .llaisys_types import llaisysDataType_t, llaisysDeviceType_t
from .tensor import llaisysTensor_t


# ---------------------------------------------------------------------------
# C structs  (must match include/llaisys/models/qwen2.h exactly)
# ---------------------------------------------------------------------------

class LlaisysQwen2Meta(Structure):
    """Corresponds to C struct LlaisysQwen2Meta."""
    _fields_ = [
        ("dtype",     llaisysDataType_t),
        ("nlayer",    c_size_t),
        ("hs",        c_size_t),   # hidden_size
        ("nh",        c_size_t),   # num_attention_heads
        ("nkvh",      c_size_t),   # num_key_value_heads
        ("dh",        c_size_t),   # head_dim
        ("di",        c_size_t),   # intermediate_size
        ("maxseq",    c_size_t),   # max_position_embeddings
        ("voc",       c_size_t),   # vocab_size
        ("epsilon",   c_float),    # rms_norm_eps
        ("theta",     c_float),    # rope_theta
        ("end_token", c_int64),    # eos_token_id
    ]


class LlaisysQwen2Weights(Structure):
    """Corresponds to C struct LlaisysQwen2Weights.

    Single tensors are bare llaisysTensor_t handles.
    Per-layer fields are pointers to C arrays of llaisysTensor_t[nlayer].
    """
    _fields_ = [
        ("in_embed",    llaisysTensor_t),
        ("out_embed",   llaisysTensor_t),
        ("out_norm_w",  llaisysTensor_t),
        ("attn_norm_w", POINTER(llaisysTensor_t)),
        ("attn_q_w",    POINTER(llaisysTensor_t)),
        ("attn_q_b",    POINTER(llaisysTensor_t)),
        ("attn_k_w",    POINTER(llaisysTensor_t)),
        ("attn_k_b",    POINTER(llaisysTensor_t)),
        ("attn_v_w",    POINTER(llaisysTensor_t)),
        ("attn_v_b",    POINTER(llaisysTensor_t)),
        ("attn_o_w",    POINTER(llaisysTensor_t)),
        ("mlp_norm_w",  POINTER(llaisysTensor_t)),
        ("mlp_gate_w",  POINTER(llaisysTensor_t)),
        ("mlp_up_w",    POINTER(llaisysTensor_t)),
        ("mlp_down_w",  POINTER(llaisysTensor_t)),
    ]


# ---------------------------------------------------------------------------
# Model handle  (opaque pointer)
# ---------------------------------------------------------------------------
llaisysQwen2Model_t = c_void_p


# ---------------------------------------------------------------------------
# Register function signatures with the shared library
# ---------------------------------------------------------------------------

def load_qwen2_model(lib):
    """Register all Qwen2 model C API signatures with the shared library."""

    lib.llaisysQwen2ModelCreate.argtypes = [
        POINTER(LlaisysQwen2Meta),  # meta
        llaisysDeviceType_t,        # device
        POINTER(c_int),             # device_ids array
        c_int,                      # ndevice
    ]
    lib.llaisysQwen2ModelCreate.restype = llaisysQwen2Model_t

    lib.llaisysQwen2ModelDestroy.argtypes = [llaisysQwen2Model_t]
    lib.llaisysQwen2ModelDestroy.restype = None

    lib.llaisysQwen2ModelWeights.argtypes = [llaisysQwen2Model_t]
    lib.llaisysQwen2ModelWeights.restype = POINTER(LlaisysQwen2Weights)

    lib.llaisysQwen2ModelInfer.argtypes = [
        llaisysQwen2Model_t,        # model
        POINTER(c_int64),           # token_ids array
        c_size_t,                   # ntoken
    ]
    lib.llaisysQwen2ModelInfer.restype = c_int64
