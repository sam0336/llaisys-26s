-- NVIDIA CUDA targets for LLAISYS
-- Included from xmake.lua only when --nv-gpu=y is set.

-- CUDA architecture flags: support a broad range of GPU generations.
-- Users can override with: xmake f --nv-gpu=y --cugencodes="compute_89,sm_89"
if not has_config("cugencodes") then
    add_cugencodes(
        "compute_75", "sm_75",   -- Turing (RTX 20xx, T4)
        "compute_80", "sm_80",   -- Ampere (A100, A30)
        "compute_86", "sm_86",   -- Ampere (RTX 30xx, A40)
        "compute_89", "sm_89",   -- Ada Lovelace (RTX 40xx, L40)
        "compute_90a", "sm_90a"  -- Hopper (H100, H200)
    )
end

-- NVIDIA device target: CUDA Runtime API wrapper + device resources.
target("llaisys-device-nvidia")
    set_kind("static")
    add_rules("cuda")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    add_includedirs("../include")
    add_files("../src/device/nvidia/*.cu")
    on_install(function (target) end)
target_end()

-- NVIDIA ops target: per-operator CUDA kernels.
target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-tensor")
    add_rules("cuda")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    add_includedirs("../include")
    add_files("../src/ops/*/nvidia/*.cu")
    on_install(function (target) end)
target_end()

-- NVIDIA models target: model inference on GPU.
target("llaisys-models-nvidia")
    set_kind("static")
    add_deps("llaisys-ops")
    add_rules("cuda")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    add_includedirs("../include")
    add_files("../src/models/qwen2/nvidia/*.cu")
    on_install(function (target) end)
target_end()
