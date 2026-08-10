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

-- CUDA source files are now compiled directly into the shared library
-- (llaisys target) to avoid static-library symbol-dropping issues.
-- These targets remain as empty placeholders to satisfy the dep chain.

target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx17")
    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    on_install(function (target) end)
target_end()

target("llaisys-models-nvidia")
    set_kind("static")
    add_deps("llaisys-ops")
    set_languages("cxx17")
    on_install(function (target) end)
target_end()
