-- MUSA (Moore Threads) targets for LLAISYS
-- Included from xmake.lua only when --musa-gpu=y is set.
-- Build with: xmake f --musa-gpu=y --cuda=/usr/local/musa -c
-- mcc -mtgpu selects the MUSA backend; default architecture is fine for MTT S5000.

-- MUSA source files are compiled directly into the shared library
-- (llaisys target) to avoid static-library symbol-dropping issues.
-- These targets remain as empty placeholders to satisfy the dep chain.

target("llaisys-device-musa")
    set_kind("static")
    set_languages("cxx17")
    on_install(function (target) end)
target_end()

target("llaisys-ops-musa")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    on_install(function (target) end)
target_end()

target("llaisys-models-musa")
    set_kind("static")
    add_deps("llaisys-ops")
    set_languages("cxx17")
    on_install(function (target) end)
target_end()
