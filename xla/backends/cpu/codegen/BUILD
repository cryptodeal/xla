load("//xla:xla.bzl", "xla_cc_test", "xla_internal")
load(
    "//xla/tsl/platform:build_config_root.bzl",
    "if_llvm_aarch64_available",
    "if_llvm_powerpc_available",
    "if_llvm_system_z_available",
    "if_llvm_x86_available",
)
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla/backends/cpu:xla_backend_cpu_internal_access"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "contiguous_section_memory_manager",
    srcs = ["contiguous_section_memory_manager.cc"],
    hdrs = ["contiguous_section_memory_manager.h"],
    deps = [
        "//xla:util",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:ExecutionEngine",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Support",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "cpu_features",
    srcs = ["cpu_features.cc"],
    hdrs = ["cpu_features.h"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:platform_port",
    ],
)

cc_library(
    name = "ir_compiler",
    srcs = ["ir_compiler.cc"],
    hdrs = ["ir_compiler.h"],
    deps = [
        ":polynomial_approximations",
        "//xla:util",
        "//xla/service/llvm_ir:llvm_util",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Instrumentation",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:Object",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:TargetParser",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "jit_compiler",
    srcs = ["jit_compiler.cc"],
    hdrs = ["jit_compiler.h"],
    deps = [
        ":compiled_function_library",
        ":contiguous_section_memory_manager",
        ":cpu_features",
        ":ir_compiler",
        "//xla:util",
        "//xla/backends/cpu/runtime:function_library",
        "//xla/service/cpu:orc_jit_memory_mapper",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:ExecutionEngine",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:OrcShared",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:TargetParser",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/profiler/lib:traceme",
        "@tsl//tsl/profiler/lib:traceme_encode",
    ] + if_llvm_aarch64_available([
        "@llvm-project//llvm:AArch64CodeGen",  # fixdeps: keep
    ]) + if_llvm_powerpc_available([
        "@llvm-project//llvm:PowerPCCodeGen",  # fixdeps: keep
    ]) + if_llvm_system_z_available([
        "@llvm-project//llvm:SystemZCodeGen",  # fixdeps: keep
    ]) + if_llvm_x86_available([
        "@llvm-project//llvm:X86CodeGen",  # fixdeps: keep
    ]) + xla_internal(["service/cpu:named_orc_jit_memory_mapper"]),
)

xla_cc_test(
    name = "jit_compiler_test",
    srcs = ["jit_compiler_test.cc"],
    deps = [
        ":jit_compiler",
        "//xla:util",
        "//xla/backends/cpu/runtime:function_library",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:AsmParser",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:OrcShared",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "polynomial_approximations",
    srcs = ["polynomial_approximations.cc"],
    hdrs = ["polynomial_approximations.h"],
    deps = [
        ":vector_ir_builder",
        "//xla:xla_data_proto_cc",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/service/llvm_ir:math_ops",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TransformUtils",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "target_machine_features",
    srcs = ["target_machine_features.cc"],
    hdrs = ["target_machine_features.h"],
    deps = [
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/cpu:alignment",
        "@com_google_absl//absl/container:flat_hash_map",
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:ir_headers",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "vector_ir_builder",
    srcs = ["vector_ir_builder.cc"],
    hdrs = ["vector_ir_builder.h"],
    deps = [
        ":target_machine_features",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/service/llvm_ir:llvm_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "kernel_api_ir_builder",
    srcs = ["kernel_api_ir_builder.cc"],
    hdrs = ["kernel_api_ir_builder.h"],
    deps = [
        "//xla:cpu_function_runtime",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/runtime:buffer_use",
        "//xla/service:buffer_assignment",
        "//xla/service/llvm_ir:ir_array",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:ir_headers",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "llvm_ir_kernel_spec",
    srcs = ["llvm_ir_kernel_spec.cc"],
    hdrs = ["llvm_ir_kernel_spec.h"],
    deps = [
        "//xla/codegen:kernel_spec",
        "//xla/codegen:llvm_ir_kernel_source",
        "//xla/service:buffer_assignment",
        "//xla/stream_executor:launch_dim",
    ],
)

cc_library(
    name = "elemental_kernel_emitter",
    srcs = ["elemental_kernel_emitter.cc"],
    hdrs = ["elemental_kernel_emitter.h"],
    deps = [
        ":kernel_api_ir_builder",
        ":llvm_ir_kernel_spec",
        ":target_machine_features",
        "//xla:shape_util",
        "//xla:util",
        "//xla/codegen:kernel_emitter",
        "//xla/codegen:kernel_spec",
        "//xla/codegen:llvm_ir_kernel_source",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "//xla/service:elemental_ir_emitter",
        "//xla/service/cpu:backend_config_proto_cc",
        "//xla/service/cpu:elemental_ir_emitter",
        "//xla/service/cpu:ir_emitter",
        "//xla/service/cpu:parallel_loop_emitter",
        "//xla/service/cpu:shape_partition",
        "//xla/service/llvm_ir:ir_array",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/service/llvm_ir:loop_emitter",
        "//xla/stream_executor:launch_dim",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:JITLink",
        "@llvm-project//llvm:ir_headers",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "compiled_function_library",
    srcs = ["compiled_function_library.cc"],
    hdrs = ["compiled_function_library.h"],
    deps = [
        "//xla/backends/cpu/runtime:function_library",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:ExecutionEngine",
        "@llvm-project//llvm:OrcJIT",
    ],
)
