# copybara:uncomment load("@rules_python//python:proto.bzl", "py_proto_library")
load("//xla:xla.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load(
    "//xla/tsl/platform:build_config.bzl",
    "tf_proto_library",
)
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla:internal"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
        "//xla:internal",
    ],
    packages = [
        "//tensorflow/core/tfrt/ifrt/...",
        "//third_party/australis/...",
        "//third_party/gxlang/...",
        "//third_party/openxla_pjrt_plugin/...",
        "//third_party/py/jax/...",
    ],
)

cc_library(
    name = "worker_thread",
    srcs = ["worker_thread.cc"],
    hdrs = ["worker_thread.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/synchronization",
        "@tsl//tsl/platform:env",
    ],
)

cc_library(
    name = "event_pool",
    srcs = ["event_pool.cc"],
    hdrs = ["event_pool.h"],
    deps = [
        "//xla:types",
        "//xla/stream_executor:event",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "semaphore",
    srcs = ["semaphore.cc"],
    hdrs = ["semaphore.h"],
    deps = [
        "//xla:types",
        "@com_google_absl//absl/synchronization",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "semaphore_test",
    srcs = ["semaphore_test.cc"],
    deps = [
        ":semaphore",
        "//xla:test",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "tracked_device_buffer",
    srcs = ["tracked_device_buffer.cc"],
    hdrs = ["tracked_device_buffer.h"],
    deps = [
        ":event_pool",
        ":pjrt_client",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla/service:executable",
        "//xla/service:maybe_owning_device_memory",
        "//xla/service:shaped_buffer",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor:event",
        "//xla/tsl/concurrency:async_value",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/profiler/lib:connected_traceme",
        "@tsl//tsl/profiler/lib:context_types_hdrs",
    ],
)

xla_cc_test(
    name = "tracked_device_buffer_test",
    srcs = ["tracked_device_buffer_test.cc"],
    deps = [
        ":pjrt_client",
        ":pjrt_common",
        ":pjrt_future",
        ":tracked_device_buffer",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:test",
        "//xla:util",
        "//xla/client:client_library",
        "//xla/client:local_client",
        "//xla/service:cpu_plugin",
        "//xla/stream_executor:device_memory_allocator",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "local_device_state",
    srcs = ["local_device_state.cc"],
    hdrs = ["local_device_state.h"],
    deps = [
        ":event_pool",
        ":pjrt_common",
        ":semaphore",
        ":worker_thread",
        "//xla:util",
        "//xla/client:local_client",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/protobuf:error_codes_proto_impl_cc",
        "//xla/tsl/util:env_var",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/profiler/lib:traceme",
    ],
)

cc_library(
    name = "pjrt_api",
    srcs = ["pjrt_api.cc"],
    hdrs = ["pjrt_api.h"],
    deps = [
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_helpers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "pjrt_api_test",
    srcs = ["pjrt_api_test.cc"],
    deps = [
        ":pjrt_api",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_wrapper_impl",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/protobuf:error_codes_proto_impl_cc",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:status_matchers",
    ],
)

cc_library(
    name = "pjrt_client",
    srcs = ["pjrt_client.cc"],
    hdrs = ["pjrt_client.h"],
    visibility = internal_visibility(["//xla:friends"]),
    deps = [
        ":pjrt_common",
        ":pjrt_compiler",
        ":pjrt_device_description",
        ":pjrt_executable",
        ":pjrt_future",
        ":pjrt_layout",
        ":utils",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/pjrt/distributed:key_value_store_interface",
        "//xla/service:computation_placer_hdr",
        "//xla/service:hlo_cost_analysis",
        "//xla/tsl/framework:allocator",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "pjrt_client_test_common",
    testonly = 1,
    srcs = ["pjrt_client_test.cc"],
    hdrs = ["pjrt_client_test.h"],
    deps = [
        ":pjrt_client",
        ":pjrt_compiler",
        "//xla:cpu_function_runtime",
        "//xla:shape_util",
        "//xla:test",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/parser:hlo_parser",
        "//xla/tests:literal_test_util",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:statusor",
    ],
    alwayslink = 1,
)

cc_library(
    name = "pjrt_executable",
    srcs = ["pjrt_executable.cc"],
    hdrs = ["pjrt_executable.h"],
    visibility = internal_visibility([":friends"]),
    deps = [
        ":compile_options_proto_cc",
        ":executable_metadata_proto_cc",
        ":execute_options_proto_cc",
        ":pjrt_common",
        ":pjrt_layout",
        "//xla:shape_layout",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/client:executable_build_options",
        "//xla/ffi:execution_context",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "//xla/service:compiler",
        "//xla/service:computation_layout",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:hlo_proto_cc",
        "//xla/service:hlo_value",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "pjrt_executable_test",
    srcs = ["pjrt_executable_test.cc"],
    deps = [
        ":compile_options_proto_cc",
        ":pjrt_executable",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/client:executable_build_options",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:status_matchers",
    ],
)

cc_library(
    name = "pjrt_device_description",
    hdrs = ["pjrt_device_description.h"],
    deps = [
        ":pjrt_common",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "pjrt_compiler",
    srcs = ["pjrt_compiler.cc"],
    hdrs = ["pjrt_compiler.h"],
    visibility = internal_visibility([":friends"]),
    deps = [
        ":metrics",
        ":pjrt_device_description",
        ":pjrt_executable",
        "//xla/hlo/builder:xla_computation",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@llvm-project//mlir:IR",
        "@tsl//tsl/platform:fingerprint",
    ],
)

xla_cc_test(
    name = "pjrt_compiler_test",
    srcs = ["pjrt_compiler_test.cc"],
    deps = [
        ":metrics",
        ":pjrt_client",
        ":pjrt_compiler",
        ":pjrt_device_description",
        "//xla/hlo/builder:xla_computation",
        "//xla/tsl/lib/monitoring:cell_reader",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:status_matchers",
    ],
)

cc_library(
    name = "pjrt_common",
    hdrs = ["pjrt_common.h"],
    visibility = internal_visibility([":friends"]),
    deps = [
        "//xla/tsl/lib/gtl:int_type",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    visibility = internal_visibility(["//xla:friends"]),
    deps = [
        ":layout_mode",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/client:executable_build_options",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/service:computation_placer",
        "//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "pjrt_layout",
    hdrs = ["pjrt_layout.h"],
    visibility = ["//xla:friends"],
    deps = [
        "//xla:shape_util",
        "//xla/hlo/parser:hlo_parser",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:casts",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "layout_mode",
    srcs = ["layout_mode.cc"],
    hdrs = ["layout_mode.h"],
    visibility = ["//xla:friends"],
    deps = [
        "//xla:shape_util",
        "//xla/hlo/parser:hlo_parser",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "metrics",
    srcs = ["metrics.cc"],
    hdrs = ["metrics.h"],
    deps = [
        "//xla/tsl/lib/monitoring:counter",
        "//xla/tsl/lib/monitoring:gauge",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
    ],
)

tf_proto_library(
    name = "stream_executor_executable_proto",
    srcs = ["stream_executor_executable.proto"],
    protodeps = [
        ":compile_options_proto",
        "//xla:xla_proto",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "stream_executor_executable",
    srcs = ["stream_executor_executable.cc"],
    hdrs = ["stream_executor_executable.h"],
    deps = [
        ":pjrt_common",
        ":pjrt_executable",
        ":stream_executor_executable_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:compiler",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "pjrt_stream_executor_client",
    srcs = ["pjrt_stream_executor_client.cc"],
    hdrs = ["pjrt_stream_executor_client.h"],
    visibility = internal_visibility(["//xla:friends"]),
    deps = [
        ":event_pool",
        ":host_callback",
        ":host_memory_spaces",
        ":local_device_state",
        ":metrics",
        ":mlir_to_hlo",
        ":pjrt_client",
        ":pjrt_common",
        ":pjrt_compiler",
        ":pjrt_device_description",
        ":pjrt_executable",
        ":pjrt_future",
        ":semaphore",
        ":tracked_device_buffer",
        ":transpose",
        ":utils",
        "//xla:executable_run_options",
        "//xla:literal",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/client:executable_build_options",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/pjrt/distributed:protocol_proto_cc",
        "//xla/service:compiler",
        "//xla/service:computation_layout",
        "//xla/service:computation_placer",
        "//xla/service:dump",
        "//xla/service:executable",
        "//xla/service:generic_transfer_manager",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:hlo_proto_cc",
        "//xla/service:maybe_owning_device_memory",
        "//xla/service:shaped_buffer",
        "//xla/service:transfer_manager",
        "//xla/service/gpu:gpu_executable_run_options",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor:stream",
        "//xla/tsl/framework:allocator",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@tsl//tsl/platform:casts",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:fingerprint",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/profiler/lib:connected_traceme",
        "@tsl//tsl/profiler/lib:context_types_hdrs",
        "@tsl//tsl/profiler/lib:traceme",
    ],
)

xla_cc_test(
    name = "pjrt_stream_executor_client_test",
    srcs = ["pjrt_stream_executor_client_test.cc"],
    deps = [
        ":pjrt_client",
        ":pjrt_future",
        ":pjrt_stream_executor_client",
        "//xla:literal",
        "//xla:literal_comparison",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:test",
        "//xla:xla_data_proto_cc",
        "//xla/client:client_library",
        "//xla/hlo/builder:xla_builder",
        "//xla/service:cpu_plugin",
        "//xla/service:platform_util",
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "interpreter_device",
    srcs = ["interpreter_device.cc"],
    hdrs = ["interpreter_device.h"],
    deps = [
        ":pjrt_stream_executor_client",
        "//xla/client:client_library",
        "//xla/service:interpreter_plugin",
        "//xla/service:platform_util",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "mlir_to_hlo",
    srcs = ["mlir_to_hlo.cc"],
    hdrs = ["mlir_to_hlo.h"],
    visibility = [":friends"],
    deps = [
        "//xla:debug_options_flags",
        "//xla:util",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/translate/mhlo_to_hlo:mlir_hlo_to_hlo",
        "//xla/mlir/utils:error_util",
        "//xla/mlir_hlo:hlo_dialect_registration",
        "//xla/mlir_hlo:mhlo_passes",
        "//xla/service/spmd/shardy:constants",
        "//xla/service/spmd/shardy:utils",
        "//xla/service/spmd/shardy/sdy_round_trip:pipelines",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:BytecodeWriter",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MLProgramDialect",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ReconcileUnrealizedCasts",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@shardy//shardy/dialect/sdy/ir:register",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:register",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:stablehlo_passes",
        "@stablehlo//:stablehlo_portable_api",
        "@stablehlo//:stablehlo_serialization",
        "@stablehlo//:version",
        "@tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "mlir_to_hlo_test",
    srcs = ["mlir_to_hlo_test.cc"],
    deps = [
        ":mlir_to_hlo",
        "//xla:test",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
        "@stablehlo//:stablehlo_portable_api",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "pjrt_future",
    srcs = ["pjrt_future.cc"],
    hdrs = ["pjrt_future.h"],
    visibility = internal_visibility([":friends"]),
    deps = [
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/concurrency:ref_count",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "pjrt_future_test",
    srcs = ["pjrt_future_test.cc"],
    deps = [
        ":pjrt_future",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "host_memory_spaces",
    srcs = ["host_memory_spaces.cc"],
    hdrs = ["host_memory_spaces.h"],
    deps = [
        ":pjrt_client",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:fingerprint",
    ],
)

# Transitional forwarding target. Use pjrt/plugin/xla_cpu:xla_cpu_pjrt_client instead.
cc_library(
    name = "tfrt_cpu_pjrt_client",
    hdrs = ["tfrt_cpu_pjrt_client.h"],
    visibility = internal_visibility([
        "//xla:friends",
    ]),
    deps = [
        "//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
    ],
)

cc_library(
    name = "lru_cache",
    hdrs = ["lru_cache.h"],
    deps = [
        "@com_google_absl//absl/container:node_hash_map",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "lru_cache_test",
    srcs = ["lru_cache_test.cc"],
    deps = [
        ":lru_cache",
        "//xla:test",
        "@tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "transpose",
    srcs = [
        "transpose.cc",
        "transpose_kernels.h",
    ],
    hdrs = ["transpose.h"],
    visibility = [":friends"],
    deps = [
        ":lru_cache",
        "//xla:compiler_macros",
        "//xla:ef57",
        "//xla:permutation_util",
        "//xla:util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/profiler/lib:traceme",
    ],
)

xla_cc_test(
    name = "transpose_test",
    srcs = ["transpose_test.cc"],
    deps = [
        ":transpose",
        "//xla:array",
        "//xla:permutation_util",
        "//xla:shape_util",
        "//xla:test",
        "//xla:util",
        "//xla/tsl/protobuf:error_codes_proto_impl_cc",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/numeric:int128",
        "@eigen_archive//:eigen3",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test_benchmark",
        "@tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "pjrt_c_api_client",
    srcs = ["pjrt_c_api_client.cc"],
    hdrs = ["pjrt_c_api_client.h"],
    deps = [
        ":compile_options_proto_cc",
        ":mlir_to_hlo",
        ":pjrt_api",
        ":pjrt_client",
        ":pjrt_common",
        ":pjrt_compiler",
        ":pjrt_device_description",
        ":pjrt_executable",
        ":pjrt_future",
        ":pjrt_layout",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/translate/mhlo_to_hlo:mlir_hlo_to_hlo",
        "//xla/mlir_hlo:mhlo_passes",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_helpers",
        "//xla/pjrt/c:pjrt_c_api_layouts_extension_hdrs",
        "//xla/pjrt/c:pjrt_c_api_memory_descriptions_extension_hdrs",
        "//xla/pjrt/c:pjrt_c_api_profiler_extension_hdrs",
        "//xla/pjrt/c:pjrt_c_api_stream_extension_hdrs",
        "//xla/pjrt/distributed:key_value_store_interface",
        "//xla/service:computation_placer_hdr",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:hlo_proto_cc",
        "//xla/tsl/framework:allocator",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@tsl//tsl/platform:casts",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:fingerprint",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "pjrt_c_api_client_test",
    srcs = ["pjrt_c_api_client_test.cc"],
    tags = [
        "nomsan",
    ],
    deps = [
        ":mlir_to_hlo",
        ":pjrt_api",
        ":pjrt_c_api_client",
        ":pjrt_client",
        ":pjrt_compiler",
        ":pjrt_device_description",
        ":pjrt_executable",
        "//xla:cpu_function_runtime",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla/hlo/builder:xla_builder",
        "//xla/pjrt/c:pjrt_c_api_cpu_internal",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/tests:literal_test_util",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
        "@stablehlo//:version",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "tf_pjrt_client",
    srcs = ["tf_pjrt_client.cc"],
    hdrs = ["tf_pjrt_client.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_client",
        ":pjrt_common",
        ":pjrt_compiler",
        ":pjrt_executable",
        ":pjrt_future",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/service:computation_placer_hdr",
        "//xla/service:hlo_cost_analysis",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:casts",
        "@tsl//tsl/platform:errors",
    ],
)

xla_cc_test(
    name = "tf_pjrt_client_test",
    srcs = ["tf_pjrt_client_test.cc"],
    deps = [
        ":tf_pjrt_client",
        "//xla:literal_util",
        "//xla/hlo/parser:hlo_parser",
        "//xla/pjrt/plugin/xla_cpu:cpu_client_options",
        "//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "host_callback",
    srcs = ["host_callback.cc"],
    hdrs = ["host_callback.h"],
    visibility = internal_visibility([":friends"]),
    deps = [
        ":pjrt_client",
        ":pjrt_executable",
        ":pjrt_future",
        "//xla:shape_util",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "host_callback_test",
    srcs = ["host_callback_test.cc"],
    deps = [
        ":host_callback",
        ":pjrt_client",
        "//xla/tests:literal_test_util",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_proto_library(
    name = "compile_options_proto",
    srcs = ["compile_options.proto"],
    protodeps = [
        "//xla:xla_data_proto",
        "//xla:xla_proto",
        "//xla/stream_executor:device_description_proto",
    ],
    visibility = ["//visibility:public"],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "compile_options_py_pb2",
#     visibility = ["//visibility:public"],
#     deps = [":compile_options_proto"],
# )
# copybara:uncomment_end

tf_proto_library(
    name = "execute_options_proto",
    srcs = ["execute_options.proto"],
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "executable_metadata_proto",
    srcs = ["executable_metadata.proto"],
    protodeps = [
        "//xla/service:hlo_proto",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "exceptions",
    hdrs = ["exceptions.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "status_casters",
    hdrs = ["status_casters.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    visibility = internal_visibility([":friends"]),
    deps = [
        ":exceptions",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@tsl//tsl/platform:macros",
    ],
)
