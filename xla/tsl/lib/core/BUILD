# Description:
#   Tensor Standard Libraries.
#
#   The libraries in this package are not allowed to have ANY dependencies
#   to other TF components outside of TSL.

load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")
load("//xla/tsl/platform:build_config.bzl", "tsl_cc_test")
load(
    "//xla/tsl/platform:rules_cc.bzl",
    "cc_library",
)

# TODO(rdzhabarov): Tighten visibility after migration is complete.
package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "legacy_lib_core_status_test_util_header",
    srcs = [
        "status_test_util.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        "//tensorflow/core:__pkg__",
        "//xla/tsl/lib/core:__pkg__",
        "//tensorflow/core/lib/core:__pkg__",
    ]),
)

cc_library(
    name = "status_test_util",
    testonly = 1,
    hdrs = ["status_test_util.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@tsl//tsl/platform:status_matchers",
        "@tsl//tsl/platform:test",
    ],
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "bitmap.h",
        "bits.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility(["//tensorflow/core:__pkg__"]),
)

filegroup(
    name = "legacy_lib_core_all_headers",
    srcs = [
        "bitmap.h",
        "bits.h",
        "status_test_util.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/core:__pkg__",
    ]),
)

filegroup(
    name = "legacy_lib_core_all_tests",
    srcs = [
        "bitmap_test.cc",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility(["//tensorflow/core:__pkg__"]),
)

filegroup(
    name = "legacy_lib_core_headers",
    srcs = [
        "bitmap.h",
        "bits.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility(["//tensorflow/core:__pkg__"]),
)

cc_library(
    name = "bitmap",
    srcs = ["bitmap.cc"],
    hdrs = ["bitmap.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@com_google_absl//absl/numeric:bits",
        "@tsl//tsl/platform:logging",
    ],
    alwayslink = 1,
)

cc_library(
    name = "bits",
    hdrs = ["bits.h"],
    deps = [
        "@com_google_absl//absl/numeric:bits",
        "@tsl//tsl/platform:logging",
    ],
)

tsl_cc_test(
    name = "bits_test",
    size = "small",
    srcs = ["bits_test.cc"],
    deps = [
        ":bits",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)
