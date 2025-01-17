load("@tsl//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")
load("@tsl//third_party/gpus:cuda_configure.bzl", "cuda_configure")
load("@tsl//third_party/gpus:rocm_configure.bzl", "rocm_configure")
load("@tsl//third_party/gpus:sycl_configure.bzl", "sycl_configure")
load("@tsl//third_party/tensorrt:tensorrt_configure.bzl", "tensorrt_configure")
load("@tsl//tools/toolchains/remote:configure.bzl", "remote_execution_configure")

def _xla_workspace_impl(mctx):
    cuda_configure(name = "local_config_cuda")
    remote_execution_configure(name = "local_config_remote_execution")
    rocm_configure(name = "local_config_rocm")
    sycl_configure(name = "local_config_sycl")
    tensorrt_configure(name = "local_config_tensorrt")
    tf_http_archive(
        name = "eigen_archive",
        build_file = "@tsl//third_party/eigen3:eigen_archive.BUILD",
        sha256 = "1f4babf536ce8fc2129dbf92ff3be54cd18ffb2171e9eb40edd00f0a045a54fa",
        strip_prefix = "eigen-33d0937c6bdf5ec999939fb17f2a553183d14a74",
        urls = tf_mirror_urls("https://gitlab.com/libeigen/eigen/-/archive/33d0937c6bdf5ec999939fb17f2a553183d14a74/eigen-33d0937c6bdf5ec999939fb17f2a553183d14a74.tar.gz"),
    )
    tf_http_archive(
        name = "farmhash_archive",
        build_file = "@tsl//third_party/farmhash:farmhash.BUILD",
        sha256 = "18392cf0736e1d62ecbb8d695c31496b6507859e8c75541d7ad0ba092dc52115",
        strip_prefix = "farmhash-0d859a811870d10f53a594927d0d0b97573ad06d",
        urls = tf_mirror_urls("https://github.com/google/farmhash/archive/0d859a811870d10f53a594927d0d0b97573ad06d.tar.gz"),
    )
    tf_http_archive(
        name = "ml_dtypes",
        build_file = "//third_party/py/ml_dtypes:ml_dtypes.BUILD",
        link_files = {
            "//third_party/py/ml_dtypes:ml_dtypes.tests.BUILD": "tests/BUILD.bazel",
            "//third_party/py/ml_dtypes:LICENSE": "LICENSE",
        },
        sha256 = "4a03237ef6345e1467a33d126176b9c6a7539b0f60a34b344f39b3c9e8b82438",
        strip_prefix = "ml_dtypes-215c9f02a121e6286662b2efd30546c71054d5e5/ml_dtypes",
        urls = tf_mirror_urls("https://github.com/jax-ml/ml_dtypes/archive/215c9f02a121e6286662b2efd30546c71054d5e5/ml_dtypes-215c9f02a121e6286662b2efd30546c71054d5e5.tar.gz"),
    )
    tf_http_archive(
        name = "com_github_grpc_grpc",
        sha256 = "b956598d8cbe168b5ee717b5dafa56563eb5201a947856a6688bbeac9cac4e1f",
        strip_prefix = "grpc-b54a5b338637f92bfcf4b0bc05e0f57a5fd8fadd",
        system_build_file = "@tsl//third_party/systemlibs:grpc.BUILD",
        patch_file = [
            "@tsl//third_party/grpc:generate_cc_env_fix.patch",
            "@tsl//third_party/grpc:register_go_toolchain.patch",
        ],
        system_link_files = {
            "@tsl//third_party/systemlibs:BUILD": "bazel/BUILD",
            "@tsl//third_party/systemlibs:grpc.BUILD": "src/compiler/BUILD",
            "@tsl//third_party/systemlibs:grpc.bazel.grpc_deps.bzl": "bazel/grpc_deps.bzl",
            "@tsl//third_party/systemlibs:grpc.bazel.grpc_extra_deps.bzl": "bazel/grpc_extra_deps.bzl",
            "@tsl//third_party/systemlibs:grpc.bazel.cc_grpc_library.bzl": "bazel/cc_grpc_library.bzl",
            "@tsl//third_party/systemlibs:grpc.bazel.generate_cc.bzl": "bazel/generate_cc.bzl",
            "@tsl//third_party/systemlibs:grpc.bazel.protobuf.bzl": "bazel/protobuf.bzl",
        },
        urls = tf_mirror_urls("https://github.com/grpc/grpc/archive/b54a5b338637f92bfcf4b0bc05e0f57a5fd8fadd.tar.gz"),
    )
    tf_http_archive(
        name = "com_google_protobuf",
        patch_file = ["@tsl//third_party/protobuf:protobuf.patch"],
        sha256 = "f66073dee0bc159157b0bd7f502d7d1ee0bc76b3c1eac9836927511bdc4b3fc1",
        strip_prefix = "protobuf-3.21.9",
        system_build_file = "@tsl//third_party/systemlibs:protobuf.BUILD",
        system_link_files = {
            "@tsl//third_party/systemlibs:protobuf.bzl": "protobuf.bzl",
            "@tsl//third_party/systemlibs:protobuf_deps.bzl": "protobuf_deps.bzl",
        },
        urls = tf_mirror_urls("https://github.com/protocolbuffers/protobuf/archive/v3.21.9.zip"),
    )
    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

xla_workspace = module_extension(
    implementation = _xla_workspace_impl,
)
