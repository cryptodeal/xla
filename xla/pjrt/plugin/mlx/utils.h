/* Copyright 2024 The OpenXLA Authors.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_PJRT_PLUGIN_STABLEHLO_MLX_UTILS_H_
#define XLA_PJRT_PLUGIN_STABLEHLO_MLX_UTILS_H_

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"
#include "mlx/mlx.h"
#include "xla/literal.h"
#include "xla/util.h"

namespace mx = mlx::core;

namespace utils {
namespace dtype {
xla::PrimitiveType asXlaPrimitiveType(mx::Dtype dtype);

absl::StatusOr<mx::Dtype> fromXlaPrimitiveType(xla::PrimitiveType dtype);

absl::StatusOr<mx::Dtype> fromMlirType(mlir::Type type);

int mlirTypeByteWidth(mlir::Type type);
}  // namespace dtype

namespace shape {}  // namespace shape

namespace array {
absl::StatusOr<mx::array> fromHostBuffer(
    const void* data, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    xla::PrimitiveType type);

absl::StatusOr<mx::array> fromHostLiteral(const xla::LiteralSlice& literal);

absl::StatusOr<mx::array> fromDenseElementsAttr(mlir::DenseElementsAttr attr);

absl::StatusOr<mx::array> fromOperand(
    mlir::Value operand, absl::Span<const mx::array> block_args,
    const std::unordered_map<mlir::Operation*, std::vector<mx::array>>&
        transient_buffers,
    const std::unordered_map<const mlir::Value*, mx::array>& init_values);

absl::StatusOr<mx::array> fromOperand(
    mlir::Value operand, absl::Span<const mx::array> block_args,
    const std::unordered_map<mlir::Operation*, std::vector<mx::array>>&
        transient_buffers);
}  // namespace array

void printVector(const std::string& name, const std::vector<int32_t>& vec,
                 bool indent = false);
}  // namespace utils

#endif  // XLA_PJRT_PLUGIN_STABLEHLO_MLX_UTILS_H_