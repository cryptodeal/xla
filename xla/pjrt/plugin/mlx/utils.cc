#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "xla/pjrt/plugin/mlx/utils.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"
#include "mlx/mlx.h"
#include "tsl/platform/statusor.h"
#include "xla/mlir/utils/type_util.h"
#include "xla/shape_util.h"

#include "xla/literal.h"
#include "xla/util.h"

namespace mx = mlx::core;

std::tuple<mx::Shape, mx::Shape, mx::Strides> shapeAndBytes(
    absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides, mx::Dtype dtype) {
  mx::Shape shape(dims.size());
  mx::Strides strides;
  mx::Shape raw_buffer_shape = {shape.size() && !byte_strides.has_value() ? 1
                                                                          : 0};
  for (auto i = 0; i < dims.size(); ++i) {
    shape[i] = static_cast<int32_t>(dims[i]);
    if (byte_strides.has_value()) {
      auto stride_value = byte_strides.value()[i];
      strides.push_back(stride_value / dtype.size());
      if (strides[i] != 0) {
        raw_buffer_shape[0] = std::max(
            raw_buffer_shape[0], static_cast<int32_t>(stride_value) * shape[i]);
      }
    } else {
      raw_buffer_shape[0] *= static_cast<int32_t>(dims[i]);
    }
  }
  if (!byte_strides.has_value()) {
    raw_buffer_shape[0] *= dtype.size();
  } else if (raw_buffer_shape[0] == 0 and
             std::find(dims.begin(), dims.end(), 0) == dims.end()) {
    raw_buffer_shape[0] += dtype.size();
  }
  return std::make_tuple(shape, raw_buffer_shape, strides);
}

namespace utils {

namespace dtype {
xla::PrimitiveType asXlaPrimitiveType(mx::Dtype dtype) {
  switch (dtype) {
    case mx::bool_:
      return xla::PrimitiveType::PRED;
    case mx::uint8:
      return xla::PrimitiveType::U8;
    case mx::uint16:
      return xla::PrimitiveType::U16;
    case mx::uint32:
      return xla::PrimitiveType::U32;
    case mx::uint64:
      return xla::PrimitiveType::U64;
    case mx::int8:
      return xla::PrimitiveType::S8;
    case mx::int16:
      return xla::PrimitiveType::S16;
    case mx::int32:
      return xla::PrimitiveType::S32;
    case mx::int64:
      return xla::PrimitiveType::S64;
    case mx::float16:
      return xla::PrimitiveType::F16;
    case mx::float32:
      return xla::PrimitiveType::F32;
    case mx::bfloat16:
      return xla::PrimitiveType::BF16;
    case mx::complex64:
      return xla::PrimitiveType::C64;
  }
}

absl::StatusOr<mx::Dtype> fromXlaPrimitiveType(xla::PrimitiveType dtype) {
  switch (dtype) {
    case xla::PrimitiveType::PRED:
      return mx::bool_;
    case xla::PrimitiveType::BF16:
      return mx::bfloat16;
    case xla::PrimitiveType::F16:
      return mx::float16;
    case xla::PrimitiveType::F32:
      return mx::float32;
    case xla::PrimitiveType::S8:
      return mx::int8;
    case xla::PrimitiveType::S16:
      return mx::int16;
    case xla::PrimitiveType::S32:
      return mx::int32;
    case xla::PrimitiveType::S64:
      return mx::int64;
    case xla::PrimitiveType::U8:
      return mx::uint8;
    case xla::PrimitiveType::U16:
      return mx::uint16;
    case xla::PrimitiveType::U32:
      return mx::uint32;
    case xla::PrimitiveType::U64:
      return mx::uint64;
    case xla::PrimitiveType::C64:
      return mx::complex64;
    default:
      return absl::InvalidArgumentError("Unsupported type");
  }
}

absl::StatusOr<mx::Dtype> fromMlirType(mlir::Type type) {
  auto primitive_type = xla::ConvertMlirTypeToPrimitiveType(type);
  if (primitive_type == xla::PrimitiveType::PRIMITIVE_TYPE_INVALID) {
    return xla::Internal("Unsupported type: %s",
                         xla::PrimitiveType_Name(primitive_type));
  }
  return fromXlaPrimitiveType(primitive_type);
}

int mlirTypeByteWidth(mlir::Type type) {
  return xla::primitive_util::ByteWidth(
      xla::ConvertMlirTypeToPrimitiveType(type));
}
}  // namespace dtype

namespace array {
absl::StatusOr<mx::array> fromHostBuffer(
    const void* data, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    xla::PrimitiveType type) {
  TF_ASSIGN_OR_RETURN(mx::Dtype dtype, dtype::fromXlaPrimitiveType(type));
  auto [shape, raw_buffer_shape, strides] =
      shapeAndBytes(dims, byte_strides, dtype);
  mx::array typed_view = mx::view(mx::array(reinterpret_cast<const char*>(data),
                                            raw_buffer_shape, mx::uint8),
                                  dtype);
  if (byte_strides.has_value()) {
    auto res = mx::contiguous(mx::as_strided(typed_view, shape, strides, 0));
    res.eval();
    return res;
  } else {
    auto res = mx::reshape(typed_view, shape);
    res.eval();
    return res;
  }
}

absl::StatusOr<mx::array> fromHostLiteral(const xla::LiteralSlice& literal) {
  return fromHostBuffer(literal.untyped_data(), literal.shape().dimensions(),
                        std::nullopt, literal.shape().element_type());
}

template <typename T>
absl::StatusOr<mx::array> fromDenseElementsAttr(mlir::DenseElementsAttr attr) {
  auto attr_type = attr.getType();
  // convert to mlx shape
  std::vector<int32_t> shape(attr_type.getShape().begin(),
                             attr_type.getShape().end());

  // handle splat values
  if (attr.isSplat()) {
    if constexpr (std::is_same<T, xla::bfloat16>::value) {
      return mx::full<mx::bfloat16_t>(
          shape, static_cast<mx::bfloat16_t>(attr.getSplatValue<T>()));
    } else if constexpr (std::is_same<T, xla::half>::value) {
      return mx::full<mx::float16_t>(
          shape, static_cast<mx::float16_t>(attr.getSplatValue<T>()));
    } else if constexpr (std::is_same<T, std::complex<float>>::value) {
      return mx::full<mx::complex64_t>(
          shape, static_cast<mx::complex64_t>(attr.getSplatValue<T>()));
    } else {
      return mx::full<T>(shape, attr.getSplatValue<T>());
    }
  }

  // handle non-splat values that are expanded to fit shape
  auto it = attr.getValues<T>();
  auto buffer = llvm::to_vector(it);
  if (attr.size() != attr.getNumElements()) {
    if constexpr (std::is_same<T, xla::bfloat16>::value) {
      return mx::full(
          shape,
          mx::array(reinterpret_cast<const mx::bfloat16_t*>(buffer.data()),
                    {static_cast<int32_t>(attr.size())}));
    } else if constexpr (std::is_same<T, xla::half>::value) {
      return mx::full(
          shape,
          mx::array(reinterpret_cast<const mx::float16_t*>(buffer.data()),
                    {static_cast<int32_t>(attr.size())}));
    } else if constexpr (std::is_same<T, std::complex<float>>::value) {
      return mx::full(
          shape,
          mx::array(reinterpret_cast<const mx::complex64_t*>(buffer.data()),
                    {static_cast<int32_t>(attr.size())}));
    } else {
      return mx::full(shape, mx::array(buffer.begin(),
                                       {static_cast<int32_t>(attr.size())}));
    }
  }

  if constexpr (std::is_same<T, xla::bfloat16>::value) {
    return mx::array(reinterpret_cast<const mx::bfloat16_t*>(buffer.data()),
                     shape);
  } else if constexpr (std::is_same<T, xla::half>::value) {
    return mx::array(reinterpret_cast<const mx::float16_t*>(buffer.data()),
                     shape);
  } else if constexpr (std::is_same<T, std::complex<float>>::value) {
    return mx::array(reinterpret_cast<const mx::complex64_t*>(buffer.data()),
                     shape);
  } else {
    return mx::array(buffer.begin(), shape);
  }
}

absl::StatusOr<mx::array> fromDenseElementsAttr(mlir::DenseElementsAttr attr) {
  auto element_type =
      xla::ConvertMlirTypeToPrimitiveType(attr.getType().getElementType());
  switch (element_type) {
    case xla::PrimitiveType::PRED:
      return fromDenseElementsAttr<bool>(attr);
    case xla::PrimitiveType::U8:
      return fromDenseElementsAttr<uint8_t>(attr);
    case xla::PrimitiveType::S8:
      return fromDenseElementsAttr<int8_t>(attr);
    case xla::PrimitiveType::U16:
      return fromDenseElementsAttr<uint16_t>(attr);
    case xla::PrimitiveType::S16:
      return fromDenseElementsAttr<int16_t>(attr);
    case xla::PrimitiveType::U32:
      return fromDenseElementsAttr<uint32_t>(attr);
    case xla::PrimitiveType::S32:
      return fromDenseElementsAttr<int32_t>(attr);
    case xla::PrimitiveType::U64:
      return fromDenseElementsAttr<uint64_t>(attr);
    case xla::PrimitiveType::S64:
      return fromDenseElementsAttr<int64_t>(attr);
    case xla::PrimitiveType::F16:
      return fromDenseElementsAttr<xla::half>(attr);
    case xla::PrimitiveType::BF16:
      return fromDenseElementsAttr<xla::bfloat16>(attr);
    case xla::PrimitiveType::F32:
      return fromDenseElementsAttr<float>(attr);
    case xla::PrimitiveType::C64:
      return fromDenseElementsAttr<std::complex<float>>(attr);
    default:
      return xla::Internal("Unsupported type: %s",
                           xla::PrimitiveType_Name(element_type));
  }
}

absl::StatusOr<mx::array> fromOperand(
    mlir::Value operand, absl::Span<const mx::array> block_args,
    const std::unordered_map<mlir::Operation*, std::vector<mx::array>>&
        transient_buffers,
    const std::unordered_map<const mlir::Value*, mx::array>& init_values) {}

absl::StatusOr<mx::array> fromOperand(
    mlir::Value operand, absl::Span<const mx::array> block_args,
    const std::unordered_map<mlir::Operation*, std::vector<mx::array>>&
        transient_buffers) {
  return fromOperand(operand, block_args, transient_buffers, {});
}

}  // namespace array

void printVector(const std::string& name, const std::vector<int32_t>& vec,
                 bool indent) {
  if (indent) {
    std::cout << "\t";
  }
  std::cout << name.c_str() << ": { ";
  for (auto i = 0; i < vec.size(); i++) {
    std::cout << std::to_string(vec[i]);
    if (i < vec.size() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << " }" << std::endl;
}
}  // namespace utils
