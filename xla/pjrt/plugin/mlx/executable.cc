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

#include "xla/pjrt/plugin/mlx/executable.h"

#include <memory>
#include <optional>
#include <utility>
#include <unordered_map>
#include <vector>
#include <type_traits>

// TODO(@cryptodeal): might need to update `BUILD`
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlx/mlx.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/reference/Api.h"
#include "stablehlo/transforms/Passes.h"
#include "xla/hlo/translate/stablehlo.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/mlir_hlo/stablehlo_ext/transforms/passes.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/plugin/mlx/buffer.h"
#include "xla/pjrt/plugin/mlx/logging.h"
#include "xla/service/computation_placer.h"
#include "tsl/platform/statusor.h"

#define DEBUG_TYPE "stablehlo-pjrt"

namespace mx = mlx::core;

typedef absl::StatusOr<mx::array> StatusOrArray;
typedef absl::StatusOr<std::vector<mx::array>> StatusOrArrays;
typedef absl::StatusOr<int> StatusOrInt;

namespace mlir::stablehlo {

#define UNIMPLEMENTED(name) \
  xla::Unimplemented("MlirPjrtBuffer::" #name " is not implemented")

using xla::DeviceAssignment;
using xla::PjRtBuffer;
using xla::PjRtClient;
using xla::PjRtDevice;
using xla::PjRtFuture;
using xla::PjRtLoadedExecutable;
using xla::PjRtMemorySpace;

mlir::OwningOpRef<ModuleOp> cloneIntoContext(ModuleOp module,
                                             MLIRContext& context) {
  // Clone the module into the context. MHLO->StableHLO just in case.
  PassManager pm(module->getContext());
  pm.addPass(mhlo::createHloLegalizeToStablehloPass());
  if (failed(pm.run(module))) {
    LOG(ERROR) << "Failed to convert MHLO to StableHLO";
    return nullptr;
  }

  std::string bytecode;
  llvm::raw_string_ostream os(bytecode);
  mlir::BytecodeWriterConfig config;
  mlir::OwningOpRef<mlir::ModuleOp> cloned = module.clone();
  if (mlir::failed(mlir::writeBytecodeToFile(*cloned, os, config))) {
    LOG(ERROR) << "Failed to write bytecode to string\n";
    return nullptr;
  }
  return *parseStablehloModule(bytecode, context);
}

LogicalResult decomposeChloToStablehlo(ModuleOp module) {
  PassManager pm(module->getContext());
  stablehlo_ext::createChloLegalizeToStablehloPipeline(pm);
  if (failed(pm.run(module))) {
    return module->emitError() << "Failed to recompose CHLO";
  }
  return success();
}

std::optional<Operation*> getUnsupportedOp(ModuleOp module) {
  std::optional<Operation*> unsupported_op(std::nullopt);
  module.walk([&unsupported_op](Operation* op) {
    auto cc = llvm::dyn_cast<stablehlo::CustomCallOp>(op);
    if (cc) {
      unsupported_op = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return unsupported_op;
}

LogicalResult runHardwareIndependentOptimizations(ModuleOp module) {
  PassManager pm(module->getContext());
  pm.addNestedPass<func::FuncOp>(
      stablehlo::createStablehloAggressiveFolderPass());
  pm.addNestedPass<func::FuncOp>(
      stablehlo::createStablehloAggressiveSimplificationPass());
  if (failed(pm.run(module))) {
    return module->emitError()
           << "Failed to run hardware independent optimizations";
  }
  return success();
}

mlir::ShapedType getResultType(ModuleOp module, mx::Dtype dtype,
                               SmallVector<int64_t> shape) {
  switch (dtype) {
    case mx::bfloat16:
      return RankedTensorType::get(ArrayRef<int64_t>(shape),
                                   FloatType::getBF16(module.getContext()));
    case mx::float16:
      return RankedTensorType::get(ArrayRef<int64_t>(shape),
                                   FloatType::getF16(module.getContext()));
    case mx::int8:
      return RankedTensorType::get(
          ArrayRef<int64_t>(shape),
          IntegerType::get(module.getContext(), 8,
                           mlir::IntegerType::Signless));
    case mx::uint8:
      return RankedTensorType::get(
          ArrayRef<int64_t>(shape),
          IntegerType::get(module.getContext(), 8,
                           mlir::IntegerType::Unsigned));
    case mx::int16:
      return RankedTensorType::get(
          ArrayRef<int64_t>(shape),
          IntegerType::get(module.getContext(), 16,
                           mlir::IntegerType::Signless));
    case mx::uint16:
      return RankedTensorType::get(
          ArrayRef<int64_t>(shape),
          IntegerType::get(module.getContext(), 16,
                           mlir::IntegerType::Unsigned));
    case mx::int32:
      return RankedTensorType::get(
          ArrayRef<int64_t>(shape),
          IntegerType::get(module.getContext(), 32,
                           mlir::IntegerType::Signless));
    case mx::uint32:
      return RankedTensorType::get(
          ArrayRef<int64_t>(shape),
          IntegerType::get(module.getContext(), 32,
                           mlir::IntegerType::Unsigned));
    case mx::int64:
      return RankedTensorType::get(
          ArrayRef<int64_t>(shape),
          IntegerType::get(module.getContext(), 64,
                           mlir::IntegerType::Signless));
    case mx::uint64:
      return RankedTensorType::get(
          ArrayRef<int64_t>(shape),
          IntegerType::get(module.getContext(), 64,
                           mlir::IntegerType::Unsigned));
    case mx::complex64:
      return RankedTensorType::get(
          ArrayRef<int64_t>(shape),
          ComplexType::get(FloatType::getF32(module.getContext())));
    default:
      return RankedTensorType::get(ArrayRef<int64_t>(shape),
                                   FloatType::getF32(module.getContext()));
  }
}

mx::Dtype dtypeFromType(const mlir::Type& type) {
  return llvm::TypeSwitch<Type, mx::Dtype>(type)
      .Case<IntegerType>([](auto t) {
        switch (t.getWidth()) {
          case 1:
            return mx::bool_;
          case 8:
            return t.isSignless() ? mx::int8 : mx::uint8;
          case 16:
            return t.isSignless() ? mx::int16 : mx::uint16;
          case 32:
            return t.isSignless() ? mx::int32 : mx::uint32;
          default:
            return t.isSignless() ? mx::int64 : mx::uint64;
        }
      })
      .Case<BFloat16Type>([](auto t) { return mx::bfloat16; })
      .Case<Float16Type>([](auto t) { return mx::float16; })
      .Case<ComplexType>([](auto t) { return mx::complex64; })
      // default float32
      .Default([](auto t) { return mx::float32; });
}

// Returns true if the given `ReduceOp` matches
// `zml.Tensor.argMax` op implementation.
bool isArgMaxReduce(stablehlo::ReduceOp& reduce_op) {
  Block& body_block = reduce_op.getBody().front();
  unsigned op_count = 0;
  // TODO (@cryptodeal): This is a lazy verification; maybe check operands
  // for each operation in the block.
  for (auto& op : body_block.getOperations()) {
    switch (op_count) {
      case 0: {
        auto compare_op = llvm::dyn_cast<stablehlo::CompareOp>(op);
        if (compare_op &&
            compare_op.getComparisonDirection() == ComparisonDirection::GT) {
          break;
        } else
          return false;
      }
      case 1: {
        auto compare_op = llvm::dyn_cast<stablehlo::CompareOp>(op);
        if (compare_op &&
            compare_op.getComparisonDirection() == ComparisonDirection::NE) {
          break;
        } else
          return false;
      }
      case 2:
        if (llvm::dyn_cast<stablehlo::OrOp>(op)) {
          break;
        } else
          return false;
      case 3:
        if (llvm::dyn_cast<stablehlo::SelectOp>(op)) {
          break;
        } else
          return false;
      case 4: {
        auto compare_op = llvm::dyn_cast<stablehlo::CompareOp>(op);
        if (compare_op &&
            compare_op.getComparisonDirection() == ComparisonDirection::EQ) {
          break;
        } else
          return false;
      }
      case 5: {
        auto compare_op = llvm::dyn_cast<stablehlo::CompareOp>(op);
        if (compare_op &&
            compare_op.getComparisonDirection() == ComparisonDirection::LT) {
          break;
        } else
          return false;
      }
      case 6:
        if (llvm::dyn_cast<stablehlo::AndOp>(op)) {
          break;
        } else
          return false;
      case 7:
        if (llvm::dyn_cast<stablehlo::OrOp>(op)) {
          break;
        } else
          return false;
      case 8:
        if (llvm::dyn_cast<stablehlo::SelectOp>(op)) {
          break;
        } else
          return false;
      case 9:
        if (llvm::dyn_cast<stablehlo::ReturnOp>(op)) {
          break;
        } else
          return false;
      default:
        return false;
    }
    op_count++;
  }
  return true;
}

// Returns true if the given `ReduceOp` matches
// `zml.Tensor.sum` op implementation.
bool isSumReduce(stablehlo::ReduceOp& reduce_op) {
  Block& body_block = reduce_op.getBody().front();
  unsigned op_count = 0;
  // TODO (@cryptodeal): This is a lazy verification; maybe check operands
  // for each operation in the block.
  for (auto& op : body_block.getOperations()) {
    switch (op_count) {
      case 0:
        if (llvm::dyn_cast<stablehlo::AddOp>(op)) {
          break;
        } else
          return false;
      case 1:
        if (llvm::dyn_cast<stablehlo::ReturnOp>(op)) {
          break;
        } else
          return false;
    }
    op_count++;
  }
  return true;
}

// Returns true if the given `ReduceOp` matches
// `zml.Tensor.max` op implementation.
bool isMaxReduce(stablehlo::ReduceOp& reduce_op) {
  Block& body_block = reduce_op.getBody().front();
  unsigned op_count = 0;
  // TODO (@cryptodeal): This is a lazy verification; maybe check operands
  // for each operation in the block.
  for (auto& op : body_block.getOperations()) {
    switch (op_count) {
      case 0:
        if (llvm::dyn_cast<stablehlo::MaxOp>(op)) {
          break;
        } else
          return false;
      case 1:
        if (llvm::dyn_cast<stablehlo::ReturnOp>(op)) {
          break;
        } else
          return false;
    }
    op_count++;
  }
  return true;
}

// Returns true if the given `ReduceOp` matches
// `zml.Tensor.max` op implementation.
bool isMinReduce(stablehlo::ReduceOp& reduce_op) {
  Block& body_block = reduce_op.getBody().front();
  unsigned op_count = 0;
  // TODO (@cryptodeal): This is a lazy verification; maybe check operands
  // for each operation in the block.
  for (auto& op : body_block.getOperations()) {
    switch (op_count) {
      case 0:
        if (llvm::dyn_cast<stablehlo::MinOp>(op)) {
          break;
        } else
          return false;
      case 1:
        if (llvm::dyn_cast<stablehlo::ReturnOp>(op)) {
          break;
        } else
          return false;
    }
    op_count++;
  }
  return true;
}

// Returns true if the given `ReduceOp` matches
// `zml.Tensor.any` op implementation.
bool isAnyReduce(stablehlo::ReduceOp& reduce_op) {
  Block& body_block = reduce_op.getBody().front();
  unsigned op_count = 0;
  // TODO (@cryptodeal): This is a lazy verification; maybe check operands
  // for each operation in the block.
  for (auto& op : body_block.getOperations()) {
    switch (op_count) {
      case 0:
        if (llvm::dyn_cast<stablehlo::OrOp>(op)) {
          break;
        } else
          return false;
      case 1:
        if (llvm::dyn_cast<stablehlo::ReturnOp>(op)) {
          break;
        } else
          return false;
    }
    op_count++;
  }
  return true;
}

absl::StatusOr<mx::array> getOperandArray(
    const Value& operand, const std::vector<mx::array> block_args,
    const std::unordered_map<Operation*, std::vector<mx::array>>&
        transient_buffers,
    const std::unordered_map<const Value*, mx::array>& init_values) {
  // init values are used for handling custom logic
  // (e.g. `stablehlo::ReduceOp`); if match is found, use
  // this value.
  if (auto search = init_values.find(&operand); search != init_values.end()) {
    return search->second;
  }
  // check if operand is the result of a previous operation
  if (auto defining_op = operand.getDefiningOp()) {
    if (auto search = transient_buffers.find(defining_op);
        search != transient_buffers.end()) {
      switch (search->second.size()) {
        case 0:
          return absl::InternalError("Failed to find array for operand");
        case 1:
          return search->second[0];
        default:
          std::optional<mx::array> array;
          for (auto i = 0; i < defining_op->getNumResults(); i++) {
            Value maybe_res = defining_op->getResult(i);
            if (maybe_res == operand) {
              array = search->second[i];
              break;
            }
          }
          if (array.has_value()) {
            return array.value();
          } else
            return absl::InternalError("Failed to find array for operand");
      }
    } else
      return absl::InternalError("Failed to find array for operand");

  } else {
    // if block arguments contains the arg at the given index,
    // push to args; else return early
    return block_args[operand.cast<BlockArgument>().getArgNumber()];
  }
}

absl::StatusOr<mx::array> getOperandArray(
    const Value& operand, const std::vector<mx::array> block_args,
    const std::unordered_map<Operation*, std::vector<mx::array>>&
        transient_buffers) {
  return getOperandArray(operand, block_args, transient_buffers, {});
}

std::vector<int32_t> getMlxShape(const mlir::ShapedType& dense_elements_type) {
  std::vector<int32_t> shape;
  if (dense_elements_type.hasRank()) {
    for (auto dim : dense_elements_type.getShape()) {
      shape.push_back(static_cast<int32_t>(dim));
    }
  }
  return shape;
}

absl::StatusOr<int> getByteWidth(const mlir::Type& t) {
  return llvm::TypeSwitch<Type, StatusOrInt>(t)
      .Case<IntegerType>([](auto t) -> StatusOrInt {
        auto bit_width = t.getWidth();
        if (bit_width < 8 || bit_width % 8 != 0) {
          return absl::UnimplementedError(
              "RngBitGeneratorOp: element type width < 8 bits "
              "or not a multiple of 8 bits");
        } else
          return bit_width / 8;
      })
      .Case<BFloat16Type>([](auto t) -> StatusOrInt { return 2; })
      .Case<Float16Type>([](auto t) -> StatusOrInt { return 2; })
      .Case<Float32Type>([](auto t) -> StatusOrInt { return 4; })
      .Default([](auto t) -> StatusOrInt {
        return absl::UnimplementedError("Unsupported element type");
      });
}

template <typename T>
mx::array denseElementsArray(const mlir::DenseElementsAttr& attr) {
  auto attr_type = attr.getType();
  // convert to mlx shape
  std::vector<int32_t> shape = getMlxShape(attr_type);
  auto it = attr.getValues<T>();
  std::vector<T> buffer(it.begin(), it.end());
  // TODO (@cryptodeal): handle complex64
  if constexpr (std::is_same<T, xla::bfloat16>::value) {
    return attr.isSplat()
               ? mx::full<mx::bfloat16_t>(
                     shape, static_cast<mx::bfloat16_t>(buffer[0]))
               : mx::array(
                     reinterpret_cast<const mx::bfloat16_t*>(buffer.data()),
                     shape);
  } else if constexpr (std::is_same<T, xla::half>::value) {
    return attr.isSplat()
               ? mx::full<mx::float16_t>(shape,
                                         static_cast<mx::float16_t>(buffer[0]))
               : mx::array(
                     reinterpret_cast<const mx::float16_t*>(buffer.data()),
                     shape);
  } else if constexpr (std::is_same<T, std::complex<float>>::value) {
    return attr.isSplat()
               ? mx::full<mx::complex64_t>(
                     shape,
                     reinterpret_cast<const mx::complex64_t*>(buffer.data())[0])
               : mx::array(
                     reinterpret_cast<const mx::complex64_t*>(buffer.data()),
                     shape);
  } else {
    return attr.isSplat() ? mx::full<T>(shape, buffer[0])
                          : mx::array(buffer.begin(), shape);
  }
}

// TODO(@cryptodeal): `Compile` must verify there are no unsupported types.
absl::StatusOr<mx::array> getDenseElementsArray(
    const mlir::DenseElementsAttr& attr) {
  return llvm::TypeSwitch<Type, StatusOrArray>(attr.getType().getElementType())
      // handle integer types
      .Case<IntegerType>([&attr](auto type) -> StatusOrArray {
        switch (type.getWidth()) {
          // handle bool
          case 1:
            return denseElementsArray<bool>(attr);
          case 8:
            return type.isSignless() ? denseElementsArray<int8_t>(attr)
                                     : denseElementsArray<uint8_t>(attr);
          case 16:
            return type.isSignless() ? denseElementsArray<int16_t>(attr)
                                     : denseElementsArray<uint16_t>(attr);
          case 32:
            return type.isSignless() ? denseElementsArray<int32_t>(attr)
                                     : denseElementsArray<uint32_t>(attr);
          // default is 64 bit width
          default:
            return type.isSignless() ? denseElementsArray<int64_t>(attr)
                                     : denseElementsArray<uint64_t>(attr);
        }
      })
      .Case<BFloat16Type>([&attr](auto type) -> StatusOrArray {
        return denseElementsArray<xla::bfloat16>(attr);
      })
      .Case<Float16Type>([&attr](auto type) -> StatusOrArray {
        return denseElementsArray<xla::half>(attr);
      })
      .Case<ComplexType>([&attr](auto type) -> StatusOrArray {
        return denseElementsArray<std::complex<float>>(attr);
      })
      .Case<Float32Type>([&attr](auto type) -> StatusOrArray {
        return denseElementsArray<float>(attr);
      })
      .Default([](auto type) -> StatusOrArray {
        return absl::UnimplementedError("Unsupported datatype");
      });
}

absl::StatusOr<std::vector<mx::array>> evalFunc(
    ModuleOp& module, func::FuncOp& func, const std::vector<mx::array>& args) {
  std::unordered_map<Operation*, std::vector<mx::array>> transient_buffers;
  Operation* result_op = nullptr;
  for (Operation& op : func.front().getOperations()) {
    // op.dump();

    // switch on the operation type
    auto maybe_result =
        llvm::TypeSwitch<Operation*, StatusOrArrays>(&op)
            // TODO(@cryptodeal): handle `func` namespace ops
            .Case<func::ReturnOp>([&args, &transient_buffers,
                                   &result_op](auto op) -> StatusOrArrays {
              std::vector<mx::array> res;
              for (Value val : op.getOperands()) {
                TF_ASSIGN_OR_RETURN(
                    auto result_array,
                    getOperandArray(val, args, transient_buffers));
                res.emplace_back(result_array);
              }
              result_op = op;
              return res;
            })
            .Case<func::CallOp>([&args, &module, &transient_buffers](
                                    auto op) -> StatusOrArrays {
              auto callee =
                  module.lookupSymbol<mlir::func::FuncOp>(op.getCallee());
              std::vector<mx::array> operands;
              for (Value val : op.getOperands()) {
                TF_ASSIGN_OR_RETURN(
                    auto operand_array,
                    getOperandArray(val, args, transient_buffers));
                operands.emplace_back(operand_array);
              }
              return evalFunc(module, callee, operands);
            })
            // Handle StableHLO nullary ops
            .Case<stablehlo::ConstantOp>(
                [&transient_buffers](auto op) -> StatusOrArrays {
                  TF_ASSIGN_OR_RETURN(
                      auto value,
                      getDenseElementsArray(
                          mlir::cast<mlir::DenseElementsAttr>(op.getValue())));
                  return std::vector<mx::array>{value};
                })
            .Case<stablehlo::IotaOp>([](auto op) -> StatusOrArrays {
              auto result_type = op.getResult().getType();
              auto shape = getMlxShape(mlir::cast<ShapedType>(result_type));
              auto iota_dimension = op.getIotaDimension();
              std::vector<int32_t> dimensions;
              for (auto i = 0; i < dimensions.size(); i++) {
                dimensions.push_back(i != iota_dimension ? 1 : shape[i]);
              }
              auto init_arange =
                  mx::arange(static_cast<double>(shape[iota_dimension]),
                             dtypeFromType(result_type.getElementType()));
              return std::vector<mx::array>{mx::broadcast_to(
                  dimensions.size() ? mx::reshape(init_arange, dimensions)
                                    : init_arange,
                  shape)};
            })
            // .Case<stablehlo::DynamicIotaOp>([](auto op) {})
            /*
              .Case<stablehlo::CreateTokenOp>([](auto op) {})
              Deprecated see:
                https://github.com/openxla/stablehlo/issues/2340
                https://github.com/openxla/stablehlo/pull/2283
            */

            // Handle StableHLO unary elementwise op
            .Case<stablehlo::AbsOp>(
                [&args, &transient_buffers](auto op) -> StatusOrArrays {
                  TF_ASSIGN_OR_RETURN(auto operand,
                                      getOperandArray(op.getOperand(), args,
                                                      transient_buffers));
                  return std::vector<mx::array>{mx::abs(operand)};
                })
            .Case<stablehlo::CbrtOp>([&args, &transient_buffers](
                                         auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(
                  auto operand,
                  getOperandArray(op.getOperand(), args, transient_buffers));
              return std::vector<mx::array>{mx::power(
                  operand,
                  mx::full<float>(operand.shape(), 1 / 3, operand.dtype()))};
            })
            .Case<stablehlo::CeilOp>(
                [&args, &transient_buffers](auto op) -> StatusOrArrays {
                  TF_ASSIGN_OR_RETURN(auto operand,
                                      getOperandArray(op.getOperand(), args,
                                                      transient_buffers));
                  return std::vector<mx::array>{mx::ceil(operand)};
                })
            .Case<stablehlo::ConvertOp>([&args, &transient_buffers](
                                            auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(
                  auto operand,
                  getOperandArray(op.getOperand(), args, transient_buffers));
              return std::vector<mx::array>{mx::astype(
                  operand,
                  dtypeFromType(op.getResult().getType().getElementType()))};
            })
            // .Case<stablehlo::ClzOp>([](auto op) {})
            .Case<stablehlo::CosineOp>(
                [&args, &transient_buffers](auto op) -> StatusOrArrays {
                  TF_ASSIGN_OR_RETURN(auto operand,
                                      getOperandArray(op.getOperand(), args,
                                                      transient_buffers));
                  return std::vector<mx::array>{mx::cos(operand)};
                })
            .Case<stablehlo::ExpOp>(
                [&args, &transient_buffers](auto op) -> StatusOrArrays {
                  TF_ASSIGN_OR_RETURN(auto operand,
                                      getOperandArray(op.getOperand(), args,
                                                      transient_buffers));
                  return std::vector<mx::array>{mx::exp(operand)};
                })
            .Case<stablehlo::Expm1Op>(
                [&args, &transient_buffers](auto op) -> StatusOrArrays {
                  TF_ASSIGN_OR_RETURN(auto operand,
                                      getOperandArray(op.getOperand(), args,
                                                      transient_buffers));
                  return std::vector<mx::array>{mx::expm1(operand)};
                })
            .Case<stablehlo::FloorOp>(
                [&args, &transient_buffers](auto op) -> StatusOrArrays {
                  TF_ASSIGN_OR_RETURN(auto operand,
                                      getOperandArray(op.getOperand(), args,
                                                      transient_buffers));
                  return std::vector<mx::array>{mx::floor(operand)};
                })
            .Case<stablehlo::ImagOp>(
                [&args, &transient_buffers](auto op) -> StatusOrArrays {
                  TF_ASSIGN_OR_RETURN(auto operand,
                                      getOperandArray(op.getOperand(), args,
                                                      transient_buffers));
                  return std::vector<mx::array>{mx::imag(operand)};
                })
            .Case<stablehlo::IsFiniteOp>(
                [&args, &transient_buffers](auto op) -> StatusOrArrays {
                  TF_ASSIGN_OR_RETURN(auto operand,
                                      getOperandArray(op.getOperand(), args,
                                                      transient_buffers));
                  return std::vector<mx::array>{mx::isfinite(operand)};
                })
            .Case<stablehlo::LogOp>(
                [&args, &transient_buffers](auto op) -> StatusOrArrays {
                  TF_ASSIGN_OR_RETURN(auto operand,
                                      getOperandArray(op.getOperand(), args,
                                                      transient_buffers));
                  return std::vector<mx::array>{mx::log(operand)};
                })
            .Case<stablehlo::Log1pOp>(
                [&args, &transient_buffers](auto op) -> StatusOrArrays {
                  TF_ASSIGN_OR_RETURN(auto operand,
                                      getOperandArray(op.getOperand(), args,
                                                      transient_buffers));
                  return std::vector<mx::array>{mx::log1p(operand)};
                })
            .Case<stablehlo::LogisticOp>(
                [&args, &transient_buffers](auto op) -> StatusOrArrays {
                  TF_ASSIGN_OR_RETURN(auto operand,
                                      getOperandArray(op.getOperand(), args,
                                                      transient_buffers));
                  return std::vector<mx::array>{mx::sigmoid(operand)};
                })
            .Case<stablehlo::NotOp>(
                [&args, &transient_buffers](auto op) -> StatusOrArrays {
                  TF_ASSIGN_OR_RETURN(auto operand,
                                      getOperandArray(op.getOperand(), args,
                                                      transient_buffers));
                  return std::vector<mx::array>{mx::logical_not(operand)};
                })
            .Case<stablehlo::NegOp>(
                [&args, &transient_buffers](auto op) -> StatusOrArrays {
                  TF_ASSIGN_OR_RETURN(auto operand,
                                      getOperandArray(op.getOperand(), args,
                                                      transient_buffers));
                  return std::vector<mx::array>{mx::negative(operand)};
                })
            // .Case<stablehlo::PopulationCountOp>([](auto op) {})
            .Case<stablehlo::RealOp>(
                [&args, &transient_buffers](auto op) -> StatusOrArrays {
                  TF_ASSIGN_OR_RETURN(auto operand,
                                      getOperandArray(op.getOperand(), args,
                                                      transient_buffers));
                  return std::vector<mx::array>{mx::real(operand)};
                })
            // `stablehlo::RoundOp` does not match with the mlx metal
            // implementation .Case<stablehlo::RoundOp>([](auto op) {})
            .Case<stablehlo::RoundNearestEvenOp>(
                [&args, &transient_buffers](auto op) -> StatusOrArrays {
                  TF_ASSIGN_OR_RETURN(auto operand,
                                      getOperandArray(op.getOperand(), args,
                                                      transient_buffers));
                  return std::vector<mx::array>{mx::round(operand)};
                })
            .Case<stablehlo::RsqrtOp>(
                [&args, &transient_buffers](auto op) -> StatusOrArrays {
                  TF_ASSIGN_OR_RETURN(auto operand,
                                      getOperandArray(op.getOperand(), args,
                                                      transient_buffers));
                  return std::vector<mx::array>{mx::rsqrt(operand)};
                })
            .Case<stablehlo::SignOp>(
                [&args, &transient_buffers](auto op) -> StatusOrArrays {
                  TF_ASSIGN_OR_RETURN(auto operand,
                                      getOperandArray(op.getOperand(), args,
                                                      transient_buffers));
                  return std::vector<mx::array>{mx::sign(operand)};
                })
            .Case<stablehlo::SineOp>(
                [&args, &transient_buffers](auto op) -> StatusOrArrays {
                  TF_ASSIGN_OR_RETURN(auto operand,
                                      getOperandArray(op.getOperand(), args,
                                                      transient_buffers));
                  return std::vector<mx::array>{mx::sin(operand)};
                })
            .Case<stablehlo::SqrtOp>(
                [&args, &transient_buffers](auto op) -> StatusOrArrays {
                  TF_ASSIGN_OR_RETURN(auto operand,
                                      getOperandArray(op.getOperand(), args,
                                                      transient_buffers));
                  return std::vector<mx::array>{mx::sqrt(operand)};
                })
            .Case<stablehlo::TanOp>(
                [&args, &transient_buffers](auto op) -> StatusOrArrays {
                  TF_ASSIGN_OR_RETURN(auto operand,
                                      getOperandArray(op.getOperand(), args,
                                                      transient_buffers));
                  return std::vector<mx::array>{mx::tan(operand)};
                })
            .Case<stablehlo::TanhOp>(
                [&args, &transient_buffers](auto op) -> StatusOrArrays {
                  TF_ASSIGN_OR_RETURN(auto operand,
                                      getOperandArray(op.getOperand(), args,
                                                      transient_buffers));
                  return std::vector<mx::array>{mx::tanh(operand)};
                })

            // Handle StableHLO binary elementwise ops
            .Case<stablehlo::AddOp>([&args, &transient_buffers](
                                        auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto lhs, getOperandArray(op.getLhs(), args,
                                                            transient_buffers));
              TF_ASSIGN_OR_RETURN(auto rhs, getOperandArray(op.getRhs(), args,
                                                            transient_buffers));
              return std::vector<mx::array>{mx::add(lhs, rhs)};
            })
            .Case<stablehlo::Atan2Op>([&args, &transient_buffers](
                                          auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto lhs, getOperandArray(op.getLhs(), args,
                                                            transient_buffers));
              TF_ASSIGN_OR_RETURN(auto rhs, getOperandArray(op.getRhs(), args,
                                                            transient_buffers));
              return std::vector<mx::array>{mx::arctan2(lhs, rhs)};
            })
            // TODO(@cryptodeal): implement complex op
            // .Case<stablehlo::ComplexOp>([&transient_buffers, &operands](auto
            // op) {})
            .Case<stablehlo::DivOp>([&args, &transient_buffers](
                                        auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto lhs, getOperandArray(op.getLhs(), args,
                                                            transient_buffers));
              TF_ASSIGN_OR_RETURN(auto rhs, getOperandArray(op.getRhs(), args,
                                                            transient_buffers));
              return std::vector<mx::array>{mx::divide(lhs, rhs)};
            })
            .Case<stablehlo::MaxOp>([&args, &transient_buffers](
                                        auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto lhs, getOperandArray(op.getLhs(), args,
                                                            transient_buffers));
              TF_ASSIGN_OR_RETURN(auto rhs, getOperandArray(op.getRhs(), args,
                                                            transient_buffers));
              return std::vector<mx::array>{mx::maximum(lhs, rhs)};
            })
            .Case<stablehlo::MinOp>([&args, &transient_buffers](
                                        auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto lhs, getOperandArray(op.getLhs(), args,
                                                            transient_buffers));
              TF_ASSIGN_OR_RETURN(auto rhs, getOperandArray(op.getRhs(), args,
                                                            transient_buffers));
              return std::vector<mx::array>{mx::minimum(lhs, rhs)};
            })
            .Case<stablehlo::MulOp>([&args, &transient_buffers](
                                        auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto lhs, getOperandArray(op.getLhs(), args,
                                                            transient_buffers));
              TF_ASSIGN_OR_RETURN(auto rhs, getOperandArray(op.getRhs(), args,
                                                            transient_buffers));
              return std::vector<mx::array>{mx::multiply(lhs, rhs)};
            })
            .Case<stablehlo::PowOp>([&args, &transient_buffers](
                                        auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto lhs, getOperandArray(op.getLhs(), args,
                                                            transient_buffers));
              TF_ASSIGN_OR_RETURN(auto rhs, getOperandArray(op.getRhs(), args,
                                                            transient_buffers));
              return std::vector<mx::array>{mx::power(lhs, rhs)};
            })
            .Case<stablehlo::RemOp>([&args, &transient_buffers](
                                        auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto lhs, getOperandArray(op.getLhs(), args,
                                                            transient_buffers));
              TF_ASSIGN_OR_RETURN(auto rhs, getOperandArray(op.getRhs(), args,
                                                            transient_buffers));
              return std::vector<mx::array>{mx::remainder(lhs, rhs)};
            })
            .Case<stablehlo::ShiftLeftOp>([&args, &transient_buffers](
                                              auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto lhs, getOperandArray(op.getLhs(), args,
                                                            transient_buffers));
              TF_ASSIGN_OR_RETURN(auto rhs, getOperandArray(op.getRhs(), args,
                                                            transient_buffers));
              return std::vector<mx::array>{mx::left_shift(lhs, rhs)};
            })
            /**
             * Per Metal Spec:
             * For the right-shift operator, if E1 has an unsigned type or if
             * E1 has a signed type and a nonnegative value, the vacated bits
             * are filled with zeros. If E1 has a signed type and a negative
             * value, the vacated bits are filled with ones.
             */
            .Case<stablehlo::ShiftRightArithmeticOp>(
                [&args, &transient_buffers](auto op) -> StatusOrArrays {
                  TF_ASSIGN_OR_RETURN(
                      auto lhs,
                      getOperandArray(op.getLhs(), args, transient_buffers));
                  auto lhs_dtype = lhs.dtype();
                  TF_ASSIGN_OR_RETURN(
                      auto rhs,
                      getOperandArray(op.getRhs(), args, transient_buffers));
                  auto target_dtype = lhs_dtype;
                  switch (lhs_dtype.size()) {
                    case 1:
                      target_dtype = mx::int8;
                      break;
                    case 2:
                      target_dtype = mx::int16;
                      break;
                    case 4:
                      target_dtype = mx::int32;
                      break;
                    case 8:
                      target_dtype = mx::int64;
                      break;
                    default:
                      break;
                  }
                  return std::vector<mx::array>{
                      mx::view(mx::right_shift(mx::view(lhs, target_dtype),
                                               mx::astype(rhs, target_dtype)),
                               lhs_dtype)};
                })
            // Ensures that we bitcast to `uint` type before performing the
            // right shift. Should ensure that vacated bits are zero populated.
            .Case<stablehlo::ShiftRightLogicalOp>(
                [&args, &transient_buffers](auto op) -> StatusOrArrays {
                  TF_ASSIGN_OR_RETURN(
                      auto lhs,
                      getOperandArray(op.getLhs(), args, transient_buffers));
                  auto lhs_dtype = lhs.dtype();
                  TF_ASSIGN_OR_RETURN(
                      auto rhs,
                      getOperandArray(op.getRhs(), args, transient_buffers));
                  auto target_dtype = lhs_dtype;
                  switch (lhs_dtype.size()) {
                    case 1:
                      target_dtype = mx::uint8;
                      break;
                    case 2:
                      target_dtype = mx::uint16;
                      break;
                    case 4:
                      target_dtype = mx::uint32;
                      break;
                    case 8:
                      target_dtype = mx::uint64;
                      break;
                    default:
                      break;
                  }
                  return std::vector<mx::array>{
                      mx::view(mx::right_shift(mx::view(lhs, target_dtype),
                                               mx::astype(rhs, target_dtype)),
                               lhs_dtype)};
                })
            .Case<stablehlo::SubtractOp>(
                [&args, &transient_buffers](auto op) -> StatusOrArrays {
                  TF_ASSIGN_OR_RETURN(
                      mx::array lhs,
                      getOperandArray(op.getLhs(), args, transient_buffers));
                  TF_ASSIGN_OR_RETURN(
                      mx::array rhs,
                      getOperandArray(op.getRhs(), args, transient_buffers));
                  return std::vector<mx::array>{mx::subtract(lhs, rhs)};
                })

            // Handle StableHLO binary logical elementwise ops
            .Case<stablehlo::AndOp>([&args, &transient_buffers](
                                        auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto lhs, getOperandArray(op.getLhs(), args,
                                                            transient_buffers));
              TF_ASSIGN_OR_RETURN(auto rhs, getOperandArray(op.getRhs(), args,
                                                            transient_buffers));
              if (mx::kindof(lhs.dtype()) == mx::Dtype::Kind::b) {
                return std::vector<mx::array>{mx::logical_and(lhs, rhs)};
              } else {
                return std::vector<mx::array>{mx::bitwise_and(lhs, rhs)};
              }
            })
            .Case<stablehlo::OrOp>([&args, &transient_buffers](
                                       auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto lhs, getOperandArray(op.getLhs(), args,
                                                            transient_buffers));
              TF_ASSIGN_OR_RETURN(auto rhs, getOperandArray(op.getRhs(), args,
                                                            transient_buffers));

              if (mx::kindof(lhs.dtype()) == mx::Dtype::Kind::b) {
                return std::vector<mx::array>{mx::logical_or(lhs, rhs)};
              } else {
                return std::vector<mx::array>{mx::bitwise_or(lhs, rhs)};
              }
            })
            .Case<stablehlo::XorOp>([&args, &transient_buffers](
                                        auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto lhs, getOperandArray(op.getLhs(), args,
                                                            transient_buffers));
              TF_ASSIGN_OR_RETURN(auto rhs, getOperandArray(op.getRhs(), args,
                                                            transient_buffers));
              return std::vector<mx::array>{mx::bitwise_xor(lhs, rhs)};
            })

            // TODO(@cryptodeal): probably don't need to implement all
            // of the following ops, but listing all for reference

            // Handle StableHLO communication ops
            // .Case<stablehlo::InfeedOp>([](auto op) {})
            // .Case<stablehlo::OutfeedOp>([](auto op) {})
            // .Case<stablehlo::SendOp>([](auto op) {})
            // .Case<stablehlo::RecvOp>([](auto op) {})

            // Handle StableHLO parallelism related ops
            // .Case<stablehlo::ReplicaIdOp>([](auto op) {})
            // .Case<stablehlo::PartitionIdOp>([](auto op) {})

            // Handle StableHLO control flow ops
            // .Case<stablehlo::AfterAllOp>([](auto op) {})
            // .Case<stablehlo::IfOp>([](auto op) {})
            // .Case<stablehlo::CaseOp>([](auto op) {})
            // .Case<stablehlo::WhileOp>([](auto op) {})
            // .Case<stablehlo::AllGatherOp>([](auto op) {})
            // .Case<stablehlo::AllReduceOp>([](auto op) {})
            // .Case<stablehlo::ReduceScatterOp>([](auto op) {})
            // .Case<stablehlo::AllToAllOp>([](auto op) {})

            // TODO (@cryptodeal): `stablehlo::ReduceOp` currently only supports
            // specific well defined use cases in zml. Longer term, should
            // support custom use cases of `zml.ops.reduce`.
            .Case<stablehlo::ReduceOp>([&args, &transient_buffers](
                                           auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(
                  auto operand,
                  getOperandArray(op.getOperand(0), args, transient_buffers));
              if (isArgMaxReduce(op)) {
                // std::cout << "ArgMaxReduceOp\n" << std::endl;
                auto axis = static_cast<int>(op.getDimensions()[0]);
                auto indices = mx::argmax(operand, axis);
                return std::vector<mx::array>{
                    mx::take(operand, indices, axis),
                    mx::astype(indices,
                               dtypeFromType(mlir::cast<ShapedType>(
                                                 op.getResults()[1].getType())
                                                 .getElementType()))};
              }
              if (isSumReduce(op)) {
                return std::vector<mx::array>{
                    mx::sum(operand, static_cast<int>(op.getDimensions()[0]))};
              }
              if (isMaxReduce(op)) {
                return std::vector<mx::array>{
                    mx::max(operand, static_cast<int>(op.getDimensions()[0]))};
              }
              if (isMinReduce(op)) {
                return std::vector<mx::array>{
                    mx::min(operand, static_cast<int>(op.getDimensions()[0]))};
              }
              if (isAnyReduce(op)) {
                return std::vector<mx::array>{
                    mx::any(operand, static_cast<int>(op.getDimensions()[0]))};
              }

              return absl::UnimplementedError(
                  "Unsupported custom `reduce` operation");
            })

            // Handle StableHLO tuple ops
            // .Case<stablehlo::GetTupleElementOp>([](auto op) {})
            // .Case<stablehlo::TupleOp>([](auto op) {})

            .Case<stablehlo::CompareOp>([&args, &transient_buffers](
                                            auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto lhs, getOperandArray(op.getLhs(), args,
                                                            transient_buffers));
              TF_ASSIGN_OR_RETURN(auto rhs, getOperandArray(op.getRhs(), args,
                                                            transient_buffers));
              switch (op.getComparisonDirection()) {
                case ComparisonDirection::NE:
                  return std::vector<mx::array>{mx::not_equal(lhs, rhs)};
                case ComparisonDirection::GE:
                  return std::vector<mx::array>{mx::greater_equal(lhs, rhs)};
                case ComparisonDirection::GT:
                  return std::vector<mx::array>{mx::greater(lhs, rhs)};
                case ComparisonDirection::LE:
                  return std::vector<mx::array>{mx::less_equal(lhs, rhs)};
                case ComparisonDirection::LT:
                  return std::vector<mx::array>{mx::less(lhs, rhs)};
                case ComparisonDirection::EQ:
                  return std::vector<mx::array>{mx::equal(lhs, rhs)};
              }
            })

            // Handle StableHLO Slice ops
            .Case<stablehlo::SliceOp>([&args, &transient_buffers](
                                          auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(
                  auto operand,
                  getOperandArray(op.getOperand(), args, transient_buffers));
              auto op_start_indices = op.getStartIndices();
              auto op_limit_indices = op.getLimitIndices();
              auto op_strides = op.getStrides();
              std::vector<int32_t> start_indices;
              std::vector<int32_t> limit_indices;
              std::vector<int32_t> strides;
              for (auto i = 0; i < op_start_indices.size(); i++) {
                start_indices.push_back(
                    static_cast<int32_t>(op_start_indices[i]));
                limit_indices.push_back(
                    static_cast<int32_t>(op_limit_indices[i]));
                strides.push_back(static_cast<int32_t>(op_strides[i]));
              }
              return std::vector<mx::array>{
                  mx::slice(operand, start_indices, limit_indices, strides)};
            })
            .Case<stablehlo::DynamicSliceOp>(
                [&args, &transient_buffers](auto op) -> StatusOrArrays {
                  TF_ASSIGN_OR_RETURN(mx::array operand,
                                      getOperandArray(op.getOperand(), args,
                                                      transient_buffers));
                  auto op_start_indices = op.getStartIndices();
                  std::vector<mx::array> start_indices_list;
                  for (Value val : op_start_indices) {
                    TF_ASSIGN_OR_RETURN(
                        mx::array indice_value,
                        getOperandArray(val, args, transient_buffers));
                    start_indices_list.emplace_back(indice_value);
                  }
                  mx::array start_indices = mx::concatenate(start_indices_list);
                  std::vector<int> axes;
                  axes.resize(operand.ndim());
                  std::iota(axes.begin(), axes.end(), 0);
                  std::vector<int32_t> slice_sizes;
                  for (auto i = 0; i < op.getSliceSizes().size(); i++) {
                    slice_sizes.push_back(
                        static_cast<int32_t>(op.getSliceSizes()[i]));
                  }
                  return std::vector<mx::array>{
                      mx::slice(operand, start_indices, axes, slice_sizes)};
                })
            .Case<stablehlo::DynamicUpdateSliceOp>(
                [&args, &transient_buffers](auto op) -> StatusOrArrays {
                  TF_ASSIGN_OR_RETURN(auto operand,
                                      getOperandArray(op.getOperand(), args,
                                                      transient_buffers));
                  TF_ASSIGN_OR_RETURN(
                      auto update,
                      getOperandArray(op.getUpdate(), args, transient_buffers));
                  auto op_start_indices = op.getStartIndices();
                  std::vector<mx::array> start_indices_list;
                  for (Value val : op_start_indices) {
                    TF_ASSIGN_OR_RETURN(
                        mx::array indice_value,
                        getOperandArray(val, args, transient_buffers));
                    start_indices_list.emplace_back(indice_value);
                  }
                  mx::array start_indices = mx::concatenate(start_indices_list);
                  std::vector<int32_t> axes;
                  axes.resize(operand.ndim());
                  std::iota(axes.begin(), axes.end(), 0);
                  return std::vector<mx::array>{
                      mx::slice_update(operand, update, start_indices, axes)};
                })

            // Handle StableHLO Other ops
            // .Case<stablehlo::BatchNormGradOp>([](auto op) {})
            // .Case<stablehlo::BatchNormInferenceOp>([](auto op) {})
            // .Case<stablehlo::BatchNormTrainingOp>([](auto op) {})
            .Case<stablehlo::BitcastConvertOp>([&args, &transient_buffers](
                                                   auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(
                  auto operand,
                  getOperandArray(op.getOperand(), args, transient_buffers));

              return std::vector<mx::array>{mx::view(
                  operand,
                  dtypeFromType(mlir::cast<ShapedType>(op.getResult().getType())
                                    .getElementType()))};
            })
            /*
              .Case<stablehlo::BroadcastOp>([](auto op) {})
              Deprecated see:
                https://github.com/openxla/stablehlo/issues/2340
                https://github.com/openxla/stablehlo/pull/2283
            */
            .Case<stablehlo::BroadcastInDimOp>([&args, &transient_buffers](
                                                   auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(
                  mx::array operand,
                  getOperandArray(op.getOperand(), args, transient_buffers));
              std::vector<int32_t> target_shape =
                  getMlxShape(mlir::cast<ShapedType>(op.getResult().getType()));

              // potentially reshape to pad with ones, which allows broadcasting
              auto min_rank = std::min(operand.ndim(), target_shape.size());
              if (min_rank == operand.ndim()) {
                std::vector<int32_t> padded_shape = operand.shape();
                for (auto i = 0; i < target_shape.size(); i++) {
                  if (i < min_rank) {
                    if (padded_shape[i] != target_shape[i])
                      goto endif;
                    else
                      continue;
                  } else {
                    padded_shape.push_back(1);
                  }
                }
                operand = mx::reshape(operand, padded_shape);
              }
            endif:
              return std::vector<mx::array>{
                  mx::broadcast_to(operand, target_shape)};
            })
            // .Case<stablehlo::DynamicBroadcastInDimOp>([](auto op) {})
            .Case<stablehlo::CholeskyOp>(
                [&args, &transient_buffers](auto op) -> StatusOrArrays {
                  TF_ASSIGN_OR_RETURN(auto operand,
                                      getOperandArray(op.getOperand(), args,
                                                      transient_buffers));
                  return std::vector<mx::array>{
                      mx::linalg::cholesky(operand, op.getLower() != 0)};
                })
            .Case<stablehlo::ClampOp>([&args, &transient_buffers](
                                          auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(
                  auto a_min,
                  getOperandArray(op.getMin(), args, transient_buffers));
              TF_ASSIGN_OR_RETURN(auto a, getOperandArray(op.getOperand(), args,
                                                          transient_buffers));
              TF_ASSIGN_OR_RETURN(
                  auto a_max,
                  getOperandArray(op.getMax(), args, transient_buffers));
              return std::vector<mx::array>{mx::clip(a, a_min, a_max)};
            })
            .Case<stablehlo::ConcatenateOp>(
                [&args, &transient_buffers](auto op) -> StatusOrArrays {
                  std::vector<mx::array> operands;
                  for (Value val : op.getOperands()) {
                    TF_ASSIGN_OR_RETURN(
                        auto operand_array,
                        getOperandArray(val, args, transient_buffers));
                    operands.emplace_back(operand_array);
                  }
                  return std::vector<mx::array>{
                      mx::concatenate(operands, op.getDimension())};
                })
            // .Case<stablehlo::CollectiveBroadcastOp>([](auto op) {})
            // .Case<stablehlo::CollectivePermuteOp>([](auto op) {})
            // .Case<stablehlo::CompositeOp>([](auto op) {})
            // .Case<stablehlo::ConvolutionOp>([](auto op) {})
            // .Case<stablehlo::CrossReplicaSumOp>([](auto op) {})
            // .Case<stablehlo::CustomCallOp>([](auto op) {})
            /*
              .Case<stablehlo::DotOp>([](auto op) {})
              Deprecated see:
                https://github.com/openxla/stablehlo/issues/2340
                https://github.com/openxla/stablehlo/pull/2283
            */
            .Case<stablehlo::DotGeneralOp>([&args, &transient_buffers](
                                               auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto lhs, getOperandArray(op.getLhs(), args,
                                                            transient_buffers));
              TF_ASSIGN_OR_RETURN(auto rhs, getOperandArray(op.getRhs(), args,
                                                            transient_buffers));
              auto lhs_batch_dims =
                  op.getDotDimensionNumbers().getLhsBatchingDimensions();
              auto rhs_batch_dims =
                  op.getDotDimensionNumbers().getRhsBatchingDimensions();
              auto lhs_contract_dims =
                  op.getDotDimensionNumbers().getLhsContractingDimensions();
              auto rhs_contract_dims =
                  op.getDotDimensionNumbers().getRhsContractingDimensions();

              int dim_count = 0;
              auto getDimChar = [&dim_count]() -> char {
                return 'a' + dim_count++;
              };

              std::unordered_map<int, char> lhs_batch_map;
              std::unordered_map<int, char> rhs_batch_map;
              std::unordered_map<int, char> lhs_contract_map;
              std::unordered_map<int, char> rhs_contract_map;

              std::string res_subscript;
              for (auto i = 0; i < lhs_batch_dims.size(); ++i) {
                auto dim_char = getDimChar();
                res_subscript = res_subscript + dim_char;
                lhs_batch_map.emplace(lhs_batch_dims[i], dim_char);
                rhs_batch_map.emplace(rhs_batch_dims[i], dim_char);
              }

              for (auto i = 0; i < lhs_contract_dims.size(); ++i) {
                auto dim_char = getDimChar();
                lhs_contract_map.emplace(lhs_contract_dims[i], dim_char);
                rhs_contract_map.emplace(rhs_contract_dims[i], dim_char);
              }

              std::string lhs_subscript;
              for (auto i = 0; i < lhs.ndim(); ++i) {
                if (auto match = lhs_batch_map.find(i);
                    match != lhs_batch_map.end()) {
                  lhs_subscript = lhs_subscript + match->second;
                } else if (auto match = lhs_contract_map.find(i);
                           match != lhs_contract_map.end()) {
                  lhs_subscript = lhs_subscript + match->second;
                } else {
                  auto dim_char = getDimChar();
                  res_subscript = res_subscript + dim_char;
                  lhs_subscript = lhs_subscript + dim_char;
                }
              }

              std::string rhs_subscript;
              for (auto i = 0; i < rhs.ndim(); ++i) {
                if (auto match = rhs_batch_map.find(i);
                    match != rhs_batch_map.end()) {
                  rhs_subscript = rhs_subscript + match->second;
                } else if (auto match = rhs_contract_map.find(i);
                           match != rhs_contract_map.end()) {
                  rhs_subscript = rhs_subscript + match->second;
                } else {
                  auto dim_char = getDimChar();
                  res_subscript = res_subscript + dim_char;
                  rhs_subscript = rhs_subscript + dim_char;
                }
              }

              return std::vector<mx::array>{mx::einsum(
                  lhs_subscript + "," + rhs_subscript + "->" + res_subscript,
                  {lhs, rhs})};
            })
            /*
              .Case<stablehlo::EinsumOp>([](auto op) {})
              .Case<stablehlo::UnaryEinsumOp>([](auto op) {})
              Deprecated see:
                https://github.com/openxla/stablehlo/issues/2340
                https://github.com/openxla/stablehlo/pull/2283
            */
            // .Case<stablehlo::FftOp>([](auto op) {})
            // TODO (@cryptodeal): fix implementation; works for some tests,
            // others it pulls correct values, but in the wrong order.
            .Case<stablehlo::GatherOp>([&args, &transient_buffers](
                                           auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(
                  mx::array operand,
                  getOperandArray(op.getOperand(), args, transient_buffers));
              TF_ASSIGN_OR_RETURN(mx::array start_indices,
                                  getOperandArray(op.getStartIndices(), args,
                                                  transient_buffers));

              std::cout << "Start Indices: " << start_indices << std::endl;
              auto dimension_numbers = op.getDimensionNumbers();

              auto offset_dims = dimension_numbers.getOffsetDims();
              auto collapsed_slice_dims =
                  dimension_numbers.getCollapsedSliceDims();
              auto operand_batching_dims =
                  dimension_numbers.getOperandBatchingDims();
              auto start_indices_batching_dims =
                  dimension_numbers.getStartIndicesBatchingDims();
              auto start_index_map = dimension_numbers.getStartIndexMap();
              auto index_vector_dim = dimension_numbers.getIndexVectorDim();
              auto slice_sizes = op.getSliceSizes();

              // DEBUG
              // {
              //   std::cout << "Operand Shape: { ";
              //   for (auto i : operand.shape()) {
              //     std::cout << std::to_string(i) << ", ";
              //   }
              //   std::cout << "}" << std::endl;
              //   std::cout << "Operand: " << operand << std::endl;

              //   std::cout << "Start Indices Shape: { ";
              //   for (auto i : start_indices.shape()) {
              //     std::cout << std::to_string(i) << ", ";
              //   }
              //   std::cout << "}" << std::endl;
              //   std::cout << "Start Indices: ";
              //   start_indices.eval();
              //   auto* data = start_indices.data<int32_t>();
              //   for (int i = 0; i < start_indices.data_size(); ++i) {
              //     std::cout << std::to_string(data[i]) << ", ";
              //   }
              //   std::cout << std::endl;
              //   std::cout << "offset_dims = { ";
              //   for (auto i : start_index_map) {
              //     std::cout << std::to_string(i) << ", ";
              //   }
              //   std::cout << "}" << std::endl;

              //   std::cout << "collapsed_slice_dims = { ";
              //   for (auto i : collapsed_slice_dims) {
              //     std::cout << std::to_string(i) << ", ";
              //   }
              //   std::cout << "}" << std::endl;

              //   std::cout << "operand_batching_dims = { ";
              //   for (auto i : operand_batching_dims) {
              //     std::cout << std::to_string(i) << ", ";
              //   }
              //   std::cout << "}" << std::endl;

              //   std::cout << "start_indices_batching_dims = { ";
              //   for (auto i : start_indices_batching_dims) {
              //     std::cout << std::to_string(i) << ", ";
              //   }
              //   std::cout << "}" << std::endl;

              //   std::cout << "start_index_map = { ";
              //   for (auto i : start_index_map) {
              //     std::cout << std::to_string(i) << ", ";
              //   }
              //   std::cout << "}" << std::endl;

              //   std::cout << "index_vector_dim = "
              //             << std::to_string(index_vector_dim) << std::endl;

              //   std::cout << "slice_sizes = { ";
              //   for (auto i : slice_sizes) {
              //     std::cout << std::to_string(i) << ", ";
              //   }
              //   std::cout << "}" << std::endl;
              // }
              // END DEBUG

              // Ensure start_indices has the correct shape
              mx::array adjusted_start_indices = start_indices;
              if (index_vector_dim == start_indices.ndim()) {
                adjusted_start_indices = mx::expand_dims(start_indices, -1);
              }

              // Create a full starting index tensor (sin) by scattering the
              // start indices
              std::vector<int> sin_shape = adjusted_start_indices.shape();
              sin_shape.back() = operand.ndim();
              mx::array sin = mx::zeros(sin_shape, mx::int32);
              for (size_t i = 0; i < start_index_map.size(); ++i) {
                std::vector<int32_t> a_start(adjusted_start_indices.ndim(), 0);
                a_start[a_start.size() - 1] = static_cast<int32_t>(i);
                std::vector<int32_t> a_stop = adjusted_start_indices.shape();
                a_stop[a_stop.size() - 1] = static_cast<int32_t>(i + 1);
                std::vector<int32_t> sin_start(sin.ndim(), 0);
                sin_start[sin_start.size() - 1] =
                    static_cast<int32_t>(start_index_map[i]);
                std::vector<int32_t> sin_stop = sin.shape();
                sin_stop[sin_stop.size() - 1] =
                    static_cast<int32_t>(start_index_map[i] + 1);
                sin = mx::slice_update(
                    sin, mx::slice(adjusted_start_indices, a_start, a_stop),
                    sin_start, sin_stop);
              }

              // Adjust slice sizes to account for collapsed dimensions
              std::vector<int> adjusted_slice_sizes;
              for (size_t i = 0; i < slice_sizes.size(); ++i) {
                if (std::find(collapsed_slice_dims.begin(),
                              collapsed_slice_dims.end(),
                              i) == collapsed_slice_dims.end()) {
                  adjusted_slice_sizes.push_back(slice_sizes[i]);
                }
              }

              // Gather slices using advanced indexing
              std::vector<mx::array> slices;
              int num_slices = sin.size() / sin.shape().back();
              for (int i = 0; i < num_slices; ++i) {
                std::vector<int32_t> start;
                std::vector<int32_t> stop;
                for (int j = 0; j < sin.shape().back(); ++j) {
                  int idx = mx::slice(sin, {i, j}, {i + 1, j + 1}).item<int>();
                  start.emplace_back(idx);
                  stop.emplace_back(idx + slice_sizes[j]);
                }
                slices.push_back(mx::slice(operand, start, stop));
              }

              // Stack slices into output shape
              std::vector<int> output_shape = adjusted_start_indices.shape();
              output_shape.pop_back();
              output_shape.insert(output_shape.end(),
                                  adjusted_slice_sizes.begin(),
                                  adjusted_slice_sizes.end());
              return std::vector<mx::array>{
                  mx::reshape(mx::stack(slices), output_shape)};
            })
            .Case<stablehlo::GetDimensionSizeOp>(
                [&args, &transient_buffers](auto op) -> StatusOrArrays {
                  TF_ASSIGN_OR_RETURN(mx::array operand,
                                      getOperandArray(op.getOperand(), args,
                                                      transient_buffers));
                  return std::vector<mx::array>{
                      mx::array(operand.shape(op.getDimension()))};
                })
            // .Case<stablehlo::MapOp>([](auto op) {})
            .Case<stablehlo::ReshapeOp>(
                [&args, &transient_buffers](auto op) -> StatusOrArrays {
                  TF_ASSIGN_OR_RETURN(auto operand,
                                      getOperandArray(op.getOperand(), args,
                                                      transient_buffers));
                  return std::vector<mx::array>{
                      mx::reshape(operand, getMlxShape(mlir::cast<ShapedType>(
                                               op.getResult().getType())))};
                })
            // .Case<stablehlo::DynamicReshapeOp>([](auto op) {})
            // .Case<stablehlo::ScatterOp>([&args, &transient_buffers](
            //                                 auto op) -> StatusOrArrays {
            //   std::vector<mx::array> inputs;
            //   for (Value val : op.getInputs()) {
            //     TF_ASSIGN_OR_RETURN(
            //         mx::array input_array,
            //         getOperandArray(val, args, transient_buffers));
            //     inputs.emplace_back(input_array);
            //   }
            //   // std::cout << std::to_string(inputs.size()) << " inputs"
            //   //           << std::endl;
            //   TF_ASSIGN_OR_RETURN(mx::array scatter_indices,
            //                       getOperandArray(op.getScatterIndices(),
            //                       args,
            //                                       transient_buffers));
            //   std::vector<mx::array> updates;
            //   for (Value val : op.getUpdates()) {
            //     TF_ASSIGN_OR_RETURN(
            //         mx::array update_array,
            //         getOperandArray(val, args, transient_buffers));
            //     updates.emplace_back(update_array);
            //   }
            //   // std::cout << std::to_string(updates.size()) << " updates"
            //   //           << std::endl;

            //   func::Func update_computation = op.getUpdateComputation();
            //   for (auto i = 0; i < inputs.size(); i++) {
            //     Block& body_block = op.getUpdateComputation().front();
            //     body_block.dump();
            //   }

            //   return absl::UnimplementedError("ScatterOp not implemented
            //   yet");
            // })
            .Case<stablehlo::SelectOp>([&args, &transient_buffers](
                                           auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(
                  auto condition,
                  getOperandArray(op.getOperand(0), args, transient_buffers));
              TF_ASSIGN_OR_RETURN(
                  auto x,
                  getOperandArray(op.getOperand(1), args, transient_buffers));
              TF_ASSIGN_OR_RETURN(
                  auto y,
                  getOperandArray(op.getOperand(2), args, transient_buffers));
              return std::vector<mx::array>{mx::where(condition, x, y)};
            })
            // .Case<stablehlo::SelectAndScatterOp>([](auto op) {})
            // .Case<stablehlo::SetDimensionSizeOp>([](auto op) {})
            // .Case<stablehlo::SortOp>([](auto op) {})
            // .Case<stablehlo::ReverseOp>([](auto op) {})
            // .Case<stablehlo::PadOp>([](auto op) {})
            .Case<stablehlo::TransposeOp>(
                [&args, &transient_buffers](auto op) -> StatusOrArrays {
                  TF_ASSIGN_OR_RETURN(auto operand,
                                      getOperandArray(op.getOperand(), args,
                                                      transient_buffers));
                  std::vector<int> axes;
                  for (auto axis : op.getPermutation()) {
                    axes.push_back(static_cast<int>(axis));
                  }
                  return std::vector<mx::array>{mx::transpose(operand, axes)};
                })
            // .Case<stablehlo::TriangularSolveOp>([](auto op) {})
            // .Case<stablehlo::ReduceWindowOp>([](auto op) {})
            // .Case<stablehlo::ReturnOp>([](auto op) {})
            // .Case<stablehlo::TorchIndexSelectOp>([](auto op) {})
            // .Case<stablehlo::OptimizationBarrierOp>([](auto op) {})
            // .Case<stablehlo::CrossReplicaSumOp>([](auto op) {})

            // Handle StableHLO RNG ops
            .Case<stablehlo::RngOp>(
                [&args, &transient_buffers](auto op) -> StatusOrArrays {
                  // op.dump();
                  TF_ASSIGN_OR_RETURN(
                      mx::array a,
                      getOperandArray(op.getA(), args, transient_buffers));
                  TF_ASSIGN_OR_RETURN(
                      mx::array b,
                      getOperandArray(op.getB(), args, transient_buffers));
                  TF_ASSIGN_OR_RETURN(
                      mx::array dims,
                      getOperandArray(op.getShape(), args, transient_buffers));
                  auto shape_buffer = dims.data<int64_t>();
                  std::vector<int32_t> shape;
                  for (auto i = 0; i < dims.size(); i++) {
                    shape.push_back(static_cast<int32_t>(shape_buffer[i]));
                  }
                  std::vector<mx::array> res;
                  if (op.getRngDistribution() == RngDistribution::UNIFORM) {
                    return std::vector<mx::array>{
                        mx::random::uniform(a, b, shape, a.dtype())};
                  } else {
                    return std::vector<mx::array>{mx::random::normal(
                        shape, a.dtype(),
                        (mx::astype(a, mx::float32)).item<float>(),
                        (mx::astype(b, mx::float32)).item<float>())};
                  }
                })
            .Case<stablehlo::RngBitGeneratorOp>(
                [&args, &transient_buffers](auto op) -> StatusOrArrays {
                  TF_ASSIGN_OR_RETURN(mx::array initial_state,
                                      getOperandArray(op.getInitialState(),
                                                      args, transient_buffers));
                  initial_state = mx::astype(initial_state, mx::uint32);
                  // whereas stablehlo prng state is an array of `uint64_t`, mlx
                  // is an array of `uint32_t`
                  auto output_type =
                      mlir::cast<ShapedType>(op.getOutput().getType());
                  std::vector<int32_t> out_shape;
                  for (auto dim : output_type.getShape()) {
                    out_shape.push_back(static_cast<int32_t>(dim));
                  }
                  StatusOrInt element_width =
                      getByteWidth(output_type.getElementType());
                  if (!element_width.ok()) {
                    return element_width.status();
                  }
                  // TODO (@cryptodeal): implement `THREE_FRY` and `PHILOX`
                  // algos
                  auto res = mx::random::bits(out_shape, *element_width,
                                              initial_state);
                  return std::vector<mx::array>{
                      mx::astype(mx::random::split(initial_state).second,
                                 mx::uint64),
                      res};
                })

            // Handle StableHLO Quantize ops
            // .Case<stablehlo::UniformQuantizeOp>([](auto op) {})
            // .Case<stablehlo::UniformDequantizeOp>([](auto op) {})
            // .Case<stablehlo::ReducePrecisionOp>([](auto op) {})
            /*
              .Case<stablehlo::RealDynamicSliceOp>([](auto op) {})
              Deprecated see:
                https://github.com/openxla/stablehlo/issues/2340
                https://github.com/openxla/stablehlo/pull/2283
            */
            // .Case<stablehlo::DynamicPadOp>([](auto op) {})
            // .Case<stablehlo::DynamicGatherOp>([](auto op) {})
            // .Case<stablehlo::DynamicConvOp>([](auto op) {})
            .Default([](auto op) -> StatusOrArrays {
              return absl::UnimplementedError(
                  absl::StrCat("Unsupported op: ", ToString(op)));
            });
    if (maybe_result.ok()) {
      transient_buffers.emplace(&op, *maybe_result);
    } else {
      return maybe_result.status();
    }
  }

  return transient_buffers.find(result_op)->second;
}

class MlirLoadedExecutable : public PjRtLoadedExecutable {
 public:
  MlirLoadedExecutable(ModuleOp module, DeviceAssignment assignment,
                       absl::Span<PjRtDevice* const> devices,
                       PjRtClient* client)
      : PjRtLoadedExecutable(),
        name_("MlirLoadedExecutable"),
        assignment_(assignment),
        devices_(client->devices()),
        client_(client),
        context_(),
        module_(cloneIntoContext(module, context_)) {
    // TRACE_ME_MEMBER;
  }

  static absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(
      ModuleOp module, DeviceAssignment assignment,
      absl::Span<PjRtDevice* const> devices, PjRtClient* client) {
    TRACE_ME;
    // printf("%s\n", ToString(module).c_str());
    mlir::BaseScopedDiagnosticHandler diagnostic_handler(module->getContext());
    if (failed(decomposeChloToStablehlo(module))) {
      return diagnostic_handler.ConsumeStatus();
    }
    if (auto unsupported_op = getUnsupportedOp(module)) {
      LOG(ERROR) << "Unsupported op: " << ToString(*unsupported_op)
                 << "\nFound in module: " << ToString(module);
      return absl::UnimplementedError(
          absl::StrCat("Unsupported op: ", ToString(*unsupported_op)));
    }

    // Simplify the graph using available HWI passes.
    if (failed(runHardwareIndependentOptimizations(module))) {
      return diagnostic_handler.ConsumeStatus();
    }

    auto executable = std::make_unique<MlirLoadedExecutable>(module, assignment,
                                                             devices, client);

    return executable;
  }

  PjRtClient* client() const override {
    // TRACE_ME_MEMBER;
    return client_;
  }

  const DeviceAssignment& device_assignment() const override {
    // TRACE_ME_MEMBER;
    return assignment_;
  }

  absl::Span<const PjRtLoadedExecutable::LogicalDeviceIds>
  addressable_device_logical_ids() const override {
    // TRACE_ME_MEMBER;
    LOG_UNIMPLEMENTED(addressable_device_logical_ids);
    return {};
  }

  absl::Span<PjRtDevice* const> addressable_devices() const override {
    // TRACE_ME_MEMBER;
    return devices_;
  }

  // Helper function to get default mem from device.
  PjRtMemorySpace* get_default_memory_space() const {
    // TRACE_ME_MEMBER;
    return devices_[0]->default_memory_space().value_or(nullptr);
  }

  absl::StatusOr<SmallVector<DenseElementsAttr>> evalModule(
      ModuleOp& module, const SmallVector<mlir::DenseElementsAttr>& inputs) {
    // TRACE_ME_MEMBER;
    std::vector<mx::array> block_arguments;
    for (auto input : inputs) {
      TF_ASSIGN_OR_RETURN(auto input_arr, getDenseElementsArray(input));
      block_arguments.emplace_back(input_arr);
    }

    // TODO(@cryptodeal): operations can return multiple results,
    // will need to switch to something along the lines of
    // std::unordered_map<Operation*, std::vector<mx::array>>
    // transient_buffers;

    // holds intermediary results of operation calls.
    std::unordered_map<Operation*, mx::array> transient_buffers;
    // std::cout << std::endl << std::endl;
    module.dump();
    auto main = module.lookupSymbol<mlir::func::FuncOp>("main");
    TF_ASSIGN_OR_RETURN(auto mlx_res, evalFunc(module, main, block_arguments));

    SmallVector<DenseElementsAttr> result;
    for (auto& a : mlx_res) {
      auto arr = mx::contiguous(a);
      arr.eval();
      SmallVector<int64_t> shape;
      for (auto dim : arr.shape()) {
        shape.push_back(static_cast<int64_t>(dim));
      }
      ArrayRef<char> data(arr.data<char>(), arr.nbytes());
      DenseElementsAttr out = DenseElementsAttr::getFromRawBuffer(
          getResultType(module, arr.dtype(), shape), data);
      result.push_back(out);
    }
    return result;

    // // register all function declarations
    // std::unordered_map<std::string, mlir::func::FuncOp> function_decls;
    // for (auto func : module.getOps<mlir::func::FuncOp>()) {
    //   func->dump();
    //   for (Operation& op : func.getOperations()) op.dump();
    //   //
    //   function_decls.emplace(cast<mlir::func::FuncOp>(func).getSymName().str(),
    //   // func);
    // }

    // // `main` function is entry point for walking the module
    // if (auto main = function_decls.find("main"); main !=
    // function_decls.end()) { } else
    //   return failure();

    // TODO(@cryptodeal): ensure the resulting data is allocated/freed
    // correctly.
    // if (res.has_value()) {
    //   mx::array out = res.value();
    //   mx::eval(out);
    //   SmallVector<int64_t> shape;
    //   for (auto dim : out.shape()) {
    //     shape.push_back(static_cast<int64_t>(dim));
    //   }
    //   ArrayRef<char> data(out.data<char>(), out.nbytes());
    //   DenseElementsAttr result = DenseElementsAttr::getFromRawBuffer(
    //       getResultType(module, out.dtype(), shape), data);
    //   return SmallVector<DenseElementsAttr>{result};
    // }
    // return failure();
  }

  absl::StatusOr<PjRtLoadedExecutable::Result> ExecuteWithMlxInterpreter(
      absl::Span<PjRtBuffer* const> argument_handles, ModuleOp module,
      PjRtDevice* device, bool fill_future) {
    // TRACE_ME_MEMBER;
    SmallVector<DenseElementsAttr> inputs;
    for (auto* arg : argument_handles) {
      TF_ASSIGN_OR_RETURN(auto mlirArg, GetAttributeFromBuffer(arg));
      auto mlirArgInModuleContext =
          CloneIntoContext(mlirArg, *module->getContext());
      inputs.push_back(mlirArgInModuleContext);
    }
    // LOG(INFO) << "EvalModule:\n" << ToString(module) << "\n";
    // LOG(INFO) << "Inputs: " << ToString(inputs) << "\n";
    TF_ASSIGN_OR_RETURN(auto result, evalModule(module, inputs));

    // LOG(INFO) << "Results: " << ToString(result.value()) << "\n";

    // Naive memory space selection, only using CPU global memory.
    PjRtMemorySpace* memory_space =
        device->default_memory_space().value_or(nullptr);
    std::vector<std::unique_ptr<PjRtBuffer>> buffer_results;
    for (auto res : result) {
      buffer_results.push_back(
          CreateMlirBufferFromAttribute(res, memory_space));
    }

    std::optional<PjRtFuture<>> future;
    if (fill_future) {
      // Synchronous! To make async, this would need to return a future that
      // is ready when the computation is done.
      future = PjRtFuture<>(absl::OkStatus());
    }
    return PjRtLoadedExecutable::Result{future, std::move(buffer_results)};
  }

  absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>> Execute(
      absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
      const xla::ExecuteOptions& options,
      std::optional<std::vector<PjRtFuture<>>>& returned_futures) override {
    // TRACE_ME_MEMBER;
    if (argument_handles.size() != 1) {
      // One arg handle per device.
      return absl::InvalidArgumentError(
          "MlirLoadedExecutable::Execute only supports a single argument "
          "vector");
    }

    // Single device, synchronous, can always use 0.
    PjRtDevice* device = devices_[0];
    bool fill_future = returned_futures.has_value();
    TF_ASSIGN_OR_RETURN(
        PjRtLoadedExecutable::Result result,
        ExecuteWithMlxInterpreter(argument_handles[0], module_.get(), device,
                                  fill_future));
    std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> results;
    results.push_back(std::move(result.buffers));
    if (returned_futures.has_value()) {
      returned_futures->push_back(std::move(result.future.value()));
    }
    return results;
  }

  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecuteSharded(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const xla::ExecuteOptions& options,
      std::optional<PjRtFuture<>>& returned_future, bool fill_future) override {
    // TRACE_ME_MEMBER;
    // Synchronous! To make async, have the device make a buffer with a ready
    // future that is ready when the computation is done / buffer is ready.
    TF_ASSIGN_OR_RETURN(
        PjRtLoadedExecutable::Result result,
        ExecuteWithMlxInterpreter(argument_handles, module_.get(), device,
                                  fill_future));
    if (returned_future.has_value() && fill_future) {
      returned_future = std::move(result.future);
    }
    return std::move(result.buffers);
  }

  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecutePortable(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const xla::ExecuteOptions& options,
      std::optional<PjRtFuture<>>& returned_future, bool fill_future) override {
    // TRACE_ME_MEMBER;
    // Synchronous! To make async, have the device make a buffer with a ready
    // future that is ready when the computation is done / buffer is ready.
    TF_ASSIGN_OR_RETURN(
        PjRtLoadedExecutable::Result result,
        ExecuteWithMlxInterpreter(argument_handles, module_.get(), device,
                                  fill_future));
    if (returned_future.has_value() && fill_future) {
      returned_future = std::move(result.future);
    }
    return std::move(result.buffers);
  }

  void Delete() override {
    // TRACE_ME_MEMBER;
    module_.release();
    module_ = nullptr;
  }
  bool IsDeleted() override {
    // TRACE_ME_MEMBER;
    return !module_;
  }

  // PjRtExecutable API.
  int num_replicas() const override {
    // TRACE_ME_MEMBER;
    return assignment_.replica_count();
  }
  int num_partitions() const override {
    // TRACE_ME_MEMBER;
    return assignment_.computation_count();
  }
  int64_t SizeOfGeneratedCodeInBytes() const override {
    // No generated code.. so just return 1.
    // TRACE_ME_MEMBER;
    return 1;
  }
  absl::string_view name() const override {
    // TRACE_ME_MEMBER;
    return name_;
  }

  absl::StatusOr<std::vector<std::shared_ptr<xla::HloModule>>> GetHloModules()
      const override {
    // TODO: This shouldn't be needed for an MLIR plugin, its only used in the
    // JAX layer for determining output sharding, which exists on the mlir
    // module.
    // TRACE_ME_MEMBER;
    auto moduleClone = llvm::cast<ModuleOp>(module_.get()->clone());
    TF_ASSIGN_OR_RETURN(auto hlo_module,
                        xla::ConvertStablehloToHlo(moduleClone));
    return std::vector<std::shared_ptr<xla::HloModule>>{std::move(hlo_module)};
  }

  absl::StatusOr<std::vector<std::vector<absl::string_view>>>
  GetOutputMemoryKinds() const override {
    // TRACE_ME_MEMBER;
    return UNIMPLEMENTED(GetOutputMemoryKinds);
  }

  absl::StatusOr<std::string> FingerprintExecutable() const override {
    // TRACE_ME_MEMBER;
    return UNIMPLEMENTED(FingerprintExecutable);
  }

 private:
  std::string name_;
  DeviceAssignment assignment_;
  absl::Span<PjRtDevice* const> devices_;
  PjRtClient* client_;
  std::function<std::vector<mx::array>(const std::vector<mx::array>&)>
      compiled_mlx_;

  // MLIR
  MLIRContext context_;
  mlir::OwningOpRef<ModuleOp> module_;
};

absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> StablehloMlxCompile(
    mlir::ModuleOp module, DeviceAssignment assignment, PjRtClient* client) {
  TRACE_ME;
  return MlirLoadedExecutable::Compile(module, assignment, client->devices(),
                                       client);
}

absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> StablehloMlxCompile(
    xla::XlaComputation const& computation, xla::DeviceAssignment assignment,
    xla::PjRtClient* client) {
  TRACE_ME;
  MLIRContext context;
  TF_ASSIGN_OR_RETURN(auto module,
                      ConvertHloToStablehlo(context, &computation.proto()));
  return StablehloMlxCompile(module.get(), assignment, client);
}

}  // namespace mlir::stablehlo