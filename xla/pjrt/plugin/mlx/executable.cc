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

std::vector<mx::array> getArgArrays(
    mlir::Operation::operand_range values,
    const std::unordered_map<Operation*, std::vector<mx::array>>&
        transient_buffers,
    const std::vector<mx::array>& block_arguments) {
  std::vector<mx::array> args;
  for (Value operand : values) {
    if (auto defining_op = operand.getDefiningOp()) {
      // arg was created as a result of previously executed operation
      // push array to args if found, else return early
      if (auto search = transient_buffers.find(defining_op);
          search != transient_buffers.end()) {
        switch (search->second.size()) {
          case 0:
            return args;
          case 1:
            args.push_back(search->second[0]);
            break;
          default:
            unsigned i = 0;
            while (i < defining_op->getNumResults()) {
              Value op_res = defining_op->getResult(i);
              if (op_res == operand) break;
              i++;
            }
            if (i < defining_op->getNumResults()) {
              args.push_back(search->second[i]);
            } else
              return args;
        }
      } else
        return args;
    } else {
      auto block_arg = operand.cast<BlockArgument>();
      auto arg_number = block_arg.getArgNumber();
      // if block arguments contains the arg at the given index,
      // push to args; else return early
      if (block_arguments.size() > arg_number)
        args.push_back(block_arguments[arg_number]);
      else
        return args;
    }
  }
  return args;
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

template <typename T>
mx::array denseElementsArray(const char* buffer,
                             const std::vector<int32_t>& shape, bool is_splat) {
  return is_splat ? mx::full<T>(shape, reinterpret_cast<const T*>(buffer)[0])
                  : mx::array(reinterpret_cast<const T*>(buffer), shape);
}

mx::array getDenseElementsArray(const mlir::DenseElementsAttr& attr) {
  auto attr_type = attr.getType();
  // convert to mlx shape
  std::vector<int32_t> attr_shape = getMlxShape(attr_type);
  auto attr_raw_data = attr.getRawData();
  const char* buffer_ref = attr_raw_data.data();
  auto element_type = attr_type.getElementType();

  // TODO(@cryptodeal): `Compile` must verify there are no unsupported types.

  return llvm::TypeSwitch<Type, mx::array>(element_type)
      // handle integer types
      .Case<IntegerType>([&buffer_ref, &attr, &attr_shape](auto type) {
        switch (type.getWidth()) {
          // handle bool
          case 1:
            return denseElementsArray<bool>(buffer_ref, attr_shape,
                                            attr.isSplat());
          case 8:
            return type.isSignless()
                       ? denseElementsArray<int8_t>(buffer_ref, attr_shape,
                                                    attr.isSplat())
                       : denseElementsArray<uint8_t>(buffer_ref, attr_shape,
                                                     attr.isSplat());
          case 16:
            return type.isSignless()
                       ? denseElementsArray<int16_t>(buffer_ref, attr_shape,
                                                     attr.isSplat())
                       : denseElementsArray<uint16_t>(buffer_ref, attr_shape,
                                                      attr.isSplat());
          case 32:
            return type.isSignless()
                       ? denseElementsArray<int32_t>(buffer_ref, attr_shape,
                                                     attr.isSplat())
                       : denseElementsArray<uint32_t>(buffer_ref, attr_shape,
                                                      attr.isSplat());
          // default is 64 bit width
          default:
            return type.isSignless()
                       ? denseElementsArray<int64_t>(buffer_ref, attr_shape,
                                                     attr.isSplat())
                       : denseElementsArray<uint64_t>(buffer_ref, attr_shape,
                                                      attr.isSplat());
        }
      })
      .Case<BFloat16Type>([&buffer_ref, &attr, &attr_shape](auto type) {
        return denseElementsArray<mx::bfloat16_t>(buffer_ref, attr_shape,
                                                  attr.isSplat());
      })
      .Case<Float16Type>([&buffer_ref, &attr, &attr_shape](auto type) {
        return denseElementsArray<mx::float16_t>(buffer_ref, attr_shape,
                                                 attr.isSplat());
      })
      .Case<ComplexType>([&buffer_ref, &attr, &attr_shape](auto type) {
        return denseElementsArray<mx::complex64_t>(buffer_ref, attr_shape,
                                                   attr.isSplat());
      })
      // Default is Float32Type
      .Default([&buffer_ref, &attr, &attr_shape](auto type) {
        return denseElementsArray<float>(buffer_ref, attr_shape,
                                         attr.isSplat());
      });
}

absl::StatusOr<std::vector<mx::array>> evalOp(
    ModuleOp& module, Operation* op, std::vector<mx::array>& operands) {}

FailureOr<std::vector<mx::array>> evalFunc(ModuleOp& module, func::FuncOp& func,
                                           const std::vector<mx::array>& args) {
  std::unordered_map<Operation*, std::vector<mx::array>> transient_buffers;
  std::optional<std::vector<mx::array>> result;
  for (mlir::Operation& op : func.front().getOperations()) {
    op.dump();
    auto operands = getArgArrays(op.getOperands(), transient_buffers, args);
    if (operands.size() != op.getNumOperands()) return failure();
    std::vector<mx::array> init_values;

    // switch on the operation type
    bool should_continue =
        llvm::TypeSwitch<Operation*, bool>(&op)
            // TODO(@cryptodeal): handle `func` namespace ops
            .Case<func::ReturnOp>([&operands, &result](auto op) {
              result = operands;
              return true;
            })
            .Case<func::CallOp>(
                [&module, &transient_buffers, &operands](auto op) {
                  auto callee =
                      module.lookupSymbol<mlir::func::FuncOp>(op.getCallee());
                  auto maybe_result = evalFunc(module, callee, operands);
                  if (failed(maybe_result)) return false;
                  transient_buffers.emplace(op, maybe_result.value());
                  return true;
                })
            // Handle StableHLO nullary ops
            .Case<stablehlo::ConstantOp>(
                [&transient_buffers, &operands](auto op) {
                  std::vector<mx::array> res = {getDenseElementsArray(
                      mlir::cast<mlir::DenseElementsAttr>(op.getValue()))};
                  transient_buffers.emplace(op, res);
                  return true;
                })
            .Case<stablehlo::IotaOp>([&transient_buffers, &operands](auto op) {
              auto result_type = op.getResult().getType();
              std::vector<int32_t> shape =
                  getMlxShape(mlir::cast<ShapedType>(result_type));
              auto iota_dimension = op.getIotaDimension();
              std::vector<int32_t> dimensions;
              for (auto i = 0; i < dimensions.size(); i++) {
                dimensions.push_back(i != iota_dimension ? 1 : shape[i]);
              }
              std::vector<mx::array> res = {mx::broadcast_to(
                  mx::reshape(
                      mx::arange(static_cast<double>(shape[iota_dimension]),
                                 dtypeFromType(result_type)),
                      dimensions),
                  shape)};
              transient_buffers.emplace(op, res);
              return true;
            })
            // .Case<stablehlo::DynamicIotaOp>([](auto op) {})
            /*
              .Case<stablehlo::CreateTokenOp>([](auto op) {})
              Deprecated see:
                https://github.com/openxla/stablehlo/issues/2340
                https://github.com/openxla/stablehlo/pull/2283
            */

            // Handle StableHLO unary elementwise op
            .Case<stablehlo::AbsOp>([&transient_buffers, &operands](auto op) {
              std::vector<mx::array> res = {mx::abs(operands[0])};
              transient_buffers.emplace(op, res);
              return true;
            })
            .Case<stablehlo::CbrtOp>([&transient_buffers, &operands](auto op) {
              std::vector<mx::array> res = {mx::power(
                  operands[0], mx::full<float>(operands[0].shape(), 1 / 3,
                                               operands[0].dtype()))};
              transient_buffers.emplace(op, res);
              return true;
            })
            .Case<stablehlo::CeilOp>([&transient_buffers, &operands](auto op) {
              std::vector<mx::array> res = {mx::ceil(operands[0])};
              transient_buffers.emplace(op, res);
              return true;
            })

            .Case<stablehlo::ConvertOp>([&transient_buffers,
                                         &operands](auto op) {
              std::vector<mx::array> res = {mx::astype(
                  operands[0],
                  dtypeFromType(op.getResult().getType().getElementType()))};
              transient_buffers.emplace(op, res);
              return true;
            })
            // .Case<stablehlo::ClzOp>([](auto op) {})
            .Case<stablehlo::CosineOp>(
                [&transient_buffers, &operands](auto op) {
                  std::vector<mx::array> res = {mx::cos(operands[0])};
                  transient_buffers.emplace(op, res);
                  return true;
                })
            .Case<stablehlo::ExpOp>([&transient_buffers, &operands](auto op) {
              std::vector<mx::array> res = {mx::exp(operands[0])};
              transient_buffers.emplace(op, res);
              return true;
            })
            .Case<stablehlo::Expm1Op>([&transient_buffers, &operands](auto op) {
              std::vector<mx::array> res = {mx::expm1(operands[0])};
              transient_buffers.emplace(op, res);
              return true;
            })
            .Case<stablehlo::FloorOp>([&transient_buffers, &operands](auto op) {
              std::vector<mx::array> res = {mx::floor(operands[0])};
              transient_buffers.emplace(op, res);
              return true;
            })
            .Case<stablehlo::ImagOp>([&transient_buffers, &operands](auto op) {
              std::vector<mx::array> res = {mx::imag(operands[0])};
              transient_buffers.emplace(op, res);
              return true;
            })
            .Case<stablehlo::IsFiniteOp>(
                [&transient_buffers, &operands](auto op) {
                  std::vector<mx::array> res = {mx::isfinite(operands[0])};
                  transient_buffers.emplace(op, res);
                  return true;
                })
            .Case<stablehlo::LogOp>([&transient_buffers, &operands](auto op) {
              std::vector<mx::array> res = {mx::log(operands[0])};
              transient_buffers.emplace(op, res);
              return true;
            })
            .Case<stablehlo::Log1pOp>([&transient_buffers, &operands](auto op) {
              std::vector<mx::array> res = {mx::log1p(operands[0])};
              transient_buffers.emplace(op, res);
              return true;
            })
            .Case<stablehlo::LogisticOp>(
                [&transient_buffers, &operands](auto op) {
                  std::vector<mx::array> res = {mx::sigmoid(operands[0])};
                  transient_buffers.emplace(op, res);
                  return true;
                })
            .Case<stablehlo::NotOp>([&transient_buffers, &operands](auto op) {
              std::vector<mx::array> res = {mx::logical_not(operands[0])};
              transient_buffers.emplace(op, res);
              return true;
            })
            .Case<stablehlo::NegOp>([&transient_buffers, &operands](auto op) {
              std::vector<mx::array> res = {mx::negative(operands[0])};
              transient_buffers.emplace(op, res);
              return true;
            })
            // .Case<stablehlo::PopulationCountOp>([](auto op) {})
            .Case<stablehlo::RealOp>([&transient_buffers, &operands](auto op) {
              std::vector<mx::array> res = {mx::real(operands[0])};
              transient_buffers.emplace(op, res);
              return true;
            })
            // `stablehlo::RoundOp` does not match with the mlx metal
            // implementation .Case<stablehlo::RoundOp>([](auto op) {})
            .Case<stablehlo::RoundNearestEvenOp>(
                [&transient_buffers, &operands](auto op) {
                  std::vector<mx::array> res = {mx::round(operands[0])};
                  transient_buffers.emplace(op, res);
                  return true;
                })
            .Case<stablehlo::RsqrtOp>([&transient_buffers, &operands](auto op) {
              std::vector<mx::array> res = {mx::rsqrt(operands[0])};
              transient_buffers.emplace(op, res);
              return true;
            })
            .Case<stablehlo::SignOp>([&transient_buffers, &operands](auto op) {
              std::vector<mx::array> res = {mx::sign(operands[0])};
              transient_buffers.emplace(op, res);
              return true;
            })
            .Case<stablehlo::SineOp>([&transient_buffers, &operands](auto op) {
              std::vector<mx::array> res = {mx::sin(operands[0])};
              transient_buffers.emplace(op, res);
              return true;
            })
            .Case<stablehlo::SqrtOp>([&transient_buffers, &operands](auto op) {
              std::vector<mx::array> res = {mx::sqrt(operands[0])};
              transient_buffers.emplace(op, res);
              return true;
            })
            .Case<stablehlo::TanOp>([&transient_buffers, &operands](auto op) {
              std::vector<mx::array> res = {mx::tan(operands[0])};
              transient_buffers.emplace(op, res);
              return true;
            })
            .Case<stablehlo::TanhOp>([&transient_buffers, &operands](auto op) {
              std::vector<mx::array> res = {mx::tanh(operands[0])};
              transient_buffers.emplace(op, res);
              return true;
            })

            // Handle StableHLO binary elementwise ops
            .Case<stablehlo::AddOp>([&transient_buffers, &operands](auto op) {
              std::vector<mx::array> res = {mx::add(operands[0], operands[1])};
              transient_buffers.emplace(op, res);
              return true;
            })
            .Case<stablehlo::Atan2Op>([&transient_buffers, &operands](auto op) {
              std::vector<mx::array> res = {
                  mx::arctan2(operands[0], operands[1])};
              transient_buffers.emplace(op, res);
              return true;
            })
            // TODO(@cryptodeal): implement complex op
            // .Case<stablehlo::ComplexOp>([&transient_buffers, &operands](auto
            // op) {})
            .Case<stablehlo::DivOp>([&transient_buffers, &operands](auto op) {
              std::vector<mx::array> res = {
                  mx::divide(operands[0], operands[1])};
              transient_buffers.emplace(op, res);
              return true;
            })
            .Case<stablehlo::MaxOp>([&transient_buffers, &operands](auto op) {
              std::vector<mx::array> res = {
                  mx::maximum(operands[0], operands[1])};
              transient_buffers.emplace(op, res);
              return true;
            })
            .Case<stablehlo::MinOp>([&transient_buffers, &operands](auto op) {
              std::vector<mx::array> res = {
                  mx::minimum(operands[0], operands[1])};
              transient_buffers.emplace(op, res);
              return true;
            })
            .Case<stablehlo::MulOp>([&transient_buffers, &operands](auto op) {
              std::vector<mx::array> res = {
                  mx::multiply(operands[0], operands[1])};
              transient_buffers.emplace(op, res);
              return true;
            })
            .Case<stablehlo::PowOp>([&transient_buffers, &operands](auto op) {
              std::vector<mx::array> res = {
                  mx::power(operands[0], operands[1])};
              transient_buffers.emplace(op, res);
              return true;
            })
            .Case<stablehlo::RemOp>([&transient_buffers, &operands](auto op) {
              std::vector<mx::array> res = {
                  mx::remainder(operands[0], operands[1])};
              transient_buffers.emplace(op, res);
              return true;
            })
            .Case<stablehlo::ShiftLeftOp>(
                [&transient_buffers, &operands](auto op) {
                  std::vector<mx::array> res = {
                      mx::left_shift(operands[0], operands[1])};
                  transient_buffers.emplace(op, res);
                  return true;
                })
            /**
             * Per Metal Spec:
             * For the right-shift operator, if E1 has an unsigned type or if
             E1
             * has a signed type and a nonnegative value, the vacated bits are
             * filled with zeros. If E1 has a signed type and a negative value,
             * the vacated bits are filled with ones.
             */
            .Case<stablehlo::ShiftRightArithmeticOp,
                  stablehlo::ShiftRightLogicalOp>(
                [&transient_buffers, &operands](auto op) {
                  std::vector<mx::array> res = {
                      mx::right_shift(operands[0], operands[1])};
                  transient_buffers.emplace(op, res);
                  return true;
                })
            .Case<stablehlo::SubtractOp>(
                [&transient_buffers, &operands](auto op) {
                  std::vector<mx::array> res = {
                      mx::subtract(operands[0], operands[1])};
                  transient_buffers.emplace(op, res);
                  return true;
                })

            // Handle StableHLO binary logical elementwise ops
            .Case<stablehlo::AndOp>([&transient_buffers, &operands](auto op) {
              std::vector<mx::array> res = {
                  mx::logical_and(operands[0], operands[1])};
              transient_buffers.emplace(op, res);
              return true;
            })
            .Case<stablehlo::OrOp>([&transient_buffers, &operands](auto op) {
              std::vector<mx::array> res = {
                  mx::logical_or(operands[0], operands[1])};
              transient_buffers.emplace(op, res);
              return true;
            })
            .Case<stablehlo::XorOp>([&transient_buffers, &operands](auto op) {
              std::vector<mx::array> res = {
                  mx::bitwise_xor(operands[0], operands[1])};
              transient_buffers.emplace(op, res);
              return true;
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
            .Case<stablehlo::ReduceOp>([&transient_buffers, &operands,
                                        &op](auto reduce_op) {
              if (isArgMaxReduce(reduce_op)) {
                // std::cout << "ArgMaxReduceOp\n" << std::endl;
                auto axis = static_cast<int>(reduce_op.getDimensions()[0]);
                auto indices = mx::argmax(operands[0], axis);
                // TODO (@cryptodeal): convert indices to result type
                std::vector<mx::array> res = {
                    mx::take(operands[0], indices, axis),
                    mx::astype(indices, mx::uint8)};
                transient_buffers.emplace(reduce_op, res);
                return true;
              } else if (isSumReduce(reduce_op)) {
                std::vector<mx::array> res = {
                    mx::sum(operands[0],
                            static_cast<int>(reduce_op.getDimensions()[0]))};
                transient_buffers.emplace(reduce_op, res);
                return true;
              } else if (isMaxReduce(reduce_op)) {
                std::vector<mx::array> res = {
                    mx::max(operands[0],
                            static_cast<int>(reduce_op.getDimensions()[0]))};
                transient_buffers.emplace(reduce_op, res);
                return true;
              } else if (isMinReduce(reduce_op)) {
                std::vector<mx::array> res = {
                    mx::min(operands[0],
                            static_cast<int>(reduce_op.getDimensions()[0]))};
                transient_buffers.emplace(reduce_op, res);
                return true;
              } else if (isAnyReduce(reduce_op)) {
                std::vector<mx::array> res = {
                    mx::any(operands[0],
                            static_cast<int>(reduce_op.getDimensions()[0]))};
                transient_buffers.emplace(reduce_op, res);
                return true;
              } else
                return false;
            })

            // Handle StableHLO tuple ops
            // .Case<stablehlo::GetTupleElementOp>([](auto op) {})
            // .Case<stablehlo::TupleOp>([](auto op) {})

            .Case<stablehlo::CompareOp>([&transient_buffers,
                                         &operands](auto op) {
              auto comp_dir = op.getComparisonDirection();
              std::vector<mx::array> res;
              switch (op.getComparisonDirection()) {
                case ComparisonDirection::NE:
                  res.emplace_back(mx::not_equal(operands[0], operands[1]));
                  break;
                case ComparisonDirection::GE:
                  res.emplace_back(mx::greater_equal(operands[0], operands[1]));
                  break;
                case ComparisonDirection::GT:
                  res.emplace_back(mx::greater(operands[0], operands[1]));
                  break;
                case ComparisonDirection::LE:
                  res.emplace_back(mx::less_equal(operands[0], operands[1]));
                  break;
                case ComparisonDirection::LT:
                  res.emplace_back(mx::less(operands[0], operands[1]));
                  break;
                default:
                  res.emplace_back(mx::equal(operands[0], operands[1]));
              }
              transient_buffers.emplace(op, res);
              return true;
            })

            // Handle StableHLO Slice ops
            .Case<stablehlo::SliceOp>([&transient_buffers, &operands](auto op) {
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
              std::vector<mx::array> res = {mx::slice(
                  operands[0], start_indices, limit_indices, strides)};
              transient_buffers.emplace(op, res);
              return true;
            })
            // .Case<stablehlo::DynamicSliceOp>([](auto op) {})
            // .Case<stablehlo::DynamicUpdateSliceOp>([](auto op) {})

            // Handle StableHLO Other ops
            // .Case<stablehlo::BatchNormGradOp>([](auto op) {})
            // .Case<stablehlo::BatchNormInferenceOp>([](auto op) {})
            // .Case<stablehlo::BatchNormTrainingOp>([](auto op) {})
            // .Case<stablehlo::BitcastConvertOp>([](auto op) {})
            /*
              .Case<stablehlo::BroadcastOp>([](auto op) {})
              Deprecated see:
                https://github.com/openxla/stablehlo/issues/2340
                https://github.com/openxla/stablehlo/pull/2283
            */
            .Case<stablehlo::BroadcastInDimOp>(
                [&transient_buffers, &operands](auto op) {
                  std::vector<mx::array> res = {mx::broadcast_to(
                      operands[0], getMlxShape(mlir::cast<ShapedType>(
                                       op.getResult().getType())))};
                  transient_buffers.emplace(op, res);
                  return true;
                })
            // .Case<stablehlo::DynamicBroadcastInDimOp>([](auto op) {})
            .Case<stablehlo::CholeskyOp>(
                [&transient_buffers, &operands](auto op) {
                  std::vector<mx::array> res = {
                      mx::linalg::cholesky(operands[0], op.getLower() != 0)};
                  transient_buffers.emplace(op, res);
                  return true;
                })
            .Case<stablehlo::ClampOp>([&transient_buffers, &operands](auto op) {
              std::vector<mx::array> res = {
                  mx::clip(operands[1], operands[0], operands[2])};
              transient_buffers.emplace(op, res);
              return true;
            })
            .Case<stablehlo::ConcatenateOp>(
                [&transient_buffers, &operands](auto op) {
                  std::vector<mx::array> res = {
                      mx::concatenate(operands, op.getDimension())};
                  transient_buffers.emplace(op, res);
                  return true;
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
            .Case<stablehlo::DotGeneralOp>([&transient_buffers,
                                            &operands](auto op) {
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
              for (auto i = 0; i < operands[0].ndim(); ++i) {
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
              for (auto i = 0; i < operands[1].ndim(); ++i) {
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

              std::vector<mx::array> res = {mx::einsum(
                  lhs_subscript + "," + rhs_subscript + "->" + res_subscript,
                  {operands[0], operands[1]})};
              transient_buffers.emplace(op, res);
              return true;
            })
            /*
              .Case<stablehlo::EinsumOp>([](auto op) {})
              .Case<stablehlo::UnaryEinsumOp>([](auto op) {})
              Deprecated see:
                https://github.com/openxla/stablehlo/issues/2340
                https://github.com/openxla/stablehlo/pull/2283
            */
            // .Case<stablehlo::FftOp>([](auto op) {})
            // .Case<stablehlo::GatherOp>([](auto op) {})
            .Case<stablehlo::GetDimensionSizeOp>(
                [&transient_buffers, &operands](auto op) {
                  std::vector<mx::array> res = {
                      mx::array(operands[0].shape(op.getDimension()))};
                  transient_buffers.emplace(op, res);
                  return true;
                })
            // .Case<stablehlo::MapOp>([](auto op) {})
            .Case<stablehlo::ReshapeOp>(
                [&transient_buffers, &operands](auto op) {
                  std::vector<mx::array> res = {mx::reshape(
                      operands[0], getMlxShape(mlir::cast<ShapedType>(
                                       op.getResult().getType())))};
                  transient_buffers.emplace(op, res);
                  return true;
                })
            // .Case<stablehlo::DynamicReshapeOp>([](auto op) {})
            // .Case<stablehlo::ScatterOp>([](auto op) {})
            .Case<stablehlo::SelectOp>(
                [&transient_buffers, &operands](auto op) {
                  std::vector<mx::array> res = {
                      mx::where(operands[0], operands[1], operands[2])};
                  transient_buffers.emplace(op, res);
                  return true;
                })
            // .Case<stablehlo::SelectAndScatterOp>([](auto op) {})
            // .Case<stablehlo::SetDimensionSizeOp>([](auto op) {})
            // .Case<stablehlo::SortOp>([](auto op) {})
            // .Case<stablehlo::ReverseOp>([](auto op) {})
            // .Case<stablehlo::PadOp>([](auto op) {})
            .Case<stablehlo::TransposeOp>([&transient_buffers,
                                           &operands](auto op) {
              std::vector<int> axes;
              for (auto axis : op.getPermutation()) {
                axes.push_back(static_cast<int>(axis));
              }
              std::vector<mx::array> res = {mx::transpose(operands[0], axes)};
              transient_buffers.emplace(op, res);
              return true;
            })
            // .Case<stablehlo::TriangularSolveOp>([](auto op) {})
            // .Case<stablehlo::ReduceWindowOp>([](auto op) {})
            // .Case<stablehlo::ReturnOp>([](auto op) {})
            // .Case<stablehlo::TorchIndexSelectOp>([](auto op) {})
            // .Case<stablehlo::OptimizationBarrierOp>([](auto op) {})
            // .Case<stablehlo::CrossReplicaSumOp>([](auto op) {})

            // Handle StableHLO RNG ops
            .Case<stablehlo::RngOp>([&transient_buffers, &operands](auto op) {
              auto shape_buffer = operands[2].data<int64_t>();
              std::vector<int32_t> shape;
              for (auto i = 0; i < operands[2].size(); i++) {
                shape.push_back(static_cast<int32_t>(shape_buffer[i]));
              }
              std::vector<mx::array> res;
              if (op.getRngDistribution() == RngDistribution::UNIFORM) {
                res.emplace_back(mx::random::uniform(
                    operands[0], operands[1], shape, operands[0].dtype()));
              } else {
                res.emplace_back(mx::random::normal(
                    shape, operands[0].dtype(),
                    mx::astype(operands[0], mx::float32).item<float>(),
                    mx::astype(operands[1], mx::float32).item<float>()));
              }
              transient_buffers.emplace(op, res);
              return true;
            })
            // .Case<stablehlo::RngBitGeneratorOp>([](auto op) {})

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
            .Default([](auto op) {
              // unhandled op, interrupt walk
              return false;
            });
    if (!should_continue) return failure();
  }
  if (result.has_value()) {
    return result.value();
  } else
    return failure();
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

  FailureOr<SmallVector<DenseElementsAttr>> evalModule(
      ModuleOp& module, const SmallVector<mlir::DenseElementsAttr>& inputs) {
    // TRACE_ME_MEMBER;
    std::vector<mx::array> block_arguments;
    for (auto input : inputs) {
      block_arguments.push_back(getDenseElementsArray(input));
    }

    // TODO(@cryptodeal): operations can return multiple results,
    // will need to switch to something along the lines of
    // std::unordered_map<Operation*, std::vector<mx::array>> transient_buffers;

    // holds intermediary results of operation calls.
    std::unordered_map<Operation*, mx::array> transient_buffers;
    // module.dump();
    // std::cout << std::endl << std::endl;
    auto main = module.lookupSymbol<mlir::func::FuncOp>("main");
    auto res = evalFunc(module, main, block_arguments);
    if (failed(res)) {
      return failure();
    } else {
      SmallVector<DenseElementsAttr> result;
      for (auto& arr : res.value()) {
        mx::eval(arr);
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
    }
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
    FailureOr<SmallVector<DenseElementsAttr>> result =
        evalModule(module, inputs);
    if (failed(result)) {
      return absl::InternalError("Failed to execute module");
    }
    // LOG(INFO) << "Results: " << ToString(result.value()) << "\n";

    // Naive memory space selection, only using CPU global memory.
    PjRtMemorySpace* memory_space =
        device->default_memory_space().value_or(nullptr);
    std::vector<std::unique_ptr<PjRtBuffer>> buffer_results;
    for (auto res : result.value()) {
      buffer_results.push_back(
          CreateMlirBufferFromAttribute(res, memory_space));
    }

    std::optional<PjRtFuture<>> future;
    if (fill_future) {
      // Synchronous! To make async, this would need to return a future that is
      // ready when the computation is done.
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