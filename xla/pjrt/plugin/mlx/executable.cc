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
#include <iostream>
#include <algorithm>
#include <memory>
#include <optional>
#include <utility>
#include <cstddef>
#include <functional>
#include <unordered_map>
#include <vector>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <range/v3/view/cartesian_product.hpp>
#include <range/v3/view/indices.hpp>

// TODO(@cryptodeal): might need to update `BUILD`
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/Hashing.h"

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
#include "xla/mlir/utils/type_util.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/mlir_hlo/stablehlo_ext/transforms/passes.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/plugin/mlx/buffer.h"
#include "xla/pjrt/plugin/mlx/logging.h"
#include "xla/pjrt/plugin/mlx/utils.h"
#include "xla/service/computation_placer.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

#define DEBUG_TYPE "stablehlo-pjrt"

namespace mx = mlx::core;

typedef absl::StatusOr<mx::array> StatusOrArray;
typedef absl::StatusOr<std::vector<mx::array>> StatusOrArrays;
typedef absl::StatusOr<int> StatusOrInt;

// ZML supports up to 8 dimensions
auto index_space(const std::vector<int32_t>& dims) {
  using namespace ranges;
  // `MAX_RANK` for zml is 8
  std::vector<int32_t> used_dims(8, 1);
  std::copy(dims.begin(), dims.end(), used_dims.begin());
  return views::cartesian_product(
      views::indices(used_dims[0]), views::indices(used_dims[1]),
      views::indices(used_dims[2]), views::indices(used_dims[3]),
      views::indices(used_dims[4]), views::indices(used_dims[5]),
      views::indices(used_dims[6]), views::indices(used_dims[7]));
}

template <class Tuple,
          class T = std::decay_t<std::tuple_element_t<0, std::decay_t<Tuple>>>>
std::vector<T> to_vector(Tuple&& tuple) {
  return std::apply(
      [](auto&&... elems) {
        return std::vector<T>{std::forward<decltype(elems)>(elems)...};
      },
      std::forward<Tuple>(tuple));
}

template <class Tuple,
          class T = std::decay_t<std::tuple_element_t<0, std::decay_t<Tuple>>>>
std::vector<T> to_vector(Tuple&& tuple, size_t size) {
  std::vector<T> result = to_vector(tuple);
  result.resize(size);
  return result;
}

namespace mlir::stablehlo {

#define UNIMPLEMENTED(name) \
  xla::Unimplemented("MlxPjRtBuffer::" #name " is not implemented")

using xla::DeviceAssignment;
using xla::PjRtBuffer;
using xla::PjRtClient;
using xla::PjRtDevice;
using xla::PjRtFuture;
using xla::PjRtLoadedExecutable;
using xla::PjRtMemorySpace;

void resolveFuncDeps(ModuleOp mod, std::string func_name, Tree& dep_tree,
                     Node* parent = nullptr) {
  auto new_node = dep_tree.insert(func_name, parent);
  auto func = mod.lookupSymbol<mlir::func::FuncOp>(func_name);
  auto& block = func.front();
  std::set<std::string> func_names;
  for (auto op : block.getOps<func::CallOp>()) {
    std::string callee_name = op.getCallee().str();
    func_names.insert(callee_name);
    resolveFuncDeps(mod, callee_name, dep_tree, new_node);
  }
}

absl::Status validateModule(ModuleOp mod) {
  absl::Status status;
  // TODO: some ops should have additional checks (e.g. `stablehlo::ReduceOp`,
  // `stablehlo::SortOp`, etc.) where we need additional checks at compile time
  // to ensure that we're correctly mapping to the corresponding MLX op.
  mod.walk([&status](Operation* op) {
    // Series of checks to ensure we support the module
    return llvm::TypeSwitch<Operation*, WalkResult>(op)
        .Case<ModuleOp, func::ReturnOp, stablehlo::ReturnOp, func::CallOp,
              func::FuncOp, stablehlo::ConstantOp, stablehlo::IotaOp,
              stablehlo::AbsOp, stablehlo::CbrtOp, stablehlo::CeilOp,
              stablehlo::ConvertOp, stablehlo::CosineOp, stablehlo::ExpOp,
              stablehlo::Expm1Op, stablehlo::FloorOp, stablehlo::ImagOp,
              stablehlo::IsFiniteOp, stablehlo::LogOp, stablehlo::Log1pOp,
              stablehlo::LogisticOp, stablehlo::NotOp, stablehlo::NegOp,
              stablehlo::RealOp, stablehlo::RoundNearestEvenOp,
              stablehlo::RsqrtOp, stablehlo::SignOp, stablehlo::SineOp,
              stablehlo::SqrtOp, stablehlo::TanOp, stablehlo::TanhOp,
              stablehlo::AddOp, stablehlo::Atan2Op, stablehlo::DivOp,
              stablehlo::MaxOp, stablehlo::MinOp, stablehlo::MulOp,
              stablehlo::PowOp, stablehlo::RemOp, stablehlo::ShiftLeftOp,
              stablehlo::ShiftRightArithmeticOp, stablehlo::ShiftRightLogicalOp,
              stablehlo::SubtractOp, stablehlo::AndOp, stablehlo::OrOp,
              stablehlo::XorOp, stablehlo::ReduceOp, stablehlo::CompareOp,
              stablehlo::SliceOp, stablehlo::DynamicSliceOp,
              stablehlo::DynamicUpdateSliceOp, stablehlo::BitcastConvertOp,
              stablehlo::BroadcastInDimOp, stablehlo::ConcatenateOp,
              stablehlo::DotGeneralOp, stablehlo::GatherOp,
              stablehlo::GetDimensionSizeOp, stablehlo::ReshapeOp,
              stablehlo::ScatterOp, stablehlo::SelectOp, stablehlo::SortOp,
              stablehlo::PadOp, stablehlo::TransposeOp>([&status, &op](auto o) {
          // Validate all input types are compatible with mlx
          for (auto operand : o->getOperands()) {
            auto primitive_type = xla::ConvertMlirTypeToPrimitiveType(
                mlir::cast<ShapedType>(operand.getType()).getElementType());
            switch (primitive_type) {
              case xla::PrimitiveType::PRED:
              case xla::PrimitiveType::U8:
              case xla::PrimitiveType::S8:
              case xla::PrimitiveType::U16:
              case xla::PrimitiveType::S16:
              case xla::PrimitiveType::U32:
              case xla::PrimitiveType::S32:
              case xla::PrimitiveType::U64:
              case xla::PrimitiveType::S64:
              case xla::PrimitiveType::F16:
              case xla::PrimitiveType::BF16:
              case xla::PrimitiveType::F32:
              case xla::PrimitiveType::C64:
                break;
              default: {
                std::cout << "Unsupported type: "
                          << xla::PrimitiveType_Name(primitive_type).c_str()
                          << std::endl;
                status = absl::UnimplementedError(
                    absl::StrCat("Unsupported type: ", ToString(op)));
                return WalkResult::interrupt();
              }
            }
          }

          // Validate all result types are compatible with mlx
          for (auto result : o->getResults()) {
            auto primitive_type = xla::ConvertMlirTypeToPrimitiveType(
                mlir::cast<ShapedType>(result.getType()).getElementType());
            switch (primitive_type) {
              case xla::PrimitiveType::PRED:
              case xla::PrimitiveType::U8:
              case xla::PrimitiveType::S8:
              case xla::PrimitiveType::U16:
              case xla::PrimitiveType::S16:
              case xla::PrimitiveType::U32:
              case xla::PrimitiveType::S32:
              case xla::PrimitiveType::U64:
              case xla::PrimitiveType::S64:
              case xla::PrimitiveType::F16:
              case xla::PrimitiveType::BF16:
              case xla::PrimitiveType::F32:
              case xla::PrimitiveType::C64:
                break;
              default: {
                std::cout << "Unsupported type: "
                          << xla::PrimitiveType_Name(primitive_type).c_str()
                          << std::endl;
                status = absl::UnimplementedError(
                    absl::StrCat("Unsupported type: ", ToString(op)));
                return WalkResult::interrupt();
              }
            }
          }
          return mlir::WalkResult::advance();
        })
        // TODO(@cryptodeal): These ops were supported prior to `mlx::compile`
        // need to modify the implementation to ensure no use of `mlx::eval` and
        // ensure that `mlx::random::key` is being propogated correctly.
        .Case<stablehlo::IfOp, stablehlo::CaseOp, stablehlo::WhileOp,
              stablehlo::RngOp, stablehlo::RngBitGeneratorOp>(
            [&status, &op](auto o) {
              status = absl::UnimplementedError(
                  absl::StrCat("Unsupported op: ", ToString(op)));
              return WalkResult::interrupt();
            })
        .Default([&status, &op](auto o) {
          status = absl::UnimplementedError(
              absl::StrCat("Unsupported op: ", ToString(op)));
          return WalkResult::interrupt();
        });
  });
  return status;
}

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

absl::StatusOr<utils::ScatterType> getScatterType(mlir::Block& block) {
  unsigned op_count = 0;
  utils::ScatterType scatter_type = utils::ScatterType::Replace;
  for (const auto& op : block.getOperations()) {
    switch (op_count) {
      case 0:
        if (llvm::dyn_cast<stablehlo::ReturnOp>(op)) {
          return scatter_type;
        }
        if (llvm::dyn_cast<stablehlo::AddOp>(op)) {
          scatter_type = utils::ScatterType::Add;
        } else if (llvm::dyn_cast<stablehlo::MulOp>(op)) {
          scatter_type = utils::ScatterType::Prod;
        } else if (llvm::dyn_cast<stablehlo::MulOp>(op)) {
          scatter_type = utils::ScatterType::Prod;
        } else if (llvm::dyn_cast<stablehlo::MaxOp>(op)) {
          scatter_type = utils::ScatterType::Max;
        } else if (llvm::dyn_cast<stablehlo::MinOp>(op)) {
          scatter_type = utils::ScatterType::Min;
        } else {
          return xla::Internal("Unsupported comparator: %s", ToString(block));
        }
        break;
      case 1:
        if (llvm::dyn_cast<stablehlo::ReturnOp>(op)) {
          return scatter_type;
        }
      default:
        return xla::Internal("Unsupported comparator: %s", ToString(block));
    }
  }
}

absl::StatusOr<std::pair<unsigned, mx::ComparatorType>> getSortInfo(
    mlir::Block& block) {
  unsigned op_count = 0;
  mx::ComparatorType comparator = mx::ComparatorType::LessThan;
  unsigned input_idx = 0;
  for (const auto& op : block.getOperations()) {
    switch (op_count) {
      case 0: {
        if (auto compare_op = llvm::dyn_cast<stablehlo::CompareOp>(op)) {
          switch (compare_op.getComparisonDirection()) {
            case ComparisonDirection::GT: {
              auto lhs = llvm::dyn_cast<BlockArgument>(compare_op.getLhs());
              auto rhs = llvm::dyn_cast<BlockArgument>(compare_op.getRhs());
              if (lhs && rhs) {
                unsigned lhs_idx = static_cast<unsigned>(lhs.getArgNumber());
                unsigned rhs_idx = static_cast<unsigned>(rhs.getArgNumber());
                if (rhs_idx - 1 == lhs_idx) {
                  comparator = mx::ComparatorType::GreaterThan;
                  input_idx = lhs_idx == 0 ? 0 : lhs_idx / 2;
                  break;
                }
              }
              return xla::Internal("Unsupported comparator: %s",
                                   ToString(block));
            }
            case ComparisonDirection::LT: {
              auto lhs = llvm::dyn_cast<BlockArgument>(compare_op.getLhs());
              auto rhs = llvm::dyn_cast<BlockArgument>(compare_op.getRhs());
              if (lhs && rhs) {
                unsigned lhs_idx = static_cast<unsigned>(lhs.getArgNumber());
                unsigned rhs_idx = static_cast<unsigned>(rhs.getArgNumber());
                if (rhs_idx - 1 == lhs_idx) {
                  comparator = mx::ComparatorType::LessThan;
                  input_idx = lhs_idx == 0 ? 0 : lhs_idx / 2;
                  break;
                }
              }
              return xla::Internal("Unsupported comparator: %s",
                                   ToString(block));
            }
            default:
              return xla::Internal("Unsupported comparator: %s",
                                   ToString(block));
          }
          break;
        }
      }
      case 1:
        if (llvm::dyn_cast<stablehlo::ReturnOp>(op)) break;
      default:
        return xla::Internal("Unsupported comparator: %s", ToString(block));
    }
    op_count++;
  }
  return std::make_pair(input_idx, comparator);
}

mx::array getOperandArray(
    const Value& operand,
    const std::unordered_map<Operation*, std::vector<mx::array>>& block_ctx,
    const std::vector<mx::array>& inputs) {
  // check if operand is the result of a previous operation
  if (auto defining_op = operand.getDefiningOp()) {
    if (auto search = block_ctx.find(defining_op); search != block_ctx.end()) {
      for (auto i = 0; i < defining_op->getNumResults(); i++) {
        Value maybe_res = defining_op->getResult(i);
        if (maybe_res == operand) {
          return search->second[i];
        }
      }
    }
  } else {
    return inputs[operand.cast<BlockArgument>().getArgNumber()];
  }
}

std::pair<mx::Dtype, mx::Shape> getValueInfo(Value v) {
  auto val_type = mlir::cast<ShapedType>(v.getType());
  return std::make_pair(*utils::dtype::fromMlirType(val_type.getElementType()),
                        std::vector<int32_t>(val_type.getShape().begin(),
                                             val_type.getShape().end()));
}

std::pair<mx::Dtype, mx::Shape> getTypeInfo(Type v) {
  auto val_type = mlir::cast<ShapedType>(v);
  return std::make_pair(*utils::dtype::fromMlirType(val_type.getElementType()),
                        std::vector<int32_t>(val_type.getShape().begin(),
                                             val_type.getShape().end()));
}

CallOpInfo getCallOpInfo(ModuleOp mod, func::FuncOp o) {
  auto func_type = o.getFunctionType();

  std::vector<std::pair<mx::Dtype, mx::Shape>> input_types;
  for (auto arg : func_type.getInputs()) {
    input_types.emplace_back(getTypeInfo(arg));
  }
  std::vector<std::pair<mx::Dtype, mx::Shape>> result_types;
  for (auto result : func_type.getResults()) {
    result_types.emplace_back(getTypeInfo(result));
  }
  return CallOpInfo(input_types, result_types, o.getName().str());
}

CallOpInfo getCallOpInfo(ModuleOp mod, func::CallOp o) {
  std::vector<std::pair<mx::Dtype, mx::Shape>> input_types;
  for (auto arg : o.getOperands()) {
    input_types.emplace_back(getValueInfo(arg));
  }
  std::vector<std::pair<mx::Dtype, mx::Shape>> result_types;
  for (auto result : o.getResults()) {
    result_types.emplace_back(getValueInfo(result));
  }
  auto func = mod.lookupSymbol<mlir::func::FuncOp>(o.getCallee());
  return CallOpInfo(input_types, result_types, func.getName().str());
}

IotaOpInfo getIotaOpInfo(stablehlo::IotaOp o) {
  std::pair<mx::Dtype, mx::Shape> result_info = getValueInfo(o.getResult());
  auto iota_dim = o.getIotaDimension();
  std::vector<int32_t> dimensions;
  for (auto i = 0; i < dimensions.size(); i++) {
    dimensions.push_back(i != iota_dim ? 1 : result_info.second[i]);
  }
  return IotaOpInfo(std::vector<std::pair<mx::Dtype, mx::Shape>>{},
                    {result_info}, dimensions, result_info.first, iota_dim,
                    result_info.second);
}

CbrtOpInfo getCbrtOpInfo(stablehlo::CbrtOp o) {
  return CbrtOpInfo({getValueInfo(o.getOperand())},
                    {getValueInfo(o.getResult())});
}

BroadcastInDimOpInfo getBroadcastInDimOpInfo(stablehlo::BroadcastInDimOp o) {
  auto result_info = getValueInfo(o.getResult());
  return BroadcastInDimOpInfo(
      {getValueInfo(o.getOperand())}, {result_info},
      std::vector<int32_t>(o.getBroadcastDimensions().begin(),
                           o.getBroadcastDimensions().end()),
      result_info.second);
}

DotGeneralOpInfo getDotGeneralOpInfo(stablehlo::DotGeneralOp o) {
  auto dot_dim_nums = o.getDotDimensionNumbers();
  std::vector<int32_t> lhs_batch_dims(
      dot_dim_nums.getLhsBatchingDimensions().begin(),
      dot_dim_nums.getLhsBatchingDimensions().end());
  std::vector<int32_t> rhs_batch_dims(
      dot_dim_nums.getRhsBatchingDimensions().begin(),
      dot_dim_nums.getRhsBatchingDimensions().end());
  std::vector<int32_t> lhs_contract_dims(
      dot_dim_nums.getLhsContractingDimensions().begin(),
      dot_dim_nums.getLhsContractingDimensions().end());
  std::vector<int32_t> rhs_contract_dims(
      dot_dim_nums.getRhsContractingDimensions().begin(),
      dot_dim_nums.getRhsContractingDimensions().end());
  return DotGeneralOpInfo({getValueInfo(o.getLhs()), getValueInfo(o.getRhs())},
                          {getValueInfo(o.getResult())}, lhs_batch_dims,
                          rhs_batch_dims, lhs_contract_dims, rhs_contract_dims);
}

GatherOpInfo getGatherOpInfo(stablehlo::GatherOp o) {
  std::vector<std::pair<mx::Dtype, mx::Shape>> operand_info;
  for (auto operand : o.getOperands()) {
    operand_info.emplace_back(getValueInfo(operand));
  }
  operand_info.emplace_back(getValueInfo(o.getStartIndices()));
  auto result_info = getValueInfo(o.getResult());
  auto dim_nums = o.getDimensionNumbers();
  std::vector<int32_t> offset_dims(
      o.getDimensionNumbers().getOffsetDims().begin(),
      o.getDimensionNumbers().getOffsetDims().end());
  std::vector<int32_t> collapsed_slice_dims(
      o.getDimensionNumbers().getCollapsedSliceDims().begin(),
      o.getDimensionNumbers().getCollapsedSliceDims().end());
  std::vector<int32_t> operand_batching_dims(
      o.getDimensionNumbers().getOperandBatchingDims().begin(),
      o.getDimensionNumbers().getOperandBatchingDims().end());
  std::vector<int32_t> start_indices_batching_dims(
      o.getDimensionNumbers().getStartIndicesBatchingDims().begin(),
      o.getDimensionNumbers().getStartIndicesBatchingDims().end());
  std::vector<int32_t> start_index_map(
      o.getDimensionNumbers().getStartIndexMap().begin(),
      o.getDimensionNumbers().getStartIndexMap().end());
  auto index_vector_dim = o.getDimensionNumbers().getIndexVectorDim();
  std::vector<int32_t> slice_sizes(o.getSliceSizes().begin(),
                                   o.getSliceSizes().end());
  return GatherOpInfo(operand_info, {getValueInfo(o.getResult())},
                      collapsed_slice_dims, index_vector_dim, offset_dims,
                      operand_batching_dims, result_info.second, slice_sizes,
                      start_index_map, start_indices_batching_dims);
}

ScatterOpInfo getScatterOpInfo(stablehlo::ScatterOp o) {
  std::vector<std::pair<mx::Dtype, mx::Shape>> inputs_info;
  for (auto input : o.getInputs()) {
    inputs_info.emplace_back(getValueInfo(input));
  }
  for (auto update : o.getUpdates()) {
    inputs_info.emplace_back(getValueInfo(update));
  }
  inputs_info.emplace_back(getValueInfo(o.getScatterIndices()));
  std::vector<std::pair<mx::Dtype, std::vector<int32_t>>> results_info;
  for (auto result : o.getResults()) {
    results_info.emplace_back(getValueInfo(result));
  }
  // Get scatter dimension numbers
  auto scatter_dim_nums = o.getScatterDimensionNumbers();
  std::vector<int32_t> update_window_dims(
      scatter_dim_nums.getUpdateWindowDims().begin(),
      scatter_dim_nums.getUpdateWindowDims().end());
  std::vector<int32_t> inserted_window_dims(
      scatter_dim_nums.getInsertedWindowDims().begin(),
      scatter_dim_nums.getInsertedWindowDims().end());
  std::vector<int32_t> input_batching_dims(
      scatter_dim_nums.getInputBatchingDims().begin(),
      scatter_dim_nums.getInputBatchingDims().end());

  std::vector<int32_t> scatter_indices_batching_dims(
      scatter_dim_nums.getScatterIndicesBatchingDims().begin(),
      scatter_dim_nums.getScatterIndicesBatchingDims().end());

  std::vector<int32_t> scatter_dims_to_operand_dims(
      scatter_dim_nums.getScatterDimsToOperandDims().begin(),
      scatter_dim_nums.getScatterDimsToOperandDims().end());

  auto index_vector_dim = scatter_dim_nums.getIndexVectorDim();
  // Get update computation
  mlir::Block& update_computation = o.getUpdateComputation().front();

  // TODO(@cryptodeal): need to add a check in `Compile`
  utils::ScatterType scatter_type = *getScatterType(update_computation);
  return ScatterOpInfo(
      inputs_info, results_info, index_vector_dim, input_batching_dims,
      inserted_window_dims, scatter_dims_to_operand_dims,
      scatter_indices_batching_dims, scatter_type, update_window_dims);
}

PadOpInfo getPadOpInfo(stablehlo::PadOp o) {
  auto result_info = getValueInfo(o.getResult());
  std::vector<int32_t> edge_pad_low(o.getEdgePaddingLow().begin(),
                                    o.getEdgePaddingLow().end());
  std::vector<int32_t> interior_pad(o.getInteriorPadding().begin(),
                                    o.getInteriorPadding().end());
  return PadOpInfo(
      {getValueInfo(o.getOperand()), getValueInfo(o.getPaddingValue())},
      {result_info}, result_info.second, edge_pad_low, interior_pad);
}

void compileCallOp(OpLookup& op_lookup, ModuleOp mod, func::FuncOp o) {
  CallOpInfo call_op_info = getCallOpInfo(mod, o);
  if (auto search = op_lookup.call.find(call_op_info);
      search == op_lookup.call.end()) {
    std::cout << "Compiling CallOp: " << o.getName().str().c_str() << std::endl;
    auto& block = o.front();
    auto call_op_func = [&op_lookup, &block, mod,
                         call_op_info](const std::vector<mx::array>& inputs) {
      std::unordered_map<Operation*, std::vector<mx::array>> block_ctx;
      std::vector<mx::array> res;
      for (Operation& op : block.getOperations()) {
        res =
            llvm::TypeSwitch<Operation*, std::vector<mx::array>>(&op)
                .Case<func::ReturnOp, stablehlo::ReturnOp>([&block_ctx,
                                                            &inputs](auto o) {
                  std::vector<mx::array> res;
                  for (Value val : o.getOperands()) {
                    res.emplace_back(getOperandArray(val, block_ctx, inputs));
                  }
                  return res;
                })
                .Case<func::CallOp>([&block_ctx, &inputs, mod,
                                     &op_lookup](auto o) {
                  auto call_op_info = getCallOpInfo(mod, o);
                  std::vector<mx::array> operands;
                  for (Value val : o.getOperands()) {
                    operands.emplace_back(
                        getOperandArray(val, block_ctx, inputs));
                  }
                  return op_lookup.call.find(call_op_info)->second(operands);
                })
                // Handle StableHLO nullary ops
                .Case<stablehlo::ConstantOp>([](auto o) {
                  return std::vector<mx::array>{
                      *utils::array::fromDenseElementsAttr(
                          mlir::cast<mlir::DenseElementsAttr>(o.getValue()))};
                })
                .Case<stablehlo::IotaOp>([&op_lookup](auto o) {
                  return op_lookup.iota.find(getIotaOpInfo(o))->second({});
                })
                // .Case<stablehlo::DynamicIotaOp>([](auto o) {})
                /*
                  .Case<stablehlo::CreateTokenOp>([](auto o) {})
                  Deprecated see:
                    https://github.com/openxla/stablehlo/issues/2340
                    https://github.com/openxla/stablehlo/pull/2283
                */
                // Handle StableHLO unary elementwise op
                .Case<stablehlo::AbsOp>([&block_ctx, &inputs](auto o) {
                  return std::vector<mx::array>{mx::abs(
                      getOperandArray(o.getOperand(), block_ctx, inputs))};
                })
                .Case<stablehlo::CbrtOp>([&block_ctx, &inputs,
                                          &op_lookup](auto o) {
                  return op_lookup.cbrt.find(getCbrtOpInfo(o))
                      ->second(
                          {getOperandArray(o.getOperand(), block_ctx, inputs)});
                })
                .Case<stablehlo::CeilOp>([&block_ctx, &inputs](auto o) {
                  return std::vector<mx::array>{mx::ceil(
                      getOperandArray(o.getOperand(), block_ctx, inputs))};
                })
                .Case<stablehlo::ConvertOp>([&block_ctx, &inputs](auto o) {
                  return std::vector<mx::array>{mx::astype(
                      getOperandArray(o.getOperand(), block_ctx, inputs),
                      *utils::dtype::fromMlirType(
                          o.getResult().getType().getElementType()))};
                })
                // .Case<stablehlo::ClzOp>([](auto o) {})
                .Case<stablehlo::CosineOp>([&block_ctx, &inputs](auto o) {
                  return std::vector<mx::array>{mx::cos(
                      getOperandArray(o.getOperand(), block_ctx, inputs))};
                })
                .Case<stablehlo::ExpOp>([&block_ctx, &inputs](auto o) {
                  return std::vector<mx::array>{mx::exp(
                      getOperandArray(o.getOperand(), block_ctx, inputs))};
                })
                .Case<stablehlo::Expm1Op>([&block_ctx, &inputs](auto o) {
                  return std::vector<mx::array>{mx::expm1(
                      getOperandArray(o.getOperand(), block_ctx, inputs))};
                })
                .Case<stablehlo::FloorOp>([&block_ctx, &inputs](auto o) {
                  return std::vector<mx::array>{mx::floor(
                      getOperandArray(o.getOperand(), block_ctx, inputs))};
                })
                .Case<stablehlo::ImagOp>([&block_ctx, &inputs](auto o) {
                  return std::vector<mx::array>{mx::imag(
                      getOperandArray(o.getOperand(), block_ctx, inputs))};
                })
                .Case<stablehlo::IsFiniteOp>([&block_ctx, &inputs](auto o) {
                  return std::vector<mx::array>{mx::isfinite(
                      getOperandArray(o.getOperand(), block_ctx, inputs))};
                })
                .Case<stablehlo::LogOp>([&block_ctx, &inputs](auto o) {
                  return std::vector<mx::array>{mx::log(
                      getOperandArray(o.getOperand(), block_ctx, inputs))};
                })
                .Case<stablehlo::Log1pOp>([&block_ctx, &inputs](auto o) {
                  return std::vector<mx::array>{mx::log1p(
                      getOperandArray(o.getOperand(), block_ctx, inputs))};
                })
                .Case<stablehlo::LogisticOp>([&block_ctx, &inputs](auto o) {
                  return std::vector<mx::array>{mx::sigmoid(
                      getOperandArray(o.getOperand(), block_ctx, inputs))};
                })
                .Case<stablehlo::NotOp>([&block_ctx, &inputs](auto o) {
                  return std::vector<mx::array>{mx::logical_not(
                      getOperandArray(o.getOperand(), block_ctx, inputs))};
                })
                .Case<stablehlo::NegOp>([&block_ctx, &inputs](auto o) {
                  return std::vector<mx::array>{mx::negative(
                      getOperandArray(o.getOperand(), block_ctx, inputs))};
                })
                // .Case<stablehlo::PopulationCountOp>([](auto o) {})
                .Case<stablehlo::RealOp>([&block_ctx, &inputs](auto o) {
                  return std::vector<mx::array>{mx::real(
                      getOperandArray(o.getOperand(), block_ctx, inputs))};
                })
                // TODO(@cryptodeal): `stablehlo::RoundOp` does not match with
                // the mlx metal implementation
                // .Case<stablehlo::RoundOp>([](auto o) {})
                .Case<stablehlo::RoundNearestEvenOp>(
                    [&block_ctx, &inputs](auto o) {
                      return std::vector<mx::array>{mx::round(
                          getOperandArray(o.getOperand(), block_ctx, inputs))};
                    })
                .Case<stablehlo::RsqrtOp>([&block_ctx, &inputs](auto o) {
                  return std::vector<mx::array>{mx::rsqrt(
                      getOperandArray(o.getOperand(), block_ctx, inputs))};
                })
                .Case<stablehlo::SignOp>([&block_ctx, &inputs](auto o) {
                  return std::vector<mx::array>{mx::sign(
                      getOperandArray(o.getOperand(), block_ctx, inputs))};
                })
                .Case<stablehlo::SineOp>([&block_ctx, &inputs](auto o) {
                  return std::vector<mx::array>{mx::sin(
                      getOperandArray(o.getOperand(), block_ctx, inputs))};
                })
                .Case<stablehlo::SqrtOp>([&block_ctx, &inputs](auto o) {
                  return std::vector<mx::array>{mx::sqrt(
                      getOperandArray(o.getOperand(), block_ctx, inputs))};
                })
                .Case<stablehlo::TanOp>([&block_ctx, &inputs](auto o) {
                  return std::vector<mx::array>{mx::tan(
                      getOperandArray(o.getOperand(), block_ctx, inputs))};
                })
                .Case<stablehlo::TanhOp>([&block_ctx, &inputs](auto o) {
                  return std::vector<mx::array>{mx::tanh(
                      getOperandArray(o.getOperand(), block_ctx, inputs))};
                })
                // Handle StableHLO binary elementwise ops
                .Case<stablehlo::AddOp>([&block_ctx, &inputs](auto o) {
                  return std::vector<mx::array>{
                      mx::add(getOperandArray(o.getLhs(), block_ctx, inputs),
                              getOperandArray(o.getRhs(), block_ctx, inputs))};
                })
                .Case<stablehlo::Atan2Op>([&block_ctx, &inputs](auto o) {
                  return std::vector<mx::array>{mx::arctan2(
                      getOperandArray(o.getLhs(), block_ctx, inputs),
                      getOperandArray(o.getRhs(), block_ctx, inputs))};
                })
                // TODO(@cryptodeal): implement complex op
                // .Case<stablehlo::ComplexOp>([&block_ctx](auto
                // op) {})
                .Case<stablehlo::DivOp>([&block_ctx, &inputs](auto o) {
                  return std::vector<mx::array>{mx::divide(
                      getOperandArray(o.getLhs(), block_ctx, inputs),
                      getOperandArray(o.getRhs(), block_ctx, inputs))};
                })
                .Case<stablehlo::MaxOp>([&block_ctx, &inputs](auto o) {
                  return std::vector<mx::array>{mx::maximum(
                      getOperandArray(o.getLhs(), block_ctx, inputs),
                      getOperandArray(o.getRhs(), block_ctx, inputs))};
                })
                .Case<stablehlo::MinOp>([&block_ctx, &inputs](auto o) {
                  return std::vector<mx::array>{mx::minimum(
                      getOperandArray(o.getLhs(), block_ctx, inputs),
                      getOperandArray(o.getRhs(), block_ctx, inputs))};
                })
                .Case<stablehlo::MulOp>([&block_ctx, &inputs](auto o) {
                  return std::vector<mx::array>{mx::multiply(
                      getOperandArray(o.getLhs(), block_ctx, inputs),
                      getOperandArray(o.getRhs(), block_ctx, inputs))};
                })
                .Case<stablehlo::PowOp>([&block_ctx, &inputs](auto o) {
                  return std::vector<mx::array>{mx::power(
                      getOperandArray(o.getLhs(), block_ctx, inputs),
                      getOperandArray(o.getRhs(), block_ctx, inputs))};
                })
                .Case<stablehlo::RemOp>([&block_ctx, &inputs](auto o) {
                  return std::vector<mx::array>{mx::remainder(
                      getOperandArray(o.getLhs(), block_ctx, inputs),
                      getOperandArray(o.getRhs(), block_ctx, inputs))};
                })
                .Case<stablehlo::ShiftLeftOp>([&block_ctx, &inputs](auto o) {
                  return std::vector<mx::array>{mx::left_shift(
                      getOperandArray(o.getLhs(), block_ctx, inputs),
                      getOperandArray(o.getRhs(), block_ctx, inputs))};
                })
                .Case<stablehlo::ShiftRightArithmeticOp>(
                    [&block_ctx, &inputs](auto o) {
                      /**
                       * Per Metal Spec:
                       * For the right-shift operator, if E1 has an unsigned
                       * type or if E1 has a signed type and a nonnegative
                       * value, the vacated bits are filled with zeros. If E1
                       * has a signed type and a negative value, the vacated
                       * bits are filled with ones.
                       */
                      auto lhs = getOperandArray(o.getLhs(), block_ctx, inputs);
                      auto rhs = getOperandArray(o.getRhs(), block_ctx, inputs);
                      auto target_dtype = rhs.dtype();
                      switch (lhs.dtype().size()) {
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
                      }
                      return std::vector<mx::array>{mx::view(
                          mx::right_shift(mx::view(lhs, target_dtype),
                                          mx::astype(rhs, target_dtype)),
                          lhs.dtype())};
                    })
                .Case<stablehlo::ShiftRightLogicalOp>(
                    [&block_ctx, &inputs](auto o) {
                      // Ensures that we bitcast to `uint` type before
                      // performing the right shift. Should ensure that vacated
                      // bits are zero populated.
                      auto lhs = getOperandArray(o.getLhs(), block_ctx, inputs);
                      auto rhs = getOperandArray(o.getRhs(), block_ctx, inputs);
                      auto target_dtype = rhs.dtype();
                      switch (lhs.dtype().size()) {
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
                      }
                      return std::vector<mx::array>{mx::view(
                          mx::right_shift(mx::view(lhs, target_dtype),
                                          mx::astype(rhs, target_dtype)),
                          lhs.dtype())};
                    })
                .Case<stablehlo::SubtractOp>([&block_ctx, &inputs](auto o) {
                  return std::vector<mx::array>{mx::subtract(
                      getOperandArray(o.getLhs(), block_ctx, inputs),
                      getOperandArray(o.getRhs(), block_ctx, inputs))};
                })
                // Handle StableHLO binary logical elementwise ops
                .Case<stablehlo::AndOp>([&block_ctx, &inputs](auto o) {
                  auto lhs = getOperandArray(o.getLhs(), block_ctx, inputs);
                  auto rhs = getOperandArray(o.getRhs(), block_ctx, inputs);
                  switch (mx::kindof(lhs.dtype())) {
                    case mx::Dtype::Kind::b:
                      return std::vector<mx::array>{mx::logical_and(lhs, rhs)};
                    default:
                      return std::vector<mx::array>{mx::bitwise_and(lhs, rhs)};
                  }
                })
                .Case<stablehlo::OrOp>([&block_ctx, &inputs](auto o) {
                  auto lhs = getOperandArray(o.getLhs(), block_ctx, inputs);
                  auto rhs = getOperandArray(o.getRhs(), block_ctx, inputs);
                  switch (mx::kindof(lhs.dtype())) {
                    case mx::Dtype::Kind::b:
                      return std::vector<mx::array>{mx::logical_or(lhs, rhs)};
                    default:
                      return std::vector<mx::array>{mx::bitwise_or(lhs, rhs)};
                  }
                })
                .Case<stablehlo::XorOp>([&block_ctx, &inputs](auto o) {
                  return std::vector<mx::array>{mx::bitwise_xor(
                      getOperandArray(o.getLhs(), block_ctx, inputs),
                      getOperandArray(o.getRhs(), block_ctx, inputs))};
                })
                // Handle StableHLO communication ops
                // .Case<stablehlo::InfeedOp>([](auto o) {})
                // .Case<stablehlo::OutfeedOp>([](auto o) {})
                // .Case<stablehlo::SendOp>([](auto o) {})
                // .Case<stablehlo::RecvOp>([](auto o) {})

                // Handle StableHLO parallelism related ops
                // .Case<stablehlo::ReplicaIdOp>([](auto o) {})
                // .Case<stablehlo::PartitionIdOp>([](auto o) {})

                // Handle StableHLO control flow ops
                // .Case<stablehlo::AfterAllOp>([](auto o) {})
                // TODO(@cryptodeal): need custom kernels in order to
                // remove the need to call `mx::eval` for the following ops
                // .Case<stablehlo::IfOp>([](auto o))
                // .Case<stablehlo::CaseOp>([](auto o))
                // .Case<stablehlo::WhileOp>([](auto o))
                // .Case<stablehlo::AllGatherOp>([](auto o) {})
                // .Case<stablehlo::AllReduceOp>([](auto o) {})
                // .Case<stablehlo::ReduceScatterOp>([](auto o) {})
                // .Case<stablehlo::AllToAllOp>([](auto o) {})
                .Case<stablehlo::ReduceOp>([&block_ctx, &inputs](auto o) {
                  auto operand =
                      getOperandArray(o.getOperand(0), block_ctx, inputs);
                  if (isArgMaxReduce(o)) {
                    auto axis = static_cast<int>(o.getDimensions()[0]);
                    auto indices = mx::argmax(operand, axis);

                    mx::Dtype result_type = *utils::dtype::fromMlirType(
                        mlir::cast<ShapedType>(o.getResults()[1].getType())
                            .getElementType());
                    return std::vector<mx::array>{
                        mx::take(operand, indices, axis),
                        mx::astype(indices, result_type)};
                  }
                  if (isSumReduce(o)) {
                    return std::vector<mx::array>{mx::sum(
                        operand, static_cast<int>(o.getDimensions()[0]))};
                  }
                  if (isMaxReduce(o)) {
                    return std::vector<mx::array>{mx::max(
                        operand, static_cast<int>(o.getDimensions()[0]))};
                  }
                  if (isMinReduce(o)) {
                    return std::vector<mx::array>{mx::min(
                        operand, static_cast<int>(o.getDimensions()[0]))};
                  }
                  // TODO(@cryptodeal): further cleanup module validation
                  // done via call to `Compile` to check for valid reduce types.

                  // For now, we assume `mx::any` if none of the above

                  // if (isAnyReduce(o)) {
                  return std::vector<mx::array>{
                      mx::any(operand, static_cast<int>(o.getDimensions()[0]))};
                  //}
                })
                // Handle StableHLO tuple ops
                // .Case<stablehlo::GetTupleElementOp>([](auto o) {})
                // .Case<stablehlo::TupleOp>([](auto o) {})
                .Case<stablehlo::CompareOp>([&block_ctx, &inputs](auto o) {
                  auto lhs = getOperandArray(o.getLhs(), block_ctx, inputs);
                  auto rhs = getOperandArray(o.getRhs(), block_ctx, inputs);
                  switch (o.getComparisonDirection()) {
                    case ComparisonDirection::NE:
                      return std::vector<mx::array>{mx::not_equal(lhs, rhs)};
                    case ComparisonDirection::GE:
                      return std::vector<mx::array>{
                          mx::greater_equal(lhs, rhs)};
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
                .Case<stablehlo::SliceOp>([&block_ctx, &inputs](auto o) {
                  std::vector<int32_t> start_indices(
                      o.getStartIndices().begin(), o.getStartIndices().end());
                  std::vector<int32_t> limit_indices(
                      o.getLimitIndices().begin(), o.getLimitIndices().end());
                  std::vector<int32_t> strides(o.getStrides().begin(),
                                               o.getStrides().end());
                  return std::vector<mx::array>{mx::slice(
                      getOperandArray(o.getOperand(), block_ctx, inputs),
                      start_indices, limit_indices, strides)};
                })
                .Case<stablehlo::DynamicSliceOp>([&block_ctx, &inputs](auto o) {
                  auto operand =
                      getOperandArray(o.getOperand(), block_ctx, inputs);
                  std::vector<mx::array> indices;
                  for (auto val : o.getStartIndices()) {
                    indices.emplace_back(
                        getOperandArray(val, block_ctx, inputs));
                  }
                  auto start_indices = mx::concatenate(indices);
                  std::vector<int> axes(operand.ndim());
                  std::iota(axes.begin(), axes.end(), 0);
                  std::vector<int32_t> slice_sizes(o.getSliceSizes().begin(),
                                                   o.getSliceSizes().end());
                  return std::vector<mx::array>{
                      mx::slice(operand, start_indices, axes, slice_sizes)};
                })
                .Case<stablehlo::DynamicUpdateSliceOp>(
                    [&block_ctx, &inputs](auto o) {
                      auto operand =
                          getOperandArray(o.getOperand(), block_ctx, inputs);
                      auto update =
                          getOperandArray(o.getUpdate(), block_ctx, inputs);
                      std::vector<mx::array> indices;
                      for (auto val : o.getStartIndices()) {
                        indices.emplace_back(
                            getOperandArray(val, block_ctx, inputs));
                      }

                      auto start_indices = mx::concatenate(indices);
                      std::vector<int32_t> axes(operand.ndim());
                      std::iota(axes.begin(), axes.end(), 0);
                      return std::vector<mx::array>{mx::slice_update(
                          operand, update, start_indices, axes)};
                    })
                // Handle StableHLO Other ops
                // .Case<stablehlo::BatchNormGradOp>([](auto o) {})
                // .Case<stablehlo::BatchNormInferenceOp>([](auto o) {})
                // .Case<stablehlo::BatchNormTrainingOp>([](auto o) {})
                .Case<stablehlo::BitcastConvertOp>(
                    [&block_ctx, &inputs](auto o) {
                      auto result_type = *utils::dtype::fromMlirType(
                          mlir::cast<ShapedType>(o.getResult().getType())
                              .getElementType());
                      return std::vector<mx::array>{mx::view(
                          getOperandArray(o.getOperand(), block_ctx, inputs),
                          result_type)};
                    })
                /*
                  .Case<stablehlo::BroadcastOp>([](auto o) {})
                  Deprecated see:
                    https://github.com/openxla/stablehlo/issues/2340
                    https://github.com/openxla/stablehlo/pull/2283
                */
                .Case<stablehlo::BroadcastInDimOp>([&block_ctx, &inputs,
                                                    &op_lookup](auto o) {
                  return op_lookup.broadcast_in_dim
                      .find(getBroadcastInDimOpInfo(o))
                      ->second(
                          {getOperandArray(o.getOperand(), block_ctx, inputs)});
                })
                // .Case<stablehlo::DynamicBroadcastInDimOp>([](auto o) {})
                .Case<stablehlo::CholeskyOp>([&block_ctx, &inputs](auto o) {
                  return std::vector<mx::array>{mx::linalg::cholesky(
                      getOperandArray(o.getOperand(), block_ctx, inputs),
                      o.getLower())};
                })
                .Case<stablehlo::ClampOp>([&block_ctx, &inputs](auto o) {
                  return std::vector<mx::array>{mx::clip(
                      getOperandArray(o.getOperand(), block_ctx, inputs),
                      getOperandArray(o.getMin(), block_ctx, inputs),
                      getOperandArray(o.getMax(), block_ctx, inputs))};
                })
                .Case<stablehlo::ConcatenateOp>([&block_ctx, &inputs](auto o) {
                  std::vector<mx::array> inputs;
                  for (auto val : o.getOperands()) {
                    inputs.emplace_back(
                        getOperandArray(val, block_ctx, inputs));
                  }
                  return std::vector<mx::array>{
                      mx::concatenate(inputs, o.getDimension())};
                })
                // .Case<stablehlo::CollectiveBroadcastOp>([](auto o) {})
                // .Case<stablehlo::CollectivePermuteOp>([](auto o) {})
                // .Case<stablehlo::CompositeOp>([](auto o) {})
                // .Case<stablehlo::ConvolutionOp>([](auto o) {})
                // .Case<stablehlo::CrossReplicaSumOp>([](auto o) {})
                // .Case<stablehlo::CustomCallOp>([](auto o) {})
                /*
                  .Case<stablehlo::DotOp>([](auto o) {})
                  Deprecated see:
                    https://github.com/openxla/stablehlo/issues/2340
                    https://github.com/openxla/stablehlo/pull/2283
                */
                .Case<stablehlo::DotGeneralOp>(
                    [&block_ctx, &inputs, &op_lookup](auto o) {
                      return op_lookup.dot_general.find(getDotGeneralOpInfo(o))
                          ->second(
                              {getOperandArray(o.getLhs(), block_ctx, inputs),
                               getOperandArray(o.getRhs(), block_ctx, inputs)});
                    })
                /*
              .Case<stablehlo::EinsumOp>([](auto op) {})
              .Case<stablehlo::UnaryEinsumOp>([](auto op) {})
              Deprecated see:
                https://github.com/openxla/stablehlo/issues/2340
                https://github.com/openxla/stablehlo/pull/2283
            */
                // .Case<stablehlo::FftOp>([](auto op) {})
                // TODO(@cryptodeal): definitely room for optimization here
                .Case<stablehlo::GatherOp>([&block_ctx, &inputs,
                                            &op_lookup](auto o) {
                  return op_lookup.gather.find(getGatherOpInfo(o))
                      ->second(
                          {getOperandArray(o.getOperand(), block_ctx, inputs),
                           getOperandArray(o.getStartIndices(), block_ctx,
                                           inputs)});
                })
                .Case<stablehlo::GetDimensionSizeOp>(
                    [&block_ctx, &inputs](auto o) {
                      return std::vector<mx::array>{mx::array(
                          (getOperandArray(o.getOperand(), block_ctx, inputs))
                              .shape(o.getDimension()))};
                    })
                // .Case<stablehlo::MapOp>([](auto o) {})
                .Case<stablehlo::ReshapeOp>([&block_ctx, &inputs](auto o) {
                  auto result_type =
                      mlir::cast<ShapedType>(o.getResult().getType());
                  return std::vector<mx::array>{mx::reshape(
                      getOperandArray(o.getOperand(), block_ctx, inputs),
                      std::vector<int32_t>(result_type.getShape().begin(),
                                           result_type.getShape().end()))};
                })
                // .Case<stablehlo::DynamicReshapeOp>([](auto o) {})
                .Case<stablehlo::ScatterOp>(
                    [&block_ctx, &inputs, &op_lookup](auto o) {
                      std::vector<mx::array> operands;
                      for (auto val : o.getInputs()) {
                        operands.emplace_back(
                            getOperandArray(val, block_ctx, inputs));
                      }
                      // Add updates to input list
                      for (auto val : o.getUpdates()) {
                        operands.emplace_back(
                            getOperandArray(val, block_ctx, inputs));
                      }
                      // Add scatter indices to input list
                      operands.emplace_back(getOperandArray(
                          o.getScatterIndices(), block_ctx, inputs));
                      return op_lookup.scatter.find(getScatterOpInfo(o))
                          ->second(operands);
                    })
                .Case<stablehlo::SelectOp>([&block_ctx, &inputs](auto o) {
                  return std::vector<mx::array>{mx::where(
                      getOperandArray(o.getOperand(0), block_ctx, inputs),
                      getOperandArray(o.getOperand(1), block_ctx, inputs),
                      getOperandArray(o.getOperand(2), block_ctx, inputs))};
                })
                // .Case<stablehlo::SelectAndScatterOp>([](auto o) {})
                // .Case<stablehlo::SetDimensionSizeOp>([](auto o) {})
                .Case<stablehlo::SortOp>([&block_ctx, &inputs](auto o) {
                  std::vector<mx::array> inputs;
                  for (auto val : o.getInputs()) {
                    inputs.emplace_back(
                        getOperandArray(val, block_ctx, inputs));
                  }
                  int dim = static_cast<int>(o.getDimension());
                  // Get update computation
                  mlir::Block& comparator = o.getComparator().front();
                  auto [index, comparator_type] = *getSortInfo(comparator);
                  auto indices =
                      mx::argsort(inputs[index], dim, comparator_type);
                  std::vector<mx::array> res;
                  for (auto in : inputs) {
                    res.push_back(mx::gather(in, indices, {dim}, {1}));
                  }
                  return res;
                })
                // .Case<stablehlo::ReverseOp>([](auto o) {})
                // TODO (@cryptodeal): lots of room for optimization here
                .Case<stablehlo::PadOp>([&block_ctx, &inputs,
                                         &op_lookup](auto o) {
                  return op_lookup.pad.find(getPadOpInfo(o))
                      ->second(
                          {getOperandArray(o.getOperand(), block_ctx, inputs),
                           getOperandArray(o.getPaddingValue(), block_ctx,
                                           inputs)});
                })
                .Case<stablehlo::TransposeOp>([&block_ctx, &inputs](auto o) {
                  std::vector<int> axes(o.getPermutation().begin(),
                                        o.getPermutation().end());
                  return std::vector<mx::array>{mx::transpose(
                      getOperandArray(o.getOperand(), block_ctx, inputs),
                      axes)};
                })
                // .Case<stablehlo::TriangularSolveOp>([](auto o) {})
                // .Case<stablehlo::ReduceWindowOp>([](auto o) {})
                // .Case<stablehlo::ReturnOp>([](auto o) {})
                // .Case<stablehlo::TorchIndexSelectOp>([](auto o) {})
                // .Case<stablehlo::OptimizationBarrierOp>([](auto o) {})
                // .Case<stablehlo::CrossReplicaSumOp>([](auto o) {})
                // Need to modify such that `rng.state` is held outside of scope
                // of `mx::compile`
                // TODO(@cryptodeal): re-enable once we've handled ensuring
                // state is handled correctly
                // .Case<stablehlo::RngOp>([&block_ctx](auto o) {
                //   auto rng_func = [&o](const std::vector<mx::array>& inputs)
                //   {
                //     auto a = inputs[0];
                //     auto b = inputs[1];
                //     auto result_type =
                //         mlir::cast<ShapedType>(o.getResult().getType());
                //     std::vector<int32_t>
                //     shape(result_type.getShape().begin(),
                //                                result_type.getShape().end());

                //     switch (o.getRngDistribution()) {
                //       case RngDistribution::UNIFORM:
                //         return std::vector<mx::array>{
                //             mx::random::uniform(a, b, shape, a.dtype())};
                //       default:
                //         return std::vector<mx::array>{
                //             mx::random::normal(shape, a.dtype(), a, b)};
                //     }
                //   };
                //   return mx::compile(rng_func)(
                //       {getOperandArray(o.getA(), block_ctx),
                //        getOperandArray(o.getB(), block_ctx)});
                // })
                // .Case<stablehlo::RngBitGeneratorOp>([&block_ctx](auto o) {
                //   auto rng_bit_gen_func = [&o](const std::vector<mx::array>&
                //   inputs) {

                //   };
                //   return mx::compile(rng_bit_gen_func)(
                //       {getOperandArray(o.getA(), block_ctx),
                //        getOperandArray(o.getB(), block_ctx)});
                // })
                // Handle StableHLO Quantize ops
                // .Case<stablehlo::UniformQuantizeOp>([](auto op) {})
                // .Case<stablehlo::UniformDequantizeOp>([](auto op) {})
                // .Case<stablehlo::ReducePrecisionOp>([](auto op) {})
                /*
                  .Case<stablehlo::RealDynamicSliceOp>([](auto o) {})
                  Deprecated see:
                    https://github.com/openxla/stablehlo/issues/2340
                    https://github.com/openxla/stablehlo/pull/2283
                */
                // .Case<stablehlo::DynamicPadOp>([](auto o) {})
                // .Case<stablehlo::DynamicGatherOp>([](auto o) {})
                // .Case<stablehlo::DynamicConvOp>([](auto o) {})

                .Default([&op](auto o) {
                  // Shouldn't be possible to reach this point, return
                  // empty vector.
                  return std::vector<mx::array>{};
                });
        block_ctx.emplace(&op, res);
      }
      return res;
    };
    op_lookup.call.emplace(call_op_info, mx::compile(call_op_func));
  }
}

void compileIotaOp(OpLookup& op_lookup, stablehlo::IotaOp o) {
  IotaOpInfo iota_op_info = getIotaOpInfo(o);
  auto dimensions = std::get<0>(iota_op_info.captures);
  auto dtype = std::get<1>(iota_op_info.captures);
  auto iota_dim = std::get<2>(iota_op_info.captures);
  auto shape = std::get<3>(iota_op_info.captures);
  if (auto search = op_lookup.iota.find(iota_op_info);
      search == op_lookup.iota.end()) {
    std::cout << "Compiling IotaOp" << std::endl;
    auto iota_func = [dimensions, dtype, iota_dim,
                      shape](const std::vector<mx::array>& inputs) {
      auto init_arange =
          mx::arange(static_cast<double>(shape[iota_dim]), dtype);
      if (dimensions.size()) {
        init_arange = mx::reshape(init_arange, dimensions);
      }
      return std::vector<mx::array>{mx::broadcast_to(init_arange, shape)};
    };
    op_lookup.iota.emplace(iota_op_info, mx::compile(iota_func));
  }
}

void compileCbrtOp(OpLookup& op_lookup, stablehlo::CbrtOp o) {
  CbrtOpInfo cbrt_op_info = getCbrtOpInfo(o);
  if (auto search = op_lookup.cbrt.find(cbrt_op_info);
      search == op_lookup.cbrt.end()) {
    std::cout << "Compiling CbrtOp" << std::endl;

    auto cbrt_func = [](const std::vector<mx::array>& inputs) {
      return std::vector<mx::array>{mx::power(
          inputs[0],
          mx::full<float>(inputs[0].shape(), 1 / 3, inputs[0].dtype()))};
    };
    op_lookup.cbrt.emplace(cbrt_op_info, mx::compile(cbrt_func));
  }
}

void compileBroadcastInDimOp(OpLookup& op_lookup,
                             stablehlo::BroadcastInDimOp o) {
  BroadcastInDimOpInfo broadcast_in_dim_op_info = getBroadcastInDimOpInfo(o);
  auto broadcast_dims = std::get<0>(broadcast_in_dim_op_info.captures);
  auto target_shape = std::get<1>(broadcast_in_dim_op_info.captures);
  if (auto search = op_lookup.broadcast_in_dim.find(broadcast_in_dim_op_info);
      search == op_lookup.broadcast_in_dim.end()) {
    std::cout << "Compiling BroadcastInDimOp" << std::endl;

    auto broadcast_in_dim_func = [broadcast_dims, target_shape](
                                     const std::vector<mx::array>& inputs) {
      auto operand = inputs[0];

      // potentially reshape to pad with ones, which allows
      // broadcasting
      auto min_rank = std::min(operand.ndim(), target_shape.size());
      if (min_rank == operand.ndim()) {
        std::vector<int32_t> padded_shape = operand.shape();
        for (auto i = 0; i < target_shape.size(); i++) {
          if (auto search =
                  std::find(broadcast_dims.begin(), broadcast_dims.end(), i);
              search == broadcast_dims.end()) {
            padded_shape.insert(padded_shape.begin() + i, 1);
          }
        }
        operand = mx::reshape(operand, padded_shape);
      }
      return std::vector<mx::array>{mx::broadcast_to(operand, target_shape)};
    };
    op_lookup.broadcast_in_dim.emplace(broadcast_in_dim_op_info,
                                       mx::compile(broadcast_in_dim_func));
  }
}

void compileDotGeneralOp(OpLookup& op_lookup, stablehlo::DotGeneralOp o) {
  DotGeneralOpInfo dot_general_op_info = getDotGeneralOpInfo(o);
  auto lhs_batch_dims = std::get<0>(dot_general_op_info.captures);
  auto rhs_batch_dims = std::get<1>(dot_general_op_info.captures);
  auto lhs_contract_dims = std::get<2>(dot_general_op_info.captures);
  auto rhs_contract_dims = std::get<3>(dot_general_op_info.captures);
  if (auto search = op_lookup.dot_general.find(dot_general_op_info);
      search == op_lookup.dot_general.end()) {
    std::cout << "Compiling DotGeneralOp" << std::endl;

    auto dot_general_func = [lhs_batch_dims, rhs_batch_dims, lhs_contract_dims,
                             rhs_contract_dims](
                                const std::vector<mx::array>& inputs) {
      auto lhs = inputs[0];
      auto rhs = inputs[1];

      int dim_count = 0;
      auto getDimChar = [&dim_count]() -> char { return 'a' + dim_count++; };

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
        if (auto match = lhs_batch_map.find(i); match != lhs_batch_map.end()) {
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
        if (auto match = rhs_batch_map.find(i); match != rhs_batch_map.end()) {
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
          lhs_subscript + "," + rhs_subscript + "->" + res_subscript, inputs)};
    };
    op_lookup.dot_general.emplace(dot_general_op_info,
                                  mx::compile(dot_general_func));
  }
}

void compileGatherOp(OpLookup& op_lookup, stablehlo::GatherOp o) {
  GatherOpInfo gather_op_info = getGatherOpInfo(o);
  auto collapsed_slice_dims = std::get<0>(gather_op_info.captures);
  auto index_vector_dim = std::get<1>(gather_op_info.captures);
  auto offset_dims = std::get<2>(gather_op_info.captures);
  auto operand_batching_dims = std::get<3>(gather_op_info.captures);
  auto result_shape = std::get<4>(gather_op_info.captures);
  auto slice_sizes = std::get<5>(gather_op_info.captures);
  auto start_index_map = std::get<6>(gather_op_info.captures);
  auto start_indices_batching_dims = std::get<7>(gather_op_info.captures);

  if (auto search = op_lookup.gather.find(gather_op_info);
      search == op_lookup.gather.end()) {
    std::cout << "Compiling GatherOp" << std::endl;
    auto gather_func = [collapsed_slice_dims, index_vector_dim, offset_dims,
                        operand_batching_dims, result_shape, slice_sizes,
                        start_index_map, start_indices_batching_dims](
                           const std::vector<mx::array>& inputs) {
      auto operand = inputs[0];
      auto start_indices = inputs[1];

      // Calculate batch dims
      std::vector<int32_t> batch_dims;
      for (int64_t i = 0; i < result_shape.size(); i++) {
        if (std::find(offset_dims.begin(), offset_dims.end(), i) ==
            offset_dims.end()) {
          batch_dims.emplace_back(static_cast<int32_t>(i));
        }
      }

      std::vector<mx::array> operand_indices;

      // Iterate over result index space, populating result
      for (const auto result_index_tuple : index_space(result_shape)) {
        std::vector<int32_t> result_index =
            to_vector(result_index_tuple, result_shape.size());

        std::vector<int32_t> batch_index(batch_dims.size());
        for (unsigned i = 0; i < batch_dims.size(); i++) {
          batch_index[i] = result_index[batch_dims[i]];
        }

        // Slice start index for the current batch
        std::vector<int32_t> sin_start(start_indices.ndim());
        std::vector<int32_t> sin_stop(start_indices.ndim());
        unsigned batch_idx_count = 0;

        for (auto i = 0; i < start_indices.ndim(); i++) {
          if (index_vector_dim == static_cast<int64_t>(i)) {
            sin_start[i] = 0;
            sin_stop[i] = start_indices.shape(i) + 1;
            continue;
          }
          sin_start[i] = batch_index[batch_idx_count++];
          sin_stop[i] = sin_start[i] + 1;
        }
        mx::array start_index =
            mx::flatten(mx::slice(start_indices, sin_start, sin_stop));

        // Compute full start index
        mx::array full_start_index =
            mx::zeros({static_cast<int32_t>(operand.ndim())}, mx::int32);
        for (auto d_start = 0; d_start < start_index_map.size(); d_start++) {
          int d_operand = start_index_map[d_start];
          auto index_scalar = mx::slice(start_index, {d_start}, {d_start + 1});
          full_start_index = mx::slice_update(
              full_start_index,
              mx::clip(index_scalar, mx::array({0}, mx::int32),
                       mx::array({operand.shape(d_operand) -
                                  static_cast<int32_t>(slice_sizes[d_operand])},
                                 mx::int32)),
              {d_operand}, {d_operand + 1});
        }

        // Compute full batching index
        std::vector<int32_t> full_batching_index(operand.ndim(), 0);
        for (auto i_batching = 0; i_batching < operand_batching_dims.size();
             i_batching++) {
          auto d_operand =
              static_cast<int32_t>(operand_batching_dims[i_batching]);
          auto d_start =
              static_cast<int32_t>(start_indices_batching_dims[i_batching]);
          full_batching_index[d_operand] = batch_index
              [d_start -
               (static_cast<int64_t>(d_start) < index_vector_dim ? 0 : 1)];
        }

        // Compute offset index
        std::vector<int32_t> offset_index(offset_dims.size());
        for (unsigned i = 0; i < offset_dims.size(); i++) {
          offset_index[i] = result_index[offset_dims[i]];
        }

        // Compute full offset index
        std::vector<int32_t> full_offset_index(operand.ndim(), 0);
        unsigned offset_index_count = 0;
        for (unsigned i = 0; i < full_offset_index.size(); i++) {
          if (std::find(operand_batching_dims.begin(),
                        operand_batching_dims.end(), static_cast<int64_t>(i)) !=
                  operand_batching_dims.end() ||
              std::find(collapsed_slice_dims.begin(),
                        collapsed_slice_dims.end(), static_cast<int64_t>(i)) !=
                  collapsed_slice_dims.end()) {
            continue;
          }
          full_offset_index[i] = offset_index[offset_index_count++];
        }

        operand_indices.emplace_back(
            full_start_index +
            mx::array(full_batching_index.data(),
                      {static_cast<int32_t>(full_batching_index.size())},
                      mx::int32) +
            mx::array(full_offset_index.data(),
                      {static_cast<int32_t>(full_offset_index.size())},
                      mx::int32));
      }
      std::vector<int32_t> gather_slice_sizes(operand.ndim(), 1);
      std::vector<int32_t> gather_axes(operand.ndim());
      std::iota(gather_axes.begin(), gather_axes.end(), 0);
      return std::vector<mx::array>{
          mx::reshape(mx::gather(operand,
                                 mx::split(mx::stack(operand_indices, 1),
                                           operand_indices[0].shape(0), 0),
                                 gather_axes, gather_slice_sizes),
                      result_shape)};
    };

    op_lookup.gather.emplace(gather_op_info, mx::compile(gather_func));
  }
}
void compileScatterOp(OpLookup& op_lookup, stablehlo::ScatterOp o) {
  ScatterOpInfo scatter_op_info = getScatterOpInfo(o);
  auto index_vector_dim = std::get<0>(scatter_op_info.captures);
  auto input_batching_dims = std::get<1>(scatter_op_info.captures);
  auto inserted_window_dims = std::get<2>(scatter_op_info.captures);
  auto scatter_dims_to_operand_dims = std::get<3>(scatter_op_info.captures);
  auto scatter_indices_batching_dims = std::get<4>(scatter_op_info.captures);
  auto scatter_type = std::get<5>(scatter_op_info.captures);
  auto update_window_dims = std::get<6>(scatter_op_info.captures);
  if (auto search = op_lookup.scatter.find(scatter_op_info);
      search == op_lookup.scatter.end()) {
    std::cout << "Compiling ScatterOp" << std::endl;
    auto scatter_func = [index_vector_dim, input_batching_dims,
                         inserted_window_dims, scatter_dims_to_operand_dims,
                         scatter_indices_batching_dims, scatter_type,
                         update_window_dims](const std::vector<mx::array>& in) {
      auto input_count = (in.size() - 1) / 2;
      std::vector<mx::array> inputs(in.begin(), in.begin() + input_count);
      std::vector<mx::array> updates(in.begin() + input_count, in.end() - 1);
      auto scatter_indices = in[in.size() - 1];

      std::vector<mx::array> input_indices;
      std::vector<mx::array> update_indices;

      // iterate over updates[0] index space
      for (const auto update_index_tuple : index_space(updates[0].shape())) {
        std::vector<int32_t> update_index = to_vector(
            update_index_tuple, static_cast<size_t>(updates[0].ndim()));

        // Calculate update scatter dims
        std::vector<int32_t> update_scatter_dims;
        for (auto i = 0; i < updates[0].ndim(); i++) {
          if (std::find(update_window_dims.begin(), update_window_dims.end(),
                        static_cast<int64_t>(i)) == update_window_dims.end()) {
            update_scatter_dims.emplace_back(static_cast<int32_t>(i));
          }
        }

        // Calculate update scatter index
        std::vector<int32_t> update_scatter_index(update_scatter_dims.size());
        for (auto i = 0; i < update_scatter_dims.size(); i++) {
          update_scatter_index[i] = update_index[update_scatter_dims[i]];
        }

        // Slice start index
        std::vector<int32_t> sin_start = update_scatter_index;
        std::vector<int32_t> sin_stop = update_scatter_index;
        if (index_vector_dim < scatter_indices.ndim()) {
          sin_start.insert(sin_start.begin() + index_vector_dim, 0);
          sin_stop.insert(sin_stop.begin() + index_vector_dim,
                          scatter_indices.shape(index_vector_dim));
        }
        for (auto& d : sin_stop) d += 1;
        mx::array start_index =
            mx::flatten(mx::slice(scatter_indices, sin_start, sin_stop));

        // Compute full start index
        mx::array full_start_index =
            mx::zeros({static_cast<int>(inputs[0].ndim())}, mx::int32);
        for (auto i = 0; i < scatter_dims_to_operand_dims.size(); i++) {
          auto d_input = static_cast<int32_t>(scatter_dims_to_operand_dims[i]);
          full_start_index = mx::slice_update(
              full_start_index, mx::slice(start_index, {i}, {i + 1}), {d_input},
              {d_input + 1});
        }

        // Compute full batching index
        mx::array full_batching_index =
            mx::zeros({static_cast<int>(inputs[0].ndim())}, mx::int32);
        for (auto i = 0; i < input_batching_dims.size(); i++) {
          int32_t d_input = input_batching_dims[i];
          int32_t d_start = scatter_indices_batching_dims[i];
          full_batching_index = mx::slice_update(
              full_batching_index,
              mx::array(
                  {update_scatter_index[d_start -
                                        (d_start < index_vector_dim ? 0 : 1)]}),
              {d_input}, {d_input + 1});
        }

        // Compute update window index
        std::vector<int32_t> update_window_index(update_window_dims.size());
        for (auto i = 0; i < update_window_dims.size(); i++) {
          update_window_index[i] = update_index[update_window_dims[i]];
        }

        // Compute full window index
        mx::array full_window_index =
            mx::zeros({static_cast<int32_t>(update_window_index.size() +
                                            inserted_window_dims.size() +
                                            input_batching_dims.size())},
                      mx::int32);
        unsigned update_window_index_count = 0;
        for (int32_t i = 0; i < full_window_index.size(); i++) {
          if (std::find(inserted_window_dims.begin(),
                        inserted_window_dims.end(),
                        i) != inserted_window_dims.end() ||
              std::find(input_batching_dims.begin(), input_batching_dims.end(),
                        i) != input_batching_dims.end()) {
            continue;
          }
          full_window_index = mx::slice_update(
              full_window_index,
              mx::array({update_window_index[update_window_index_count++]}),
              {i}, {i + 1});
        }

        // Compute result index
        mx::array result_index =
            full_start_index + full_batching_index + full_window_index;

        // TODO (@cryptodeal): rework this so we can still leverage
        // `mx::compile` OR ensure this cannot happen.
        // if (mx::sum(
        //         result_index >=
        //         mx::array(reinterpret_cast<const int32_t*>(
        //                       result_shape.data()),
        //                   {static_cast<int32_t>(result_shape.size())}))
        //         .item<int32_t>()) {
        //   continue;
        // }

        input_indices.push_back(result_index);
        update_indices.push_back(mx::array(
            update_index.data(), {static_cast<int>(updates[0].ndim())}));
      }
      if (update_indices.empty()) {
        return inputs;
      }
      std::vector<int32_t> scatter_axes(inputs[0].ndim());
      std::iota(scatter_axes.begin(), scatter_axes.end(), 0);
      std::vector<int32_t> gather_axes(updates[0].ndim());
      std::iota(gather_axes.begin(), gather_axes.end(), 0);
      std::vector<int32_t> gather_slice_sizes(updates[0].ndim(), 1);
      std::vector<mx::array> res;
      auto result_indices =
          mx::split(mx::stack(input_indices, 1), input_indices[0].shape(0), 0);
      auto gather_indices = mx::split(mx::stack(update_indices, 1),
                                      update_indices[0].shape(0), 0);
      auto idx_shape = gather_indices[0].shape();
      std::vector<int32_t> update_shape(inputs[0].ndim() + idx_shape.size(), 1);
      for (auto i = 0; i < idx_shape.size(); ++i) {
        update_shape[i] = idx_shape[i];
      }
      for (auto i = 0; i < inputs.size(); ++i) {
        auto update_vals =
            mx::reshape(mx::gather(updates[i], gather_indices, gather_axes,
                                   gather_slice_sizes),
                        update_shape);
        switch (scatter_type) {
          case utils::ScatterType::Replace:
            res.push_back(mx::scatter(inputs[i], result_indices, update_vals,
                                      scatter_axes));
            break;
          case utils::ScatterType::Add:
            res.push_back(mx::scatter_add(inputs[i], result_indices,
                                          update_vals, scatter_axes));
            break;
          case utils::ScatterType::Prod:
            res.push_back(mx::scatter_prod(inputs[i], result_indices,
                                           update_vals, scatter_axes));
            break;
          case utils::ScatterType::Max:
            res.push_back(mx::scatter_max(inputs[i], result_indices,
                                          update_vals, scatter_axes));
            break;
          case utils::ScatterType::Min:
            res.push_back(mx::scatter_min(inputs[i], result_indices,
                                          update_vals, scatter_axes));
            break;
        }
      }
      return res;
    };
    op_lookup.scatter.emplace(scatter_op_info, mx::compile(scatter_func));
  }
}
void compilePadOp(OpLookup& op_lookup, stablehlo::PadOp o) {
  PadOpInfo pad_op_info = getPadOpInfo(o);
  auto result_shape = std::get<0>(pad_op_info.captures);
  auto edge_pad_low = std::get<1>(pad_op_info.captures);
  auto interior_pad = std::get<2>(pad_op_info.captures);
  if (auto search = op_lookup.pad.find(pad_op_info);
      search == op_lookup.pad.end()) {
    std::cout << "Compiling PadOp" << std::endl;
    auto pad_func = [result_shape, edge_pad_low,
                     interior_pad](const std::vector<mx::array>& inputs) {
      auto operand = inputs[0];
      auto padding_value = inputs[1];
      auto edge_padding_low =
          mx::array(edge_pad_low.data(),
                    {static_cast<int32_t>(edge_pad_low.size())}, mx::int32);
      auto interior_padding =
          mx::array(interior_pad.data(),
                    {static_cast<int32_t>(edge_pad_low.size())}, mx::int32);

      // initialize array full of padding values
      mx::array result = mx::full(result_shape, padding_value);
      std::vector<mx::array> scatter_indices;
      std::vector<mx::array> gather_indices;
      // assign values to correct indices in result
      for (const auto operand_index_tuple : index_space(operand.shape())) {
        std::vector<int32_t> operand_index =
            to_vector(operand_index_tuple, static_cast<size_t>(operand.ndim()));
        mx::array result_index =
            edge_padding_low +
            mx::array(operand_index.data(),
                      {static_cast<int32_t>(operand_index.size())}, mx::int32) *
                (interior_padding +
                 mx::full<int32_t>(interior_padding.shape(), 1));
        scatter_indices.push_back(result_index);
        gather_indices.push_back(mx::array(
            operand_index.data(), {static_cast<int32_t>(operand.ndim())}));
      }
      std::vector<int32_t> axes(operand.ndim());
      std::iota(axes.begin(), axes.end(), 0);
      std::vector<int32_t> gather_slice_sizes(operand.ndim(), 1);
      auto target_indices = mx::split(mx::stack(scatter_indices, 1),
                                      scatter_indices[0].shape(0), 0);
      auto input_indices = mx::split(mx::stack(gather_indices, 1),
                                     gather_indices[0].shape(0), 0);
      auto idx_shape = input_indices[0].shape();
      std::vector<int32_t> update_shape(operand.ndim() + idx_shape.size(), 1);
      for (auto i = 0; i < idx_shape.size(); ++i) {
        update_shape[i] = idx_shape[i];
      }
      auto update_vals =
          mx::reshape(mx::gather(operand, input_indices, axes,
                                 std::vector<int32_t>(operand.ndim(), 1)),
                      update_shape);
      return std::vector<mx::array>{
          mx::scatter(result, target_indices, update_vals, axes)};
    };
    op_lookup.pad.emplace(pad_op_info, mx::compile(pad_func));
  }
}

void compileModule(ModuleOp mod, OpLookup& op_lookup) {
  // the top level block to compile
  // auto main = mod.lookupSymbol<mlir::func::FuncOp>("main");

  Tree dep_tree;
  resolveFuncDeps(mod, "main", dep_tree);

  // walk each function and pre-compile lookup table
  // for all StableHLO op variants
  for (auto op_name : dep_tree.getAllFnNames()) {
    auto func = mod.lookupSymbol<mlir::func::FuncOp>(op_name);
    auto& block = func.front();
    for (auto& op : block.getOperations()) {
      llvm::TypeSwitch<Operation*, void>(&op)
          .Case<stablehlo::IotaOp>(
              [&op_lookup](auto o) { compileIotaOp(op_lookup, o); })
          .Case<stablehlo::CbrtOp>(
              [&op_lookup](auto o) { compileCbrtOp(op_lookup, o); })
          .Case<stablehlo::BroadcastInDimOp>(
              [&op_lookup](auto o) { compileBroadcastInDimOp(op_lookup, o); })
          .Case<stablehlo::DotGeneralOp>(
              [&op_lookup](auto o) { compileDotGeneralOp(op_lookup, o); })
          .Case<stablehlo::GatherOp>(
              [&op_lookup](auto o) { compileGatherOp(op_lookup, o); })
          .Case<stablehlo::ScatterOp>(
              [&op_lookup](auto o) { compileScatterOp(op_lookup, o); })
          .Case<stablehlo::PadOp>(
              [&op_lookup](auto o) { compilePadOp(op_lookup, o); })
          .Default([](auto o) {});
    }
  }

  while (true) {
    auto compilable_fns = dep_tree.getCompilableFns();
    if (compilable_fns.empty()) {
      break;
    }
    for (auto fn_name : compilable_fns) {
      std::cout << fn_name.c_str() << std::endl;
      auto func = mod.lookupSymbol<mlir::func::FuncOp>(fn_name);
      compileCallOp(op_lookup, mod, func);
      dep_tree.remove(fn_name);
    }
  }
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
    TRACE_ME_MEMBER;
    auto main = module.lookupSymbol<mlir::func::FuncOp>("main");
    std::cout << "Started call to `mlx::compile`" << std::endl << std::endl;
    compileModule(module, op_lookup_);
    // try {
    auto& main_block = main.front();
    compiled_module_ =
        op_lookup_.call.find(getCallOpInfo(module, main))->second;

    // force compilation by running w zeroed inputs
    std::vector<mx::array> tmp_inputs;
    for (auto arg : main_block.getArguments()) {
      auto shaped_type = mlir::cast<ShapedType>(arg.getType());
      tmp_inputs.push_back(
          mx::zeros(std::vector<int32_t>(shaped_type.getShape().begin(),
                                         shaped_type.getShape().end()),
                    *utils::dtype::fromMlirType(shaped_type.getElementType())));
    }
    compiled_module_(tmp_inputs);
    std::cout << "Finished call to `mlx::compile`" << std::endl << std::endl;
    // } catch (std::exception& e) {
    //   std::cout << "Error in `mlx::compile`" << std::endl;
    // }
  }

  static absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(
      ModuleOp module, DeviceAssignment assignment,
      absl::Span<PjRtDevice* const> devices, PjRtClient* client) {
    // TRACE_ME;

    mlir::BaseScopedDiagnosticHandler diagnostic_handler(module->getContext());
    if (failed(decomposeChloToStablehlo(module))) {
      return diagnostic_handler.ConsumeStatus();
    }

    // Simplify the graph using available HWI passes.
    if (failed(runHardwareIndependentOptimizations(module))) {
      return diagnostic_handler.ConsumeStatus();
    }

    auto module_status = validateModule(module);
    if (!module_status.ok()) {
      return module_status;
    }

    // std::cout << ToString(module).c_str() << std::endl << std::endl;

    auto executable = std::make_unique<MlirLoadedExecutable>(module, assignment,
                                                             devices, client);

    return executable;
  }

  PjRtClient* client() const override {
    TRACE_ME_MEMBER;
    return client_;
  }

  const DeviceAssignment& device_assignment() const override {
    TRACE_ME_MEMBER;
    return assignment_;
  }

  absl::Span<const PjRtLoadedExecutable::LogicalDeviceIds>
  addressable_device_logical_ids() const override {
    TRACE_ME_MEMBER;
    LOG_UNIMPLEMENTED(addressable_device_logical_ids);
    return {};
  }

  absl::Span<PjRtDevice* const> addressable_devices() const override {
    TRACE_ME_MEMBER;
    return devices_;
  }

  // Helper function to get default mem from device.
  PjRtMemorySpace* get_default_memory_space() const {
    TRACE_ME_MEMBER;
    return devices_[0]->default_memory_space().value_or(nullptr);
  }

  // absl::StatusOr<std::vector<mx::array>> evalModule(
  //     ModuleOp& module, const SmallVector<mlir::DenseElementsAttr>& inputs) {
  //   TRACE_ME_MEMBER;
  //   std::vector<mx::array> block_arguments;
  //   for (auto input : inputs) {
  //     TF_ASSIGN_OR_RETURN(auto input_arr,
  //                         utils::array::fromDenseElementsAttr(input));
  //     block_arguments.emplace_back(input_arr);
  //   }

  //   // std::cout << std::endl;
  //   // module.dump();
  //   // std::cout << std::endl;
  //   auto main = module.lookupSymbol<mlir::func::FuncOp>("main");
  //   std::unordered_map<
  //       mlir::Block*,
  //       std::tuple<std::vector<mx::array>,
  //                  std::unordered_map<Operation*, std::vector<mx::array>>>>
  //       block_ctx;
  //   TF_ASSIGN_OR_RETURN(auto mlx_res, evalBlock(module, main.front(),
  //                                               block_arguments, block_ctx));

  //   for (auto& a : mlx_res) {
  //     a = mx::contiguous(a);
  //   }
  //   mx::eval(mlx_res);
  //   return mlx_res;
  // }

  absl::StatusOr<PjRtLoadedExecutable::Result> ExecuteWithMlxInterpreter(
      absl::Span<PjRtBuffer* const> argument_handles, ModuleOp module,
      PjRtDevice* device, bool fill_future) {
    TRACE_ME_MEMBER;
    SmallVector<DenseElementsAttr> input_attrs;
    for (auto* arg : argument_handles) {
      TF_ASSIGN_OR_RETURN(auto mlirArg, GetAttributeFromBuffer(arg));
      auto mlirArgInModuleContext =
          CloneIntoContext(mlirArg, *module->getContext());
      input_attrs.push_back(mlirArgInModuleContext);
    }
    // LOG(INFO) << "EvalModule:\n" << ToString(module) << "\n";
    // LOG(INFO) << "Inputs: " << ToString(inputs) << "\n";
    // TF_ASSIGN_OR_RETURN(auto mlx_res, evalModule(module, inputs));

    std::vector<mx::array> inputs;
    for (auto& attr : input_attrs) {
      TF_ASSIGN_OR_RETURN(auto input,
                          utils::array::fromDenseElementsAttr(attr));
      inputs.push_back(input);
    }

    auto mlx_res = compiled_module_(inputs);
    mx::eval(mlx_res);

    // LOG(INFO) << "Results: " << ToString(result.value()) << "\n";

    // Naive memory space selection, only using CPU global memory.
    PjRtMemorySpace* memory_space =
        device->default_memory_space().value_or(nullptr);
    std::vector<std::unique_ptr<PjRtBuffer>> buffer_results;
    for (auto i = 0; i < mlx_res.size(); i++) {
      buffer_results.push_back(
          CreateMlirBufferFromMlxArray(mlx_res[i], memory_space));
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
    TRACE_ME_MEMBER;
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
    TRACE_ME_MEMBER;
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
    TRACE_ME_MEMBER;
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
    TRACE_ME_MEMBER;
    module_.release();
    module_ = nullptr;
  }
  bool IsDeleted() override {
    TRACE_ME_MEMBER;
    return !module_;
  }

  // PjRtExecutable API.
  int num_replicas() const override {
    TRACE_ME_MEMBER;
    return assignment_.replica_count();
  }
  int num_partitions() const override {
    TRACE_ME_MEMBER;
    return assignment_.computation_count();
  }
  int64_t SizeOfGeneratedCodeInBytes() const override {
    // No generated code.. so just return 1.
    TRACE_ME_MEMBER;
    return 1;
  }
  absl::string_view name() const override {
    TRACE_ME_MEMBER;
    return name_;
  }

  absl::StatusOr<std::vector<std::shared_ptr<xla::HloModule>>> GetHloModules()
      const override {
    // TODO: This shouldn't be needed for an MLIR plugin, its only used in the
    // JAX layer for determining output sharding, which exists on the mlir
    // module.
    TRACE_ME_MEMBER;
    auto moduleClone = llvm::cast<ModuleOp>(module_.get()->clone());
    TF_ASSIGN_OR_RETURN(auto hlo_module,
                        xla::ConvertStablehloToHlo(moduleClone));
    return std::vector<std::shared_ptr<xla::HloModule>>{std::move(hlo_module)};
  }

  absl::StatusOr<std::vector<std::vector<absl::string_view>>>
  GetOutputMemoryKinds() const override {
    TRACE_ME_MEMBER;
    return UNIMPLEMENTED(GetOutputMemoryKinds);
  }

  absl::StatusOr<std::string> FingerprintExecutable() const override {
    TRACE_ME_MEMBER;
    return UNIMPLEMENTED(FingerprintExecutable);
  }

 private:
  std::string name_;
  DeviceAssignment assignment_;
  absl::Span<PjRtDevice* const> devices_;
  PjRtClient* client_;
  std::function<std::vector<mx::array>(const std::vector<mx::array>&)>
      compiled_module_;
  OpLookup op_lookup_;

  // MLIR
  MLIRContext context_;
  mlir::OwningOpRef<ModuleOp> module_;
};

absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> StablehloMlxCompile(
    mlir::ModuleOp module, DeviceAssignment assignment, PjRtClient* client) {
  // TRACE_ME;
  return MlirLoadedExecutable::Compile(module, assignment, client->devices(),
                                       client);
}

absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> StablehloMlxCompile(
    xla::XlaComputation const& computation, xla::DeviceAssignment assignment,
    xla::PjRtClient* client) {
  // TRACE_ME;
  MLIRContext context;
  TF_ASSIGN_OR_RETURN(auto module,
                      ConvertHloToStablehlo(context, &computation.proto()));
  return StablehloMlxCompile(module.get(), assignment, client);
}

}  // namespace mlir::stablehlo
