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
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/mlir_hlo/stablehlo_ext/transforms/passes.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/plugin/mlx/buffer.h"
#include "xla/pjrt/plugin/mlx/logging.h"
#include "xla/pjrt/plugin/mlx/utils.h"
#include "xla/service/computation_placer.h"
#include "tsl/platform/statusor.h"

#define DEBUG_TYPE "stablehlo-pjrt"

namespace mx = mlx::core;

typedef absl::StatusOr<mx::array> StatusOrArray;
typedef absl::StatusOr<std::vector<mx::array>> StatusOrArrays;
typedef absl::StatusOr<int> StatusOrInt;

struct {
  std::vector<mx::array> block_args;
  std::unordered_map<mlir::Operation*, std::vector<mx::array>>
      transient_buffers;
} BlockCtx;

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

int32_t currentStartIndex(mx::array& index_scalar) {
  switch (index_scalar.dtype()) {
    case mx::int64:
      return static_cast<int32_t>(index_scalar.item<int64_t>());
    case mx::int16:
      return static_cast<int32_t>(index_scalar.item<int16_t>());
    case mx::int8:
      return static_cast<int32_t>(index_scalar.item<int8_t>());
    default:
      return index_scalar.item<int32_t>();
  }
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

absl::StatusOr<mx::array> getOperandArray(
    const Value& operand,
    std::unordered_map<
        mlir::Block*,
        std::tuple<std::vector<mx::array>,
                   std::unordered_map<Operation*, std::vector<mx::array>>>>&
        block_ctx) {
  // check if operand is the result of a previous operation
  if (auto defining_op = operand.getDefiningOp()) {
    mlir::Block* parent_block = defining_op->getBlock();
    if (auto search = block_ctx.find(parent_block); search != block_ctx.end()) {
      if (auto op_search = std::get<1>(search->second).find(defining_op);
          op_search != std::get<1>(search->second).end()) {
        for (auto i = 0; i < defining_op->getNumResults(); i++) {
          Value maybe_res = defining_op->getResult(i);
          if (maybe_res == operand) {
            return op_search->second[i];
          }
        }
      }
    }
  } else {
    // check if operand is a block argument
    auto block_arg = operand.cast<BlockArgument>();
    mlir::Block* parent_block = block_arg.getOwner();
    if (auto search = block_ctx.find(parent_block); search != block_ctx.end()) {
      auto [block_args, transient_buffers] = search->second;
      return block_args[block_arg.getArgNumber()];
    }
  }
  return absl::InternalError("Failed to find array for operand");
}

absl::StatusOr<std::vector<mx::array>> evalBlock(
    ModuleOp& module, mlir::Block& block, std::vector<mx::array> args,
    std::unordered_map<
        mlir::Block*,
        std::tuple<std::vector<mx::array>,
                   std::unordered_map<Operation*, std::vector<mx::array>>>>&
        block_ctx) {
  auto [ctx, added] = block_ctx.emplace(
      &block,
      std::make_tuple(
          args, std::unordered_map<Operation*, std::vector<mx::array>>{}));
  Operation* result_op = nullptr;
  for (Operation& op : block.getOperations()) {
    // op.dump();

    // switch on the operation type
    auto maybe_result =
        llvm::TypeSwitch<Operation*, StatusOrArrays>(&op)
            // TODO(@cryptodeal): handle `func` namespace ops
            .Case<func::ReturnOp, stablehlo::ReturnOp>(
                [&block_ctx, &result_op](auto op) -> StatusOrArrays {
                  std::vector<mx::array> res;
                  for (Value val : op.getOperands()) {
                    TF_ASSIGN_OR_RETURN(auto result_array,
                                        getOperandArray(val, block_ctx));
                    res.emplace_back(result_array);
                  }
                  result_op = op;
                  return res;
                })
            .Case<func::CallOp>([&block_ctx,
                                 &module](auto op) -> StatusOrArrays {
              auto callee =
                  module.lookupSymbol<mlir::func::FuncOp>(op.getCallee());
              std::vector<mx::array> operands;
              for (Value val : op.getOperands()) {
                TF_ASSIGN_OR_RETURN(auto operand_array,
                                    getOperandArray(val, block_ctx));
                operands.emplace_back(operand_array);
              }
              auto scoped_ctx = block_ctx;
              return evalBlock(module, callee.front(), operands, scoped_ctx);
            })
            // Handle StableHLO nullary ops
            .Case<stablehlo::ConstantOp>([](auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(
                  auto value,
                  utils::array::fromDenseElementsAttr(
                      mlir::cast<mlir::DenseElementsAttr>(op.getValue())));
              return std::vector<mx::array>{value};
            })
            .Case<stablehlo::IotaOp>([](auto op) -> StatusOrArrays {
              auto result_type =
                  mlir::cast<ShapedType>(op.getResult().getType());
              std::vector<int32_t> shape(result_type.getShape().begin(),
                                         result_type.getShape().end());
              auto iota_dimension = op.getIotaDimension();
              std::vector<int32_t> dimensions;
              for (auto i = 0; i < dimensions.size(); i++) {
                dimensions.push_back(i != iota_dimension ? 1 : shape[i]);
              }
              TF_ASSIGN_OR_RETURN(
                  mx::Dtype mlx_result_type,
                  utils::dtype::fromMlirType(result_type.getElementType()));
              auto init_arange = mx::arange(
                  static_cast<double>(shape[iota_dimension]), mlx_result_type);
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
            .Case<stablehlo::AbsOp>([&block_ctx](auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto operand,
                                  getOperandArray(op.getOperand(), block_ctx));
              return std::vector<mx::array>{mx::abs(operand)};
            })
            .Case<stablehlo::CbrtOp>([&block_ctx](auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto operand,
                                  getOperandArray(op.getOperand(), block_ctx));
              return std::vector<mx::array>{mx::power(
                  operand,
                  mx::full<float>(operand.shape(), 1 / 3, operand.dtype()))};
            })
            .Case<stablehlo::CeilOp>([&block_ctx](auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto operand,
                                  getOperandArray(op.getOperand(), block_ctx));
              return std::vector<mx::array>{mx::ceil(operand)};
            })
            .Case<stablehlo::ConvertOp>([&block_ctx](
                                            auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto operand,
                                  getOperandArray(op.getOperand(), block_ctx));
              TF_ASSIGN_OR_RETURN(
                  mx::Dtype result_type,
                  utils::dtype::fromMlirType(
                      op.getResult().getType().getElementType()));
              return std::vector<mx::array>{mx::astype(operand, result_type)};
            })
            // .Case<stablehlo::ClzOp>([](auto op) {})
            .Case<stablehlo::CosineOp>([&block_ctx](auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto operand,
                                  getOperandArray(op.getOperand(), block_ctx));
              return std::vector<mx::array>{mx::cos(operand)};
            })
            .Case<stablehlo::ExpOp>([&block_ctx](auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto operand,
                                  getOperandArray(op.getOperand(), block_ctx));
              return std::vector<mx::array>{mx::exp(operand)};
            })
            .Case<stablehlo::Expm1Op>([&block_ctx](auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto operand,
                                  getOperandArray(op.getOperand(), block_ctx));
              return std::vector<mx::array>{mx::expm1(operand)};
            })
            .Case<stablehlo::FloorOp>([&block_ctx](auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto operand,
                                  getOperandArray(op.getOperand(), block_ctx));
              return std::vector<mx::array>{mx::floor(operand)};
            })
            .Case<stablehlo::ImagOp>([&block_ctx](auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto operand,
                                  getOperandArray(op.getOperand(), block_ctx));
              return std::vector<mx::array>{mx::imag(operand)};
            })
            .Case<stablehlo::IsFiniteOp>([&block_ctx](
                                             auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto operand,
                                  getOperandArray(op.getOperand(), block_ctx));
              return std::vector<mx::array>{mx::isfinite(operand)};
            })
            .Case<stablehlo::LogOp>([&block_ctx](auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto operand,
                                  getOperandArray(op.getOperand(), block_ctx));
              return std::vector<mx::array>{mx::log(operand)};
            })
            .Case<stablehlo::Log1pOp>([&block_ctx](auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto operand,
                                  getOperandArray(op.getOperand(), block_ctx));
              return std::vector<mx::array>{mx::log1p(operand)};
            })
            .Case<stablehlo::LogisticOp>([&block_ctx](
                                             auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto operand,
                                  getOperandArray(op.getOperand(), block_ctx));
              return std::vector<mx::array>{mx::sigmoid(operand)};
            })
            .Case<stablehlo::NotOp>([&block_ctx](auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto operand,
                                  getOperandArray(op.getOperand(), block_ctx));
              return std::vector<mx::array>{mx::logical_not(operand)};
            })
            .Case<stablehlo::NegOp>([&block_ctx](auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto operand,
                                  getOperandArray(op.getOperand(), block_ctx));
              return std::vector<mx::array>{mx::negative(operand)};
            })
            // .Case<stablehlo::PopulationCountOp>([](auto op) {})
            .Case<stablehlo::RealOp>([&block_ctx](auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto operand,
                                  getOperandArray(op.getOperand(), block_ctx));
              return std::vector<mx::array>{mx::real(operand)};
            })
            // `stablehlo::RoundOp` does not match with the mlx metal
            // implementation .Case<stablehlo::RoundOp>([](auto op) {})
            .Case<stablehlo::RoundNearestEvenOp>(
                [&block_ctx](auto op) -> StatusOrArrays {
                  TF_ASSIGN_OR_RETURN(
                      auto operand,
                      getOperandArray(op.getOperand(), block_ctx));
                  return std::vector<mx::array>{mx::round(operand)};
                })
            .Case<stablehlo::RsqrtOp>([&block_ctx](auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto operand,
                                  getOperandArray(op.getOperand(), block_ctx));
              return std::vector<mx::array>{mx::rsqrt(operand)};
            })
            .Case<stablehlo::SignOp>([&block_ctx](auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto operand,
                                  getOperandArray(op.getOperand(), block_ctx));
              return std::vector<mx::array>{mx::sign(operand)};
            })
            .Case<stablehlo::SineOp>([&block_ctx](auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto operand,
                                  getOperandArray(op.getOperand(), block_ctx));
              return std::vector<mx::array>{mx::sin(operand)};
            })
            .Case<stablehlo::SqrtOp>([&block_ctx](auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto operand,
                                  getOperandArray(op.getOperand(), block_ctx));
              return std::vector<mx::array>{mx::sqrt(operand)};
            })
            .Case<stablehlo::TanOp>([&block_ctx](auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto operand,
                                  getOperandArray(op.getOperand(), block_ctx));
              return std::vector<mx::array>{mx::tan(operand)};
            })
            .Case<stablehlo::TanhOp>([&block_ctx](auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto operand,
                                  getOperandArray(op.getOperand(), block_ctx));
              return std::vector<mx::array>{mx::tanh(operand)};
            })

            // Handle StableHLO binary elementwise ops
            .Case<stablehlo::AddOp>([&block_ctx](auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto lhs,
                                  getOperandArray(op.getLhs(), block_ctx));
              TF_ASSIGN_OR_RETURN(auto rhs,
                                  getOperandArray(op.getRhs(), block_ctx));
              return std::vector<mx::array>{mx::add(lhs, rhs)};
            })
            .Case<stablehlo::Atan2Op>([&block_ctx](auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto lhs,
                                  getOperandArray(op.getLhs(), block_ctx));
              TF_ASSIGN_OR_RETURN(auto rhs,
                                  getOperandArray(op.getRhs(), block_ctx));
              return std::vector<mx::array>{mx::arctan2(lhs, rhs)};
            })
            // TODO(@cryptodeal): implement complex op
            // .Case<stablehlo::ComplexOp>([&block_ctx](auto
            // op) {})
            .Case<stablehlo::DivOp>([&block_ctx](auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto lhs,
                                  getOperandArray(op.getLhs(), block_ctx));
              TF_ASSIGN_OR_RETURN(auto rhs,
                                  getOperandArray(op.getRhs(), block_ctx));
              return std::vector<mx::array>{mx::divide(lhs, rhs)};
            })
            .Case<stablehlo::MaxOp>([&block_ctx](auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto lhs,
                                  getOperandArray(op.getLhs(), block_ctx));
              TF_ASSIGN_OR_RETURN(auto rhs,
                                  getOperandArray(op.getRhs(), block_ctx));
              return std::vector<mx::array>{mx::maximum(lhs, rhs)};
            })
            .Case<stablehlo::MinOp>([&block_ctx](auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto lhs,
                                  getOperandArray(op.getLhs(), block_ctx));
              TF_ASSIGN_OR_RETURN(auto rhs,
                                  getOperandArray(op.getRhs(), block_ctx));
              return std::vector<mx::array>{mx::minimum(lhs, rhs)};
            })
            .Case<stablehlo::MulOp>([&block_ctx](auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto lhs,
                                  getOperandArray(op.getLhs(), block_ctx));
              TF_ASSIGN_OR_RETURN(auto rhs,
                                  getOperandArray(op.getRhs(), block_ctx));
              return std::vector<mx::array>{mx::multiply(lhs, rhs)};
            })
            .Case<stablehlo::PowOp>([&block_ctx](auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto lhs,
                                  getOperandArray(op.getLhs(), block_ctx));
              TF_ASSIGN_OR_RETURN(auto rhs,
                                  getOperandArray(op.getRhs(), block_ctx));
              return std::vector<mx::array>{mx::power(lhs, rhs)};
            })
            .Case<stablehlo::RemOp>([&block_ctx](auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto lhs,
                                  getOperandArray(op.getLhs(), block_ctx));
              TF_ASSIGN_OR_RETURN(auto rhs,
                                  getOperandArray(op.getRhs(), block_ctx));
              return std::vector<mx::array>{mx::remainder(lhs, rhs)};
            })
            .Case<stablehlo::ShiftLeftOp>(
                [&block_ctx](auto op) -> StatusOrArrays {
                  TF_ASSIGN_OR_RETURN(auto lhs,
                                      getOperandArray(op.getLhs(), block_ctx));
                  TF_ASSIGN_OR_RETURN(auto rhs,
                                      getOperandArray(op.getRhs(), block_ctx));
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
                [&block_ctx](auto op) -> StatusOrArrays {
                  TF_ASSIGN_OR_RETURN(auto lhs,
                                      getOperandArray(op.getLhs(), block_ctx));
                  auto lhs_dtype = lhs.dtype();
                  TF_ASSIGN_OR_RETURN(auto rhs,
                                      getOperandArray(op.getRhs(), block_ctx));
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
                [&block_ctx](auto op) -> StatusOrArrays {
                  TF_ASSIGN_OR_RETURN(auto lhs,
                                      getOperandArray(op.getLhs(), block_ctx));
                  auto lhs_dtype = lhs.dtype();
                  TF_ASSIGN_OR_RETURN(auto rhs,
                                      getOperandArray(op.getRhs(), block_ctx));
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
                [&block_ctx](auto op) -> StatusOrArrays {
                  TF_ASSIGN_OR_RETURN(mx::array lhs,
                                      getOperandArray(op.getLhs(), block_ctx));
                  TF_ASSIGN_OR_RETURN(mx::array rhs,
                                      getOperandArray(op.getRhs(), block_ctx));
                  return std::vector<mx::array>{mx::subtract(lhs, rhs)};
                })

            // Handle StableHLO binary logical elementwise ops
            .Case<stablehlo::AndOp>([&block_ctx](auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto lhs,
                                  getOperandArray(op.getLhs(), block_ctx));
              TF_ASSIGN_OR_RETURN(auto rhs,
                                  getOperandArray(op.getRhs(), block_ctx));
              if (mx::kindof(lhs.dtype()) == mx::Dtype::Kind::b) {
                return std::vector<mx::array>{mx::logical_and(lhs, rhs)};
              } else {
                return std::vector<mx::array>{mx::bitwise_and(lhs, rhs)};
              }
            })
            .Case<stablehlo::OrOp>([&block_ctx](auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto lhs,
                                  getOperandArray(op.getLhs(), block_ctx));
              TF_ASSIGN_OR_RETURN(auto rhs,
                                  getOperandArray(op.getRhs(), block_ctx));

              if (mx::kindof(lhs.dtype()) == mx::Dtype::Kind::b) {
                return std::vector<mx::array>{mx::logical_or(lhs, rhs)};
              } else {
                return std::vector<mx::array>{mx::bitwise_or(lhs, rhs)};
              }
            })
            .Case<stablehlo::XorOp>([&block_ctx](auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto lhs,
                                  getOperandArray(op.getLhs(), block_ctx));
              TF_ASSIGN_OR_RETURN(auto rhs,
                                  getOperandArray(op.getRhs(), block_ctx));
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
            // TODO (@cryptodeal): probably need to rework how we handle block
            // context
            .Case<stablehlo::WhileOp>(
                [&module, &block_ctx](auto op) -> StatusOrArrays {
                  std::vector<mx::array> operand;
                  std::unordered_set<llvm::hash_code> operand_hashes;
                  for (Value val : op.getOperand()) {
                    TF_ASSIGN_OR_RETURN(mx::array operand_array,
                                        getOperandArray(val, block_ctx));
                    operand.emplace_back(operand_array);
                    operand_hashes.emplace(mlir::hash_value(val));
                  }

                  // Get cond block
                  mlir::Block& cond = op.getCond().front();
                  // Get body block
                  mlir::Block& body = op.getBody().front();

                  // execute while loop
                  while (true) {
                    auto loop_ctx_copy = block_ctx;
                    // Evaluate condition block
                    TF_ASSIGN_OR_RETURN(
                        std::vector<mx::array> cond_res,
                        evalBlock(module, cond, operand, loop_ctx_copy));
                    if (!mx::astype(cond_res[0], mx::bool_).item<bool>()) {
                      break;
                    }

                    // Evaluate body block
                    TF_ASSIGN_OR_RETURN(
                        std::vector<mx::array> body_res,
                        evalBlock(module, body, operand, loop_ctx_copy));
                    mx::eval(body_res);
                    operand = body_res;
                  }

                  return operand;
                })

            // .Case<stablehlo::AllGatherOp>([](auto op) {})
            // .Case<stablehlo::AllReduceOp>([](auto op) {})
            // .Case<stablehlo::ReduceScatterOp>([](auto op) {})
            // .Case<stablehlo::AllToAllOp>([](auto op) {})

            // TODO (@cryptodeal): `stablehlo::ReduceOp` currently only supports
            // specific well defined use cases in zml. Longer term, should
            // support custom use cases of `zml.ops.reduce`.
            .Case<stablehlo::ReduceOp>([&block_ctx](auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto operand,
                                  getOperandArray(op.getOperand(0), block_ctx));
              if (isArgMaxReduce(op)) {
                auto axis = static_cast<int>(op.getDimensions()[0]);
                auto indices = mx::argmax(operand, axis);
                TF_ASSIGN_OR_RETURN(
                    mx::Dtype result_type,
                    utils::dtype::fromMlirType(
                        mlir::cast<ShapedType>(op.getResults()[1].getType())
                            .getElementType()));
                return std::vector<mx::array>{mx::take(operand, indices, axis),
                                              mx::astype(indices, result_type)};
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

            .Case<stablehlo::CompareOp>([&block_ctx](
                                            auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto lhs,
                                  getOperandArray(op.getLhs(), block_ctx));
              TF_ASSIGN_OR_RETURN(auto rhs,
                                  getOperandArray(op.getRhs(), block_ctx));
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
            .Case<stablehlo::SliceOp>([&block_ctx](auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto operand,
                                  getOperandArray(op.getOperand(), block_ctx));
              std::vector<int32_t> start_indices(op.getStartIndices().begin(),
                                                 op.getStartIndices().end());
              std::vector<int32_t> limit_indices(op.getLimitIndices().begin(),
                                                 op.getLimitIndices().end());
              std::vector<int32_t> strides(op.getStrides().begin(),
                                           op.getStrides().end());
              return std::vector<mx::array>{
                  mx::slice(operand, start_indices, limit_indices, strides)};
            })
            .Case<stablehlo::DynamicSliceOp>([&block_ctx](
                                                 auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(mx::array operand,
                                  getOperandArray(op.getOperand(), block_ctx));
              auto op_start_indices = op.getStartIndices();
              std::vector<mx::array> start_indices_list;
              for (Value val : op_start_indices) {
                TF_ASSIGN_OR_RETURN(mx::array indice_value,
                                    getOperandArray(val, block_ctx));
                start_indices_list.emplace_back(indice_value);
              }
              mx::array start_indices = mx::concatenate(start_indices_list);
              std::vector<int> axes(operand.ndim());
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
                [&block_ctx](auto op) -> StatusOrArrays {
                  TF_ASSIGN_OR_RETURN(
                      auto operand,
                      getOperandArray(op.getOperand(), block_ctx));
                  TF_ASSIGN_OR_RETURN(
                      auto update, getOperandArray(op.getUpdate(), block_ctx));
                  auto op_start_indices = op.getStartIndices();
                  std::vector<mx::array> start_indices_list;
                  for (Value val : op_start_indices) {
                    TF_ASSIGN_OR_RETURN(mx::array indice_value,
                                        getOperandArray(val, block_ctx));
                    start_indices_list.emplace_back(indice_value);
                  }
                  mx::array start_indices = mx::concatenate(start_indices_list);
                  std::vector<int32_t> axes(operand.ndim());
                  std::iota(axes.begin(), axes.end(), 0);
                  return std::vector<mx::array>{
                      mx::slice_update(operand, update, start_indices, axes)};
                })

            // Handle StableHLO Other ops
            // .Case<stablehlo::BatchNormGradOp>([](auto op) {})
            // .Case<stablehlo::BatchNormInferenceOp>([](auto op) {})
            // .Case<stablehlo::BatchNormTrainingOp>([](auto op) {})
            .Case<stablehlo::BitcastConvertOp>([&block_ctx](
                                                   auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto operand,
                                  getOperandArray(op.getOperand(), block_ctx));
              TF_ASSIGN_OR_RETURN(
                  mx::Dtype result_type,
                  utils::dtype::fromMlirType(
                      mlir::cast<ShapedType>(op.getResult().getType())
                          .getElementType()));
              return std::vector<mx::array>{mx::view(operand, result_type)};
            })
            /*
              .Case<stablehlo::BroadcastOp>([](auto op) {})
              Deprecated see:
                https://github.com/openxla/stablehlo/issues/2340
                https://github.com/openxla/stablehlo/pull/2283
            */
            .Case<stablehlo::BroadcastInDimOp>([&block_ctx](
                                                   auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(mx::array operand,
                                  getOperandArray(op.getOperand(), block_ctx));
              auto result_type =
                  mlir::cast<ShapedType>(op.getResult().getType());
              std::vector<int32_t> target_shape(result_type.getShape().begin(),
                                                result_type.getShape().end());
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
            .Case<stablehlo::CholeskyOp>([&block_ctx](
                                             auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto operand,
                                  getOperandArray(op.getOperand(), block_ctx));
              return std::vector<mx::array>{
                  mx::linalg::cholesky(operand, op.getLower() != 0)};
            })
            .Case<stablehlo::ClampOp>([&block_ctx](auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto a_min,
                                  getOperandArray(op.getMin(), block_ctx));
              TF_ASSIGN_OR_RETURN(auto a,
                                  getOperandArray(op.getOperand(), block_ctx));
              TF_ASSIGN_OR_RETURN(auto a_max,
                                  getOperandArray(op.getMax(), block_ctx));
              return std::vector<mx::array>{mx::clip(a, a_min, a_max)};
            })
            .Case<stablehlo::ConcatenateOp>(
                [&block_ctx](auto op) -> StatusOrArrays {
                  std::vector<mx::array> operands;
                  for (Value val : op.getOperands()) {
                    TF_ASSIGN_OR_RETURN(auto operand_array,
                                        getOperandArray(val, block_ctx));
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
            .Case<stablehlo::DotGeneralOp>([&block_ctx](
                                               auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto lhs,
                                  getOperandArray(op.getLhs(), block_ctx));
              TF_ASSIGN_OR_RETURN(auto rhs,
                                  getOperandArray(op.getRhs(), block_ctx));
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
            .Case<stablehlo::GatherOp>([&block_ctx](auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(mx::array operand,
                                  getOperandArray(op.getOperand(), block_ctx));
              TF_ASSIGN_OR_RETURN(
                  mx::array start_indices,
                  getOperandArray(op.getStartIndices(), block_ctx));

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
              auto result_type =
                  mlir::cast<ShapedType>(op.getResult().getType());
              std::vector<int32_t> result_shape(result_type.getShape().begin(),
                                                result_type.getShape().end());

              // Calculate batch dims
              std::vector<int32_t> batch_dims;
              for (int64_t i = 0; i < result_shape.size(); i++) {
                if (std::find(offset_dims.begin(), offset_dims.end(), i) ==
                    offset_dims.end()) {
                  batch_dims.emplace_back(static_cast<int32_t>(i));
                }
              }
              mx::array result = mx::zeros(result_shape, operand.dtype());

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
                std::vector<int32_t> full_start_index(operand.ndim(), 0);
                for (auto d_start = 0; d_start < start_index_map.size();
                     d_start++) {
                  auto d_operand = start_index_map[d_start];
                  auto index_scalar =
                      mx::slice(start_index, {d_start}, {d_start + 1});
                  full_start_index[d_operand] = std::clamp(
                      currentStartIndex(index_scalar), 0,
                      operand.shape(d_operand) -
                          static_cast<int32_t>(slice_sizes[d_operand]));
                }

                // Compute full batching index
                std::vector<int32_t> full_batching_index(operand.ndim(), 0);
                for (auto i_batching = 0;
                     i_batching < operand_batching_dims.size(); i_batching++) {
                  auto d_operand =
                      static_cast<int32_t>(operand_batching_dims[i_batching]);
                  auto d_start = static_cast<int32_t>(
                      start_indices_batching_dims[i_batching]);
                  full_batching_index[d_operand] =
                      batch_index[d_start - (static_cast<int64_t>(d_start) <
                                                     index_vector_dim
                                                 ? 0
                                                 : 1)];
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
                                operand_batching_dims.end(),
                                static_cast<int64_t>(i)) !=
                          operand_batching_dims.end() ||
                      std::find(collapsed_slice_dims.begin(),
                                collapsed_slice_dims.end(),
                                static_cast<int64_t>(i)) !=
                          collapsed_slice_dims.end()) {
                    continue;
                  }
                  full_offset_index[i] = offset_index[offset_index_count++];
                }

                std::vector<int32_t> operand_index(operand.ndim());
                for (unsigned i = 0; i < operand.ndim(); i++) {
                  operand_index[i] = full_start_index[i] +
                                     full_batching_index[i] +
                                     full_offset_index[i];
                }

                // slice gathered value
                std::vector<int32_t> operand_stop = operand_index;
                for (auto& d : operand_stop) d += 1;
                std::vector<int32_t> result_stop = result_index;
                for (auto& d : result_stop) d += 1;

                result =
                    mx::slice_update(result,
                                     mx::flatten(mx::slice(
                                         operand, operand_index, operand_stop)),
                                     result_index, result_stop);
              }
              return std::vector<mx::array>{result};
            })
            .Case<stablehlo::GetDimensionSizeOp>(
                [&block_ctx](auto op) -> StatusOrArrays {
                  TF_ASSIGN_OR_RETURN(
                      mx::array operand,
                      getOperandArray(op.getOperand(), block_ctx));
                  return std::vector<mx::array>{
                      mx::array(operand.shape(op.getDimension()))};
                })
            // .Case<stablehlo::MapOp>([](auto op) {})
            .Case<stablehlo::ReshapeOp>([&block_ctx](
                                            auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto operand,
                                  getOperandArray(op.getOperand(), block_ctx));
              auto result_type =
                  mlir::cast<ShapedType>(op.getResult().getType());
              return std::vector<mx::array>{mx::reshape(
                  operand, std::vector<int32_t>(result_type.getShape().begin(),
                                                result_type.getShape().end()))};
            })
            // .Case<stablehlo::DynamicReshapeOp>([](auto op) {})
            .Case<stablehlo::ScatterOp>([&block_ctx,
                                         &module](auto op) -> StatusOrArrays {
              // Get list of input(s)
              std::vector<mx::array> inputs;
              for (Value val : op.getInputs()) {
                TF_ASSIGN_OR_RETURN(mx::array input_array,
                                    getOperandArray(val, block_ctx));
                inputs.emplace_back(input_array);
              }

              // Get scatter indices
              TF_ASSIGN_OR_RETURN(
                  mx::array scatter_indices,
                  getOperandArray(op.getScatterIndices(), block_ctx));

              // Get list of updates
              std::vector<mx::array> updates;
              for (Value val : op.getUpdates()) {
                TF_ASSIGN_OR_RETURN(mx::array update_array,
                                    getOperandArray(val, block_ctx));

                updates.emplace_back(update_array);
              }

              // Get scatter dimension numbers
              auto scatter_dimension_numbers = op.getScatterDimensionNumbers();
              auto update_window_dims =
                  scatter_dimension_numbers.getUpdateWindowDims();
              auto inserted_window_dims =
                  scatter_dimension_numbers.getInsertedWindowDims();
              auto input_batching_dims =
                  scatter_dimension_numbers.getInputBatchingDims();
              auto scatter_indices_batching_dims =
                  scatter_dimension_numbers.getScatterIndicesBatchingDims();
              auto scatter_dims_to_operand_dims =
                  scatter_dimension_numbers.getScatterDimsToOperandDims();
              auto index_vector_dim =
                  scatter_dimension_numbers.getIndexVectorDim();

              // Get other scatter parameters
              auto indices_are_sorted = op.getIndicesAreSorted();
              auto unique_indices = op.getUniqueIndices();
              auto result_type =
                  mlir::cast<ShapedType>(op.getResults()[0].getType());
              std::vector<int32_t> result_shape(result_type.getShape().begin(),
                                                result_type.getShape().end());

              // Get update computation
              mlir::Block& update_computation =
                  op.getUpdateComputation().front();
              std::vector<mx::array> results = inputs;
              // iterate over updates[0] index space
              for (const auto update_index_tuple :
                   index_space(updates[0].shape())) {
                std::vector<int32_t> update_index = to_vector(
                    update_index_tuple, static_cast<size_t>(updates[0].ndim()));

                // printVector("update_index", update_index);  // DEBUG

                // Calculate update scatter dims
                std::vector<int32_t> update_scatter_dims;
                for (auto i = 0; i < updates[0].ndim(); i++) {
                  if (std::find(update_window_dims.begin(),
                                update_window_dims.end(),
                                static_cast<int64_t>(i)) ==
                      update_window_dims.end()) {
                    update_scatter_dims.emplace_back(static_cast<int32_t>(i));
                  }
                }

                // Calculate update scatter index
                std::vector<int32_t> update_scatter_index(
                    update_scatter_dims.size());
                for (auto i = 0; i < update_scatter_dims.size(); i++) {
                  update_scatter_index[i] =
                      update_index[update_scatter_dims[i]];
                }

                // printVector("update_scatter_index", update_scatter_index,
                //             true);  // DEBUG

                // Slice start index
                std::vector<int32_t> sin_start = update_scatter_index;
                std::vector<int32_t> sin_stop = update_scatter_index;
                if (index_vector_dim < scatter_indices.ndim()) {
                  sin_start.insert(sin_start.begin() + index_vector_dim, 0);
                  sin_stop.insert(sin_stop.begin() + index_vector_dim,
                                  scatter_indices.shape(index_vector_dim));
                }
                for (auto& d : sin_stop) d += 1;
                mx::array start_index = mx::flatten(
                    mx::slice(scatter_indices, sin_start, sin_stop));

                // Compute full start index
                mx::array full_start_index =
                    mx::zeros({static_cast<int>(inputs[0].ndim())}, mx::int32);
                for (auto i = 0; i < scatter_dims_to_operand_dims.size(); i++) {
                  auto d_input =
                      static_cast<int32_t>(scatter_dims_to_operand_dims[i]);
                  full_start_index = mx::slice_update(
                      full_start_index, mx::slice(start_index, {i}, {i + 1}),
                      {d_input}, {d_input + 1});
                }

                // Compute full batching index
                mx::array full_batching_index =
                    mx::zeros({static_cast<int>(inputs[0].ndim())}, mx::int32);
                for (auto i = 0; i < input_batching_dims.size(); i++) {
                  int32_t d_input = input_batching_dims[i];
                  int32_t d_start = scatter_indices_batching_dims[i];
                  full_batching_index = mx::slice_update(
                      full_batching_index,
                      mx::array({update_scatter_index
                                     [d_start -
                                      (d_start < index_vector_dim ? 0 : 1)]}),
                      {d_input}, {d_input + 1});
                }

                // Compute update window index
                std::vector<int32_t> update_window_index(
                    update_window_dims.size());
                for (auto i = 0; i < update_window_dims.size(); i++) {
                  update_window_index[i] = update_index[update_window_dims[i]];
                }

                // Compute full window index
                mx::array full_window_index = mx::zeros(
                    {static_cast<int32_t>(update_window_index.size() +
                                          inserted_window_dims.size() +
                                          input_batching_dims.size())},
                    mx::int32);
                unsigned update_window_index_count = 0;
                for (int32_t i = 0; i < full_window_index.size(); i++) {
                  if (std::find(inserted_window_dims.begin(),
                                inserted_window_dims.end(),
                                i) != inserted_window_dims.end() ||
                      std::find(input_batching_dims.begin(),
                                input_batching_dims.end(),
                                i) != input_batching_dims.end()) {
                    continue;
                  }
                  full_window_index = mx::slice_update(
                      full_window_index,
                      mx::array(
                          {update_window_index[update_window_index_count++]}),
                      {i}, {i + 1});
                }

                // Compute result index
                mx::array result_index =
                    full_start_index + full_batching_index + full_window_index;

                // Continue if result index is out of bounds
                if (mx::sum(
                        result_index >=
                        mx::array(reinterpret_cast<const int32_t*>(
                                      result_shape.data()),
                                  {static_cast<int32_t>(result_shape.size())}))
                        .item<int32_t>()) {
                  continue;
                }

                std::vector<int32_t> result_slice_axes(results[0].ndim());
                std::iota(result_slice_axes.begin(), result_slice_axes.end(),
                          0);
                std::vector<int32_t> result_slice_sizes(results[0].ndim(), 1);
                std::vector<int32_t> update_slice_stop = update_index;
                for (auto& d : update_slice_stop) d += 1;
                for (auto i = 0; i < results.size(); i++) {
                  auto scoped_ctx = block_ctx;
                  TF_ASSIGN_OR_RETURN(
                      std::vector<mx::array> update,
                      evalBlock(
                          module, update_computation,
                          {mx::slice(results[i], result_index,
                                     result_slice_axes, result_slice_sizes),
                           mx::slice(updates[i], update_index,
                                     update_slice_stop)},
                          scoped_ctx));
                  results[i] = mx::slice_update(
                      results[i], update[0], result_index, result_slice_axes);
                }
              }
              return results;
            })
            .Case<stablehlo::SelectOp>([&block_ctx](auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto condition,
                                  getOperandArray(op.getOperand(0), block_ctx));
              TF_ASSIGN_OR_RETURN(auto x,
                                  getOperandArray(op.getOperand(1), block_ctx));
              TF_ASSIGN_OR_RETURN(auto y,
                                  getOperandArray(op.getOperand(2), block_ctx));
              return std::vector<mx::array>{mx::where(condition, x, y)};
            })
            // .Case<stablehlo::SelectAndScatterOp>([](auto op) {})
            // .Case<stablehlo::SetDimensionSizeOp>([](auto op) {})
            // TODO (@cryptodeal): finish implementation
            // .Case<stablehlo::SortOp>([&block_ctx](auto op) ->
            // StatusOrArrays {
            //   // Get list of input(s)
            //   std::vector<mx::array> inputs;
            //   for (Value val : op.getInputs()) {
            //     TF_ASSIGN_OR_RETURN(
            //         mx::array input_array,
            //         getOperandArray(val, block_ctx));
            //     inputs.emplace_back(input_array);
            //   }

            //   auto dimension = op.getDimension();
            //   auto is_stable = op.getIsStable();

            //   // Get update computation
            //   mlir::Block& comparator =
            //       op.getComparator().front();

            // })
            // .Case<stablehlo::ReverseOp>([](auto op) {})
            .Case<stablehlo::PadOp>([&block_ctx](auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(mx::array operand,
                                  getOperandArray(op.getOperand(), block_ctx));

              TF_ASSIGN_OR_RETURN(
                  mx::array padding_value,
                  getOperandArray(op.getPaddingValue(), block_ctx));
              mx::array edge_padding_low = mx::array(
                  op.getEdgePaddingLow().data(),
                  {static_cast<int32_t>(op.getEdgePaddingLow().size())},
                  mx::int64);

              mx::array interior_padding = mx::array(
                  op.getInteriorPadding().data(),
                  {static_cast<int32_t>(op.getInteriorPadding().size())},
                  mx::int64);

              auto result_type =
                  mlir::cast<ShapedType>(op.getResult().getType());
              std::vector<int32_t> result_shape(result_type.getShape().begin(),
                                                result_type.getShape().end());

              // initialize array full of padding values
              mx::array result = mx::full(result_shape, padding_value);
              // assign values to correct indices in result
              for (const auto operand_index_tuple :
                   index_space(operand.shape())) {
                std::vector<int32_t> operand_index = to_vector(
                    operand_index_tuple, static_cast<size_t>(operand.ndim()));
                mx::array result_index =
                    edge_padding_low +
                    mx::array(operand_index.data(),
                              {static_cast<int32_t>(operand_index.size())},
                              mx::int32) *
                        (interior_padding +
                         mx::full<int32_t>(interior_padding.shape(), 1));

                std::vector<int> result_axes(result.ndim());
                std::iota(result_axes.begin(), result_axes.end(), 0);
                std::vector<int32_t> operand_stop = operand_index;
                for (auto& d : operand_stop) d += 1;
                result = mx::slice_update(
                    result, mx::slice(operand, operand_index, operand_stop),
                    result_axes, std::vector<int32_t>(result.ndim(), 1));
              }
              return std::vector<mx::array>{result};
            })
            .Case<stablehlo::TransposeOp>([&block_ctx](
                                              auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(auto operand,
                                  getOperandArray(op.getOperand(), block_ctx));
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
            .Case<stablehlo::RngOp>([&block_ctx](auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(mx::array a,
                                  getOperandArray(op.getA(), block_ctx));
              TF_ASSIGN_OR_RETURN(mx::array b,
                                  getOperandArray(op.getB(), block_ctx));
              TF_ASSIGN_OR_RETURN(mx::array dims,
                                  getOperandArray(op.getShape(), block_ctx));
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
            .Case<stablehlo::RngBitGeneratorOp>([&block_ctx](
                                                    auto op) -> StatusOrArrays {
              TF_ASSIGN_OR_RETURN(
                  mx::array initial_state,
                  getOperandArray(op.getInitialState(), block_ctx));
              initial_state = mx::astype(initial_state, mx::uint32);
              // whereas stablehlo prng state is an array of `uint64_t`, mlx
              // is an array of `uint32_t`
              auto output_type =
                  mlir::cast<ShapedType>(op.getOutput().getType());
              std::vector<int32_t> out_shape(output_type.getShape().begin(),
                                             output_type.getShape().end());
              // TODO (@cryptodeal): implement `THREE_FRY` and `PHILOX`
              // algos
              auto res = mx::random::bits(
                  out_shape,
                  utils::dtype::mlirTypeByteWidth(output_type.getElementType()),
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
      (&std::get<1>(ctx->second))->emplace(&op, *maybe_result);
    } else {
      return maybe_result.status();
    }
  }

  return (&std::get<1>(ctx->second))->find(result_op)->second;
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

  absl::StatusOr<
      std::tuple<SmallVector<DenseElementsAttr>, std::vector<mx::array>>>
  evalModule(ModuleOp& module,
             const SmallVector<mlir::DenseElementsAttr>& inputs) {
    // TRACE_ME_MEMBER;
    std::vector<mx::array> block_arguments;
    for (auto input : inputs) {
      TF_ASSIGN_OR_RETURN(auto input_arr,
                          utils::array::fromDenseElementsAttr(input));
      block_arguments.emplace_back(input_arr);
    }

    // std::cout << std::endl;
    // module.dump();
    // std::cout << std::endl;
    auto main = module.lookupSymbol<mlir::func::FuncOp>("main");
    std::unordered_map<
        mlir::Block*,
        std::tuple<std::vector<mx::array>,
                   std::unordered_map<Operation*, std::vector<mx::array>>>>
        block_ctx;
    TF_ASSIGN_OR_RETURN(auto mlx_res, evalBlock(module, main.front(),
                                                block_arguments, block_ctx));

    SmallVector<DenseElementsAttr> result;
    for (auto& a : mlx_res) {
      a = mx::contiguous(a);
      a.eval();
      SmallVector<int64_t> shape;
      for (auto dim : a.shape()) {
        shape.push_back(static_cast<int64_t>(dim));
      }
      ArrayRef<char> data(a.data<char>(), a.nbytes());
      DenseElementsAttr out = DenseElementsAttr::getFromRawBuffer(
          getResultType(module, a.dtype(), shape), data);
      result.push_back(out);
    }
    return std::make_tuple(result, mlx_res);
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
    TF_ASSIGN_OR_RETURN(auto module_result, evalModule(module, inputs));

    auto [result, mlx_res] = module_result;

    // LOG(INFO) << "Results: " << ToString(result.value()) << "\n";

    // Naive memory space selection, only using CPU global memory.
    PjRtMemorySpace* memory_space =
        device->default_memory_space().value_or(nullptr);
    std::vector<std::unique_ptr<PjRtBuffer>> buffer_results;
    for (auto i = 0; i < result.size(); i++) {
      buffer_results.push_back(
          CreateMlirBufferFromAttribute(mlx_res[i], result[i], memory_space));
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