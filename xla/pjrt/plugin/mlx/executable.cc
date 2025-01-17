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

std::vector<mx::array> getArgArrays(
    Operation* operation,
    const std::unordered_map<Operation*, mx::array>& transient_buffers,
    const std::vector<mx::array>& block_arguments) {
  std::vector<mx::array> args;
  for (Value operand : operation->getOperands()) {
    if (auto defining_op = operand.getDefiningOp()) {
      // arg was created as a result of previously executed operation
      // push array to args if found, else return early
      if (auto search = transient_buffers.find(defining_op);
          search != transient_buffers.end())
        args.push_back(search->second);
      else
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

    // TODO(@cryptodeal): make the module walk logic recursive to handle
    // `func::FuncOp`, which contains nested a `mlir::Operation` tree
    // that must be walked.

    // Value is only set if all operations are executed successfully.
    std::optional<mx::array> res;

    // TODO(@cryptodeal): operations can return multiple results,
    // will need to switch to something along the lines of
    // std::unordered_map<Operation*, std::vector<mx::array>> transient_buffers;

    // holds intermediary results of operation calls.
    std::unordered_map<Operation*, mx::array> transient_buffers;
    // walk the moud
    module.walk([&block_arguments, &transient_buffers, &res](Operation* op) {
      // get operands for each operation call
      auto operands = getArgArrays(op, transient_buffers, block_arguments);

      // if the number of operands does not match the expected number,
      // interrupt walk and return early.
      if (operands.size() != op->getNumOperands())
        return WalkResult::interrupt();

      // TODO(@cryptodeal): explore writing mlx extension that implements
      // various missing ops

      // switch on the operation type
      return llvm::TypeSwitch<Operation*, WalkResult>(op)
          // TODO(@cryptodeal): handle `func` namespace ops
          .Case<func::ReturnOp>([&res, &operands](auto op) {
            // set the result value and interrupt the walk.
            res = operands[0];
            return WalkResult::interrupt();
          })

          // Handle StableHLO nullary ops
          .Case<stablehlo::ConstantOp>([&transient_buffers,
                                        &operands](auto op) {
            transient_buffers.emplace(
                op, getDenseElementsArray(
                        mlir::cast<mlir::DenseElementsAttr>(op.getValue())));
            return WalkResult::advance();
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
            transient_buffers.emplace(
                op,
                mx::broadcast_to(
                    mx::reshape(
                        mx::arange(static_cast<double>(shape[iota_dimension]),
                                   dtypeFromType(result_type)),
                        dimensions),
                    shape));
            return WalkResult::advance();
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
            transient_buffers.emplace(op, mx::abs(operands[0]));
            return WalkResult::advance();
          })
          .Case<stablehlo::CbrtOp>([&transient_buffers, &operands](auto op) {
            transient_buffers.emplace(
                op, mx::power(operands[0],
                              mx::full<float>(operands[0].shape(), 1 / 3,
                                              operands[0].dtype())));
            return WalkResult::advance();
          })
          .Case<stablehlo::CeilOp>([&transient_buffers, &operands](auto op) {
            transient_buffers.emplace(op, mx::ceil(operands[0]));
            return WalkResult::advance();
          })

          .Case<stablehlo::ConvertOp>([&transient_buffers, &operands](auto op) {
            transient_buffers.emplace(
                op, mx::astype(operands[0],
                               dtypeFromType(
                                   op.getResult().getType().getElementType())));
            return WalkResult::advance();
          })
          // .Case<stablehlo::ClzOp>([](auto op) {})
          .Case<stablehlo::CosineOp>([&transient_buffers, &operands](auto op) {
            transient_buffers.emplace(op, mx::cos(operands[0]));
            return WalkResult::advance();
          })
          .Case<stablehlo::ExpOp>([&transient_buffers, &operands](auto op) {
            transient_buffers.emplace(op, mx::exp(operands[0]));
            return WalkResult::advance();
          })
          .Case<stablehlo::Expm1Op>([&transient_buffers, &operands](auto op) {
            transient_buffers.emplace(op, mx::expm1(operands[0]));
            return WalkResult::advance();
          })
          .Case<stablehlo::FloorOp>([&transient_buffers, &operands](auto op) {
            transient_buffers.emplace(op, mx::floor(operands[0]));
            return WalkResult::advance();
          })
          .Case<stablehlo::ImagOp>([&transient_buffers, &operands](auto op) {
            transient_buffers.emplace(op, mx::imag(operands[0]));
            return WalkResult::advance();
          })
          .Case<stablehlo::IsFiniteOp>(
              [&transient_buffers, &operands](auto op) {
                transient_buffers.emplace(op, mx::isfinite(operands[0]));
                return WalkResult::advance();
              })
          .Case<stablehlo::LogOp>([&transient_buffers, &operands](auto op) {
            transient_buffers.emplace(op, mx::log(operands[0]));
            return WalkResult::advance();
          })

          .Case<stablehlo::Log1pOp>([&transient_buffers, &operands](auto op) {
            transient_buffers.emplace(op, mx::log1p(operands[0]));
            return WalkResult::advance();
          })
          .Case<stablehlo::LogisticOp>(
              [&transient_buffers, &operands](auto op) {
                transient_buffers.emplace(op, mx::sigmoid(operands[0]));
                return WalkResult::advance();
              })
          .Case<stablehlo::NotOp>([&transient_buffers, &operands](auto op) {
            transient_buffers.emplace(op, mx::logical_not(operands[0]));
            return WalkResult::advance();
          })
          .Case<stablehlo::NegOp>([&transient_buffers, &operands](auto op) {
            transient_buffers.emplace(op, mx::negative(operands[0]));
            return WalkResult::advance();
          })
          // .Case<stablehlo::PopulationCountOp>([](auto op) {})
          .Case<stablehlo::RealOp>([&transient_buffers, &operands](auto op) {
            transient_buffers.emplace(op, mx::real(operands[0]));
            return WalkResult::advance();
          })
          // `stablehlo::RoundOp` does not match with the mlx metal
          // implementation .Case<stablehlo::RoundOp>([](auto op) {})
          .Case<stablehlo::RoundNearestEvenOp>(
              [&transient_buffers, &operands](auto op) {
                transient_buffers.emplace(op, mx::round(operands[0]));
                return WalkResult::advance();
              })
          .Case<stablehlo::RsqrtOp>([&transient_buffers, &operands](auto op) {
            transient_buffers.emplace(op, mx::rsqrt(operands[0]));
            return WalkResult::advance();
          })
          .Case<stablehlo::SignOp>([&transient_buffers, &operands](auto op) {
            transient_buffers.emplace(op, mx::sign(operands[0]));
            return WalkResult::advance();
          })
          .Case<stablehlo::SineOp>([&transient_buffers, &operands](auto op) {
            transient_buffers.emplace(op, mx::sin(operands[0]));
            return WalkResult::advance();
          })
          .Case<stablehlo::SqrtOp>([&transient_buffers, &operands](auto op) {
            transient_buffers.emplace(op, mx::sqrt(operands[0]));
            return WalkResult::advance();
          })
          .Case<stablehlo::TanOp>([&transient_buffers, &operands](auto op) {
            transient_buffers.emplace(op, mx::tan(operands[0]));
            return WalkResult::advance();
          })
          .Case<stablehlo::TanhOp>([&transient_buffers, &operands](auto op) {
            transient_buffers.emplace(op, mx::tanh(operands[0]));
            return WalkResult::advance();
          })

          // Handle StableHLO binary elementwise ops
          .Case<stablehlo::AddOp>([&transient_buffers, &operands](auto op) {
            transient_buffers.emplace(op, mx::add(operands[0], operands[1]));
            return WalkResult::advance();
          })
          .Case<stablehlo::Atan2Op>([&transient_buffers, &operands](auto op) {
            transient_buffers.emplace(op,
                                      mx::arctan2(operands[0], operands[1]));
            return WalkResult::advance();
          })
          // TODO(@cryptodeal): implement complex op
          // .Case<stablehlo::ComplexOp>([&transient_buffers, &operands](auto
          // op) {})
          .Case<stablehlo::DivOp>([&transient_buffers, &operands](auto op) {
            transient_buffers.emplace(op, mx::divide(operands[0], operands[1]));
            return WalkResult::advance();
          })
          .Case<stablehlo::MaxOp>([&transient_buffers, &operands](auto op) {
            transient_buffers.emplace(op,
                                      mx::maximum(operands[0], operands[1]));
            return WalkResult::advance();
          })
          .Case<stablehlo::MinOp>([&transient_buffers, &operands](auto op) {
            transient_buffers.emplace(op,
                                      mx::minimum(operands[0], operands[1]));
            return WalkResult::advance();
          })
          .Case<stablehlo::MulOp>([&transient_buffers, &operands](auto op) {
            transient_buffers.emplace(op,
                                      mx::multiply(operands[0], operands[1]));
            return WalkResult::advance();
          })
          .Case<stablehlo::PowOp>([&transient_buffers, &operands](auto op) {
            transient_buffers.emplace(op, mx::power(operands[0], operands[1]));
            return WalkResult::advance();
          })
          .Case<stablehlo::RemOp>([&transient_buffers, &operands](auto op) {
            transient_buffers.emplace(op,
                                      mx::remainder(operands[0], operands[1]));
            return WalkResult::advance();
          })
          .Case<stablehlo::ShiftLeftOp>(
              [&transient_buffers, &operands](auto op) {
                transient_buffers.emplace(
                    op, mx::left_shift(operands[0], operands[1]));
                return WalkResult::advance();
              })
          // TODO(@cryptodeal): handle shift right (xla has 2 options)
          // .Case<stablehlo::ShiftRightArithmeticOp>([&transient_buffers,
          // &operands](auto op) {
          //   transient_buffers.emplace(op, mx::right_shift(operands[0],
          //   operands[1])); return WalkResult::advance();
          // })
          // .Case<stablehlo::ShiftRightLogicalOp>([&transient_buffers,
          // &operands](auto op) {
          //   transient_buffers.emplace(op, mx::right_shift(operands[0],
          //   operands[1])); return WalkResult::advance();
          // })
          .Case<stablehlo::SubtractOp>(
              [&transient_buffers, &operands](auto op) {
                transient_buffers.emplace(
                    op, mx::subtract(operands[0], operands[1]));
                return WalkResult::advance();
              })

          // Handle StableHLO binary logical elementwise ops
          .Case<stablehlo::AndOp>([&transient_buffers, &operands](auto op) {
            transient_buffers.emplace(
                op, mx::logical_and(operands[0], operands[1]));
            return WalkResult::advance();
          })
          .Case<stablehlo::OrOp>([&transient_buffers, &operands](auto op) {
            transient_buffers.emplace(op,
                                      mx::logical_or(operands[0], operands[1]));
            return WalkResult::advance();
          })
          .Case<stablehlo::XorOp>([&transient_buffers, &operands](auto op) {
            transient_buffers.emplace(
                op, mx::bitwise_xor(operands[0], operands[1]));
            return WalkResult::advance();
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
          // .Case<stablehlo::ReduceOp>([](auto op) {})

          // Handle StableHLO tuple ops
          // .Case<stablehlo::GetTupleElementOp>([](auto op) {})
          // .Case<stablehlo::TupleOp>([](auto op) {})

          .Case<stablehlo::CompareOp>([&transient_buffers, &operands](auto op) {
            auto comp_dir = op.getComparisonDirection();
            switch (op.getComparisonDirection()) {
              case ComparisonDirection::NE:
                transient_buffers.emplace(
                    op, mx::not_equal(operands[0], operands[1]));
                break;
              case ComparisonDirection::GE:
                transient_buffers.emplace(
                    op, mx::greater_equal(operands[0], operands[1]));
                break;
              case ComparisonDirection::GT:
                transient_buffers.emplace(
                    op, mx::greater(operands[0], operands[1]));
                break;
              case ComparisonDirection::LE:
                transient_buffers.emplace(
                    op, mx::less_equal(operands[0], operands[1]));
                break;
              case ComparisonDirection::LT:
                transient_buffers.emplace(op,
                                          mx::less(operands[0], operands[1]));
                break;
              default:
                transient_buffers.emplace(op,
                                          mx::equal(operands[0], operands[1]));
            }
            return WalkResult::advance();
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

            transient_buffers.emplace(op, mx::slice(operands[0], start_indices,
                                                    limit_indices, strides));
            return WalkResult::advance();
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
                transient_buffers.emplace(
                    op, mx::broadcast_to(operands[0],
                                         getMlxShape(mlir::cast<ShapedType>(
                                             op.getResult().getType()))));
                return WalkResult::advance();
              })
          // .Case<stablehlo::DynamicBroadcastInDimOp>([](auto op) {})
          .Case<stablehlo::CholeskyOp>(
              [&transient_buffers, &operands](auto op) {
                transient_buffers.emplace(
                    op, mx::linalg::cholesky(operands[0], op.getLower() != 0));
                return WalkResult::advance();
              })
          .Case<stablehlo::ClampOp>([&transient_buffers, &operands](auto op) {
            transient_buffers.emplace(
                op, mx::clip(operands[1], operands[0], operands[2]));
            return WalkResult::advance();
          })
          .Case<stablehlo::ConcatenateOp>(
              [&transient_buffers, &operands](auto op) {
                transient_buffers.emplace(
                    op, mx::concatenate(operands, op.getDimension()));
                return WalkResult::advance();
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
            auto dot_dimension_numbers = op.getDotDimensionNumbers();
            std::unordered_set<int> lhs_used_dims;
            std::vector<int> lhs_batching_dims;
            for (auto dim : dot_dimension_numbers.getLhsBatchingDimensions()) {
              lhs_batching_dims.push_back(static_cast<int>(dim));
              lhs_used_dims.insert(static_cast<int>(dim));
            }
            std::vector<int> lhs_contracting_dims;
            for (auto dim :
                 dot_dimension_numbers.getLhsContractingDimensions()) {
              lhs_contracting_dims.push_back(static_cast<int>(dim));
              lhs_used_dims.insert(static_cast<int>(dim));
            }
            std::vector<int> lhs_concat_dims;
            for (int i = 0; i < operands[0].ndim(); i++) {
              if (lhs_used_dims.find(i) == lhs_used_dims.end()) {
                lhs_concat_dims.push_back(i);
              }
            }
            std::unordered_set<int> rhs_used_dims;
            std::vector<int> rhs_batching_dims;
            for (auto dim : dot_dimension_numbers.getRhsBatchingDimensions()) {
              rhs_batching_dims.push_back(static_cast<int>(dim));
              rhs_used_dims.insert(static_cast<int>(dim));
            }
            std::vector<int> rhs_contracting_dims;
            for (auto dim :
                 dot_dimension_numbers.getRhsContractingDimensions()) {
              rhs_contracting_dims.push_back(static_cast<int>(dim));
              rhs_used_dims.insert(static_cast<int>(dim));
            }
            std::vector<int> rhs_concat_dims;
            for (int i = 0; i < operands[1].ndim(); i++) {
              if (rhs_used_dims.find(i) == rhs_used_dims.end()) {
                rhs_concat_dims.push_back(i);
              }
            }
            transient_buffers.emplace(
                op,
                mx::batched_tensordot(operands[0], operands[1],
                                      lhs_contracting_dims, lhs_batching_dims,
                                      lhs_concat_dims, rhs_contracting_dims,
                                      rhs_batching_dims, rhs_concat_dims));
            return WalkResult::advance();
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
                transient_buffers.emplace(
                    op, mx::array(operands[0].shape(op.getDimension())));
                return WalkResult::advance();
              })
          // .Case<stablehlo::MapOp>([](auto op) {})
          .Case<stablehlo::ReshapeOp>([&transient_buffers, &operands](auto op) {
            transient_buffers.emplace(
                op, mx::reshape(operands[0], getMlxShape(mlir::cast<ShapedType>(
                                                 op.getResult().getType()))));
            return WalkResult::advance();
          })
          // .Case<stablehlo::DynamicReshapeOp>([](auto op) {})
          // .Case<stablehlo::ScatterOp>([](auto op) {})
          .Case<stablehlo::SelectOp>([&transient_buffers, &operands](auto op) {
            transient_buffers.emplace(
                op, mx::where(operands[0], operands[1], operands[2]));
            return WalkResult::advance();
          })
          // .Case<stablehlo::SelectAndScatterOp>([](auto op) {})
          // .Case<stablehlo::SetDimensionSizeOp>([](auto op) {})
          // .Case<stablehlo::SortOp>([](auto op) {})
          // .Case<stablehlo::ReverseOp>([](auto op) {})
          // .Case<stablehlo::PadOp>([](auto op) {})
          .Case<stablehlo::TransposeOp>(
              [&transient_buffers, &operands](auto op) {
                std::vector<int> axes;
                for (auto axis : op.getPermutation()) {
                  axes.push_back(static_cast<int>(axis));
                }
                transient_buffers.emplace(op, mx::transpose(operands[0], axes));
                return WalkResult::advance();
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
            transient_buffers.emplace(
                op,
                op.getRngDistribution() == RngDistribution::UNIFORM
                    ? mx::random::uniform(operands[0], operands[1], shape,
                                          operands[0].dtype())
                    : mx::random::normal(
                          shape, operands[0].dtype(),
                          mx::astype(operands[0], mx::float32).item<float>(),
                          mx::astype(operands[1], mx::float32).item<float>()));
            return WalkResult::advance();
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
            return WalkResult::interrupt();
          });
    });

    // TODO(@cryptodeal): ensure the resulting data is allocated/freed
    // correctly.
    if (res.has_value()) {
      mx::array out = res.value();
      mx::eval(out);
      SmallVector<int64_t> shape;
      for (auto dim : out.shape()) {
        shape.push_back(static_cast<int64_t>(dim));
      }
      ArrayRef<char> data(out.data<char>(), out.nbytes());
      DenseElementsAttr result = DenseElementsAttr::getFromRawBuffer(
          getResultType(module, out.dtype(), shape), data);
      return SmallVector<DenseElementsAttr>{result};
    }
    return failure();
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