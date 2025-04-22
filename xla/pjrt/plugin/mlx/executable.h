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

#ifndef XLA_PJRT_PLUGIN_STABLEHLO_MLX_EXECUTABLE_H_
#define XLA_PJRT_PLUGIN_STABLEHLO_MLX_EXECUTABLE_H_
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "mlx/mlx.h"
#include "xla/pjrt/plugin/mlx/utils.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/service/computation_placer.h"

namespace mlir::stablehlo {

struct Node {
  std::string name;
  std::vector<std::unique_ptr<Node>> children;

  Node(std::string name) : name(name) {}
};

class Tree {
 public:
  std::unique_ptr<Node> root;

  Tree() : root(nullptr) {}

  Node* insert(std::string name, Node* parent = nullptr) {
    auto new_node = std::make_unique<Node>(name);
    if (!root) {
      root = std::move(new_node);
      return root.get();
    } else if (parent) {
      parent->children.push_back(std::move(new_node));
      return parent->children.back().get();
    }
  }

  void remove(const std::string& name) {
    if (!root) return;

    // Special case: root node matches
    if (root->name == name) {
      root.reset();  // deletes entire tree
      return;
    }

    remove_helper(root, name);
  }

  std::set<std::string> getAllFnNames() {
    std::set<std::string> fn_names;
    if (!root) return fn_names;

    std::function<void(Node*)> traverse = [&](Node* node) {
      fn_names.insert(node->name);
      for (auto& child : node->children) {
        traverse(child.get());
      }
    };
    traverse(root.get());
    return fn_names;
  }

  std::unordered_set<std::string> getCompilableFns() {
    std::unordered_set<std::string> compilable_fns;
    if (!root) return compilable_fns;

    std::function<void(Node*)> traverse = [&](Node* node) {
      if (node->children.empty()) {
        compilable_fns.insert(node->name);
      } else {
        for (auto& child : node->children) {
          traverse(child.get());
        }
      }
    };
    traverse(root.get());
    return compilable_fns;
  }

 private:
  void remove_helper(std::unique_ptr<Node>& current, const std::string& name) {
    if (!current) return;

    auto& children = current->children;

    // Remove matching children
    children.erase(std::remove_if(children.begin(), children.end(),
                                  [&](std::unique_ptr<Node>& child) {
                                    return child->name == name;
                                  }),
                   children.end());

    // Recurse on remaining children
    for (auto& child : children) {
      remove_helper(child, name);
    }
  }
};

template <typename... T>
class CompiledOpInfo {
 public:
  std::tuple<T...> captures;
  std::vector<std::pair<mx::Dtype, mx::Shape>> input_metadata;
  std::vector<std::pair<mx::Dtype, mx::Shape>> output_metadata;

  CompiledOpInfo(std::vector<mx::array> inputs,
                 std::vector<std::pair<mx::Dtype, mx::Shape>> outputs,
                 T... capture_values);

  CompiledOpInfo(std::vector<std::pair<mx::Dtype, mx::Shape>> inputs,
                 std::vector<std::pair<mx::Dtype, mx::Shape>> outputs,
                 T... capture_values);

  bool operator==(const CompiledOpInfo& other) const {
    return captures == other.captures &&
           input_metadata == other.input_metadata &&
           output_metadata == other.output_metadata;
  }
};

template <typename... T>
CompiledOpInfo<T...>::CompiledOpInfo(
    std::vector<mx::array> inputs,
    std::vector<std::pair<mx::Dtype, mx::Shape>> outputs, T... capture_values)
    : captures(std::make_tuple(capture_values...)), output_metadata(outputs) {
  input_metadata.reserve(inputs.size());
  for (const auto& input : inputs) {
    input_metadata.push_back(std::make_pair(input.dtype(), input.shape()));
  }
}

template <typename... T>
CompiledOpInfo<T...>::CompiledOpInfo(
    std::vector<std::pair<mx::Dtype, mx::Shape>> inputs,
    std::vector<std::pair<mx::Dtype, mx::Shape>> outputs, T... capture_values)
    : captures(std::make_tuple(capture_values...)),
      input_metadata(inputs),
      output_metadata(outputs) {}

using CompiledOpSign =
    std::function<std::vector<mx::array>(const std::vector<mx::array>&)>;

using CallOpInfo = CompiledOpInfo<std::string>;

// stablehlo::IotaOp - Captures: (dimensions, dtype, iota_dim, result_shape)
using IotaOpInfo = CompiledOpInfo<std::vector<int32_t>, mx::Dtype, int64_t,
                                  std::vector<int32_t>>;

// stablehlo::CbrtOp - Captures: ()
using CbrtOpInfo = CompiledOpInfo<>;

// stablehlo::BroadcastInDimOp - Captures: (broadcast_dims, target_shape)
using BroadcastInDimOpInfo =
    CompiledOpInfo<std::vector<int32_t>, std::vector<int32_t>>;

// stablehlo::DotGeneralOp - Captures: (lhs_batch_dims, rhs_batch_dims,
// lhs_contract_dims, rhs_contract_dims)
using DotGeneralOpInfo =
    CompiledOpInfo<std::vector<int32_t>, std::vector<int32_t>,
                   std::vector<int32_t>, std::vector<int32_t>>;

// stablehlo::GatherOp - Captures: (collapsed_slice_dims, index_vector_dim,
// offset_dims, operand_batching_dims, result_shape, slice_sizes,
// start_index_map, start_indices_batching_dims)
using GatherOpInfo = CompiledOpInfo<std::vector<int32_t>, int64_t,
                                    std::vector<int32_t>, std::vector<int32_t>,
                                    std::vector<int32_t>, std::vector<int32_t>,
                                    std::vector<int32_t>, std::vector<int32_t>>;

// stablehlo::ScatterOp - Captures: (index_vector_dim, input_batching_dims,
// inserted_window_dims, scatter_dims_to_operand_dims,
// scatter_indices_batching_dims, scatter_type, update_window_dims)
using ScatterOpInfo =
    CompiledOpInfo<int64_t, std::vector<int32_t>, std::vector<int32_t>,
                   std::vector<int32_t>, std::vector<int32_t>,
                   utils::ScatterType, std::vector<int32_t>>;

// stablehlo::PadOp - Captures: (result_shape, edge_pad_low, interior_pad)
using PadOpInfo = CompiledOpInfo<std::vector<int32_t>, std::vector<int32_t>,
                                 std::vector<int32_t>>;

absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> StablehloMlxCompile(
    mlir::ModuleOp module, xla::DeviceAssignment assignment,
    xla::PjRtClient* client);

absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> StablehloMlxCompile(
    xla::XlaComputation const& computation, xla::DeviceAssignment assignment,
    xla::PjRtClient* client);

}  // namespace mlir::stablehlo

template <typename... T>
struct std::hash<mlir::stablehlo::CompiledOpInfo<T...>> {
  std::size_t operator()(
      const mlir::stablehlo::CompiledOpInfo<T...>& info) const noexcept {
    std::size_t seed = 0;
    hash_tuple(seed, info.captures);
    for (const auto& meta : info.input_metadata) {
      hash_combine(seed, std::hash<mx::Dtype>{}(meta.first));
      hash_combine(seed, std::hash<std::vector<int32_t>>{}(meta.second));
    }
    for (const auto& meta : info.output_metadata) {
      hash_combine(seed, std::hash<mx::Dtype>{}(meta.first));
      hash_combine(seed, std::hash<std::vector<int32_t>>{}(meta.second));
    }
    return seed;
  }
};

namespace mlir::stablehlo {
struct OpLookup {
  std::unordered_map<CompiledOpInfo<std::string>, CompiledOpSign> call;
  std::unordered_map<IotaOpInfo, CompiledOpSign> iota;
  std::unordered_map<CbrtOpInfo, CompiledOpSign> cbrt;
  std::unordered_map<BroadcastInDimOpInfo, CompiledOpSign> broadcast_in_dim;
  std::unordered_map<DotGeneralOpInfo, CompiledOpSign> dot_general;
  std::unordered_map<GatherOpInfo, CompiledOpSign> gather;
  std::unordered_map<ScatterOpInfo, CompiledOpSign> scatter;
  std::unordered_map<PadOpInfo, CompiledOpSign> pad;
};

};  // namespace mlir::stablehlo

#endif  // XLA_PJRT_PLUGIN_STABLEHLO_MLX_EXECUTABLE_H_