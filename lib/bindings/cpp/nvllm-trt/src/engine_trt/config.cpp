// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "engine_trt/config.hpp"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

using json   = nlohmann::json;
namespace ex = tensorrt_llm::executor;

namespace nvidia::nvllm::trt {

struct ExecutorConfig
{
    std::string model_path;
    std::string log_level;
    std::optional<bool> enable_chunked_context;
    std::optional<bool> normalize_log_probs;
    std::optional<uint32_t> iter_stats_max_iterations;
};

// Custom to_json function
inline void to_json(json& j, const ExecutorConfig& e)
{
    j = json{{"model_path", e.model_path}, {"log_level", e.log_level}};
    if (e.enable_chunked_context)
    {
        j["enable_chunked_context"] = e.enable_chunked_context.value();
    }
    if (e.normalize_log_probs)
    {
        j["normalize_log_probs"] = e.normalize_log_probs.value();
    }
    if (e.iter_stats_max_iterations)
    {
        j["iter_stats_max_iterations"] = e.iter_stats_max_iterations.value();
    }
}

// Custom from_json function
inline void from_json(const json& j, ExecutorConfig& e)
{
    j.at("model_path").get_to(e.model_path);
    j.at("log_level").get_to(e.log_level);

    if (j.contains("enable_chunked_context"))
    {
        e.enable_chunked_context = j.at("enable_chunked_context").get<bool>();
    }
    else
    {
        e.enable_chunked_context = std::nullopt;
    }

    if (j.contains("normalize_log_probs"))
    {
        e.normalize_log_probs = j.at("normalize_log_probs").get<bool>();
    }
    else
    {
        e.normalize_log_probs = std::nullopt;
    }

    if (j.contains("iter_stats_max_iterations"))
    {
        e.iter_stats_max_iterations = j.at("iter_stats_max_iterations").get<uint32_t>();
    }
    else
    {
        e.iter_stats_max_iterations = std::nullopt;
    }
}

Config deserialize_config(const std::string& config_json)
{
    auto config_in  = json::parse(config_json).get<ExecutorConfig>();
    auto model_path = config_in.model_path;
    auto log_level  = config_in.log_level;
    auto config     = ex::ExecutorConfig();

    // todo - expose max num tokens
    // todo - expose from engine block reuse

    if (config_in.enable_chunked_context)
    {
        spdlog::info("Enable chunked context: {}", config_in.enable_chunked_context.value() ? "true" : "false");
        config.setEnableChunkedContext(config_in.enable_chunked_context.value());
    }

    return {model_path, log_level, config};
}

}  // namespace nvidia::nvllm::trt
