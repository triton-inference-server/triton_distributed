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

#include "engine_trt/stats.hpp"

#include <nlohmann/json.hpp>

#include <deque>

using json = nlohmann::json;

namespace nvidia::nvllm::trt {

std::string serialize_iter_stats(std::deque<::tensorrt_llm::executor::IterationStats> stats)
{
    json json_stats = json::array();
    for (const auto& stat : stats)
    {
        if (stat.kvCacheStats.has_value())
        {
            json entry;
            entry["iter"]                     = stat.iter;
            entry["kv_active_blocks"]         = stat.kvCacheStats->usedNumBlocks;
            entry["kv_total_blocks"]          = stat.kvCacheStats->maxNumBlocks;
            entry["request_active_slots"]     = stat.numActiveRequests;
            entry["request_total_slots"]      = stat.maxNumActiveRequests;
            entry["request_new_active_slots"] = stat.numNewActiveRequests;
            json_stats.push_back(entry);
        }
        else
        {
            json entry;
            entry["iter"]                     = stat.iter;
            entry["request_active_slots"]     = stat.numActiveRequests;
            entry["request_total_slots"]      = stat.maxNumActiveRequests;
            entry["request_new_active_slots"] = stat.numNewActiveRequests;
            json_stats.push_back(entry);
        }
    }

    return json_stats.dump();
}

}  // namespace nvidia::nvllm::trt
