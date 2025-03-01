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

#pragma once

#include <memory>
#include <optional>
#include <string>
namespace nvidia::nvllm::trt {

class StreamingEngine
{
  public:
    StreamingEngine(const std::string& config_proto);
    StreamingEngine(void* engine);
    ~StreamingEngine();

    // accepts a string of a serialized proto::Request
    // forms the internal request object and enqueues it
    // returns a request_id provided by the engine; this must be used to cancel the request
    // accepts a client_id which can be use to identify the response
    uint64_t enqueue_request(uint64_t client_id, const std::string& json_request);

    // awaits the presence of a response
    // converts the internal format to a json and returns the string
    std::string await_responses();

    // awaits the presence of a kv events
    std::optional<std::string> await_kv_events();

    // Awaits iteration stats
    std::optional<std::string> await_iter_stats();

    // cancel request
    void cancel_request(uint64_t request_id);

    // called to start the shutdown sequence
    void shutdown();

    // returns true once the engine as started pulling requests
    // there is currently no stopping, so once an engine has_started,
    // it will always return true, even when complete
    bool is_ready() const;

    // returns true if the StreamingEngine has been both shutdown and joined
    bool has_completed() const;

  private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace nvidia::nvllm::trt
