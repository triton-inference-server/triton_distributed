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

// Public API for the StreamingEngine class
#include "nvidia/nvllm/nvllm_trt.h"

// Internal Private Implementation
#include "api/engine.hpp"

#include <optional>

extern "C" {
bool initTrtLlmPlugins(void* logger, char const* libNamespace);
}

namespace nvidia::nvllm::trt {

class StreamingEngine::Impl
{
  public:
    Impl(const std::string& config_proto);
    Impl(void* engine);
    ~Impl() = default;

    uint64_t enqueue_request(uint64_t client_id, const std::string& req_proto)
    {
        std::abort();
        return 911;
    }

    void cancel_request(uint64_t request_id) {}

    std::string await_responses()
    {
        std::abort();
        return {};
    }

    std::optional<std::string> await_kv_events()
    {
        std::abort();
        return std::nullopt;
    }

    std::optional<std::string> await_iter_stats()
    {
        std::abort();
        return std::nullopt;
    }

    void shutdown()
    {
        std::abort();
    }

    bool is_ready() const
    {
        std::abort();
        return false;
    }

    bool has_completed() const
    {
        std::abort();
        return false;
    }
};

// Private Engine Impl

StreamingEngine::Impl::Impl(const std::string& config_proto)
{
    initTrtLlmPlugins(nullptr, nullptr);
}

StreamingEngine::Impl::Impl(void* engine)
{
    initTrtLlmPlugins(nullptr, nullptr);
}

// Public Engine Impl

StreamingEngine::StreamingEngine(const std::string& config_proto) :
  m_impl{std::make_unique<Impl>(config_proto)} {}  // namespace nvidia::nvllm::trt

StreamingEngine::StreamingEngine(void* engine) :
  m_impl{std::make_unique<Impl>(engine)} {}  // namespace nvidia::nvllm::trt

StreamingEngine::~StreamingEngine()
{
    if (!m_impl->has_completed())
    {
        m_impl->shutdown();
    }
}

uint64_t StreamingEngine::enqueue_request(uint64_t client_id, const std::string& req_proto)
{
    return m_impl->enqueue_request(client_id, req_proto);
}

std::string StreamingEngine::await_responses()
{
    return m_impl->await_responses();
}

std::optional<std::string> StreamingEngine::await_kv_events()
{
    return m_impl->await_kv_events();
}

std::optional<std::string> StreamingEngine::await_iter_stats()
{
    return m_impl->await_iter_stats();
}

void StreamingEngine::cancel_request(uint64_t request_id)
{
    m_impl->cancel_request(request_id);
}

void StreamingEngine::shutdown()
{
    m_impl->shutdown();
}

bool StreamingEngine::is_ready() const
{
    return m_impl->is_ready();
}

bool StreamingEngine::has_completed() const
{
    return m_impl->has_completed();
}

}  // namespace nvidia::nvllm::trt
