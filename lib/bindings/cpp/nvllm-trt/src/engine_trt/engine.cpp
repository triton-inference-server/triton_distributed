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
#include "engine_trt/config.hpp"
#include "engine_trt/kv_event.hpp"
#include "engine_trt/request.hpp"
#include "engine_trt/response.hpp"
#include "engine_trt/stats.hpp"

// TensorRT LLM Executor
#include "NvInfer.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"

// Third-party
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace ex = tensorrt_llm::executor;

namespace nvidia::nvllm::trt {

/// Customize the logger for TensorRT LLM using a module-specific spdlog logger
class TRTLogger : public nvinfer1::ILogger
{
  public:
    TRTLogger(std::shared_ptr<spdlog::logger> logger) : m_logger(logger) {}

    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
    {
        if (severity <= nvinfer1::ILogger::Severity::kERROR)
        {
            m_logger->error("{}", msg);
        }
        else if (severity == nvinfer1::ILogger::Severity::kWARNING)
        {
            m_logger->warn("{}", msg);
        }
        else
        {
            m_logger->info("{}", msg);
        }
    }

  private:
    std::shared_ptr<spdlog::logger> m_logger;
};

class StreamingEngine::Impl
{
  public:
    Impl(const std::string& config_proto);
    Impl(void* engine);
    ~Impl() = default;

    /// Enqueues a request to the executor
    /// In this opionionated implementation, [`client_id`] is required to be unique
    uint64_t enqueue_request(uint64_t client_id, const std::string& req_json)
    {
        spdlog::trace("enqueue_request - client_id: {}", client_id);
        auto request = deserialize_request(req_json);
        request.setClientId(client_id);
        auto request_id = m_executor->enqueueRequest(request);
        spdlog::trace("request_id: {} with client_id {} was enqueued", request_id, client_id);
        return request_id;
    }

    /// Cancellation is by [`request_id`], not [`client_id`]
    void cancel_request(uint64_t request_id)
    {
        spdlog::trace("cancel_request: {}", request_id);
        m_executor->cancelRequest(request_id);
    }

    /// Issues a shutdown request to the executor. This is a blocking call.
    /// We protect it with a mutex to ensure that it is only called once.
    void shutdown()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_has_completed)
        {
            return;
        }
        m_executor->shutdown();
        m_has_completed = true;
    }

    /// Returns true if the executor is ready to accept requests.
    /// Not sure of TensorRT LLM's behavior when the executor is shutdown, so we
    /// return false if the executor has completed.
    bool is_ready() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_has_completed)
        {
            return false;
        }
        return m_executor->canEnqueueRequests();
    }

    /// Returns true if the executor has completed.
    bool has_completed() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_has_completed;
    }

    /// Awaits on the executor for responses. This is a blocking call.
    /// TensorRT LLM will throw an exception if a thread is blocked on the calls and the
    /// executor is shutdown.
    std::string await_responses()
    {
        spdlog::trace("blocking on await_responses");
        std::deque<ex::Response> responses;
        bool shutdown = false;

        try
        {
            auto v_responses = m_executor->awaitResponses();
            spdlog::trace("received {} responses", v_responses.size());

            for (auto& response : v_responses)
            {
                responses.push_back(std::move(response));
            }
        } catch (const std::exception& e)
        {
            spdlog::trace("Exception caught awaiting responses; shutting down");
            shutdown = true;
        }
        return serialize_responses(std::move(responses), shutdown);
    }

    /// Awaits for KV events. This is a blocking call with a timeout of 250ms.
    /// The current implementation will not throw an exception if the executor is shutdown,
    /// so we need timeout the call to ensure that calling thread can shutdown properly.
    std::optional<std::string> await_kv_events()
    {
        if (m_kv_cache_event_manager == nullptr)
        {
            auto manager = m_executor->getKVCacheEventManager();
            if (manager)
            {
                m_kv_cache_event_manager = *manager;
            }
        }

        if (m_kv_cache_event_manager == nullptr)
        {
            return std::nullopt;
        }

        try
        {
            auto events = m_kv_cache_event_manager->getLatestEvents({std::chrono::milliseconds(250)});
            if (!events.empty())
            {
                spdlog::trace("received {} on kv_events", events.size());
            }
            return {serialize_kv_events(std::move(events), false)};
        } catch (const std::exception& e)
        {
            spdlog::trace("Exception caught awaiting kv events; shutting down");
            return {serialize_kv_events({}, true)};
        }
    }

    // Awaits iteration stats
    std::optional<std::string> await_iter_stats()
    {
        auto iter_stats = m_executor->getLatestIterationStats();

        return serialize_iter_stats(iter_stats);
    }

  private:
    std::unique_ptr<ex::Executor> m_executor;
    std::shared_ptr<ex::KVCacheEventManager> m_kv_cache_event_manager = nullptr;
    bool m_has_completed                                              = false;
    mutable std::mutex m_mutex;
};

// Private Engine Impl

StreamingEngine::Impl::Impl(void* engine)
{
    auto nvllm_logger = spdlog::stdout_color_mt("nvllm");
    spdlog::set_default_logger(nvllm_logger);

    spdlog::info("Instantiating nvLLM from raw TensorRT LLM Executor pointer");
    m_executor.reset(reinterpret_cast<ex::Executor*>(engine));
}

StreamingEngine::Impl::Impl(const std::string& config_json)
{
    auto nvllm_logger  = spdlog::stdout_color_mt("nvllm");
    auto trtllm_logger = spdlog::stdout_color_mt("trtllm");
    spdlog::set_default_logger(nvllm_logger);

    auto config = deserialize_config(config_json);

    if (config.log_level == "error")
    {
        spdlog::set_level(spdlog::level::err);
        nvllm_logger->set_level(spdlog::level::err);
        trtllm_logger->set_level(spdlog::level::err);
    }
    else if (config.log_level == "warn")
    {
        spdlog::set_level(spdlog::level::warn);
        nvllm_logger->set_level(spdlog::level::warn);
        trtllm_logger->set_level(spdlog::level::warn);
    }
    else if (config.log_level == "info")
    {
        spdlog::set_level(spdlog::level::info);
        nvllm_logger->set_level(spdlog::level::info);
        trtllm_logger->set_level(spdlog::level::info);
    }
    else if (config.log_level == "debug")
    {
        spdlog::set_level(spdlog::level::debug);
        nvllm_logger->set_level(spdlog::level::debug);
        trtllm_logger->set_level(spdlog::level::debug);
    }
    else if (config.log_level == "trace")
    {
        spdlog::set_level(spdlog::level::trace);
        nvllm_logger->set_level(spdlog::level::trace);
        trtllm_logger->set_level(spdlog::level::trace);
    }
    else
    {
        spdlog::set_level(spdlog::level::err);
        nvllm_logger->set_level(spdlog::level::err);
        trtllm_logger->set_level(spdlog::level::err);
    }

    TRTLogger* trtLogger = new TRTLogger(trtllm_logger);
    initTrtLlmPlugins(trtLogger);

    auto kv_config = config.config.getKvCacheConfig();

    spdlog::info("Enabled block reuse: true");
    kv_config.setEnableBlockReuse(true);
    kv_config.setEventBufferMaxSize(65536);

    config.config.setKvCacheConfig(kv_config);

    m_executor = std::make_unique<ex::Executor>(config.model_path, ex::ModelType::kDECODER_ONLY, config.config);
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
