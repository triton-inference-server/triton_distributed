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

#include "nvidia/nvllm/nvllm_trt.h"

#include "api/engine.hpp"

#include <cstring>

extern "C" {

// int trtllm_mpi_session_set_communicator(void* world_comm_ptr)
// {
//     return nvidia::nvllm::trt::MpiSession::set_communicator(world_comm_ptr);
// }

nvllm_trt_engine_t nvllm_trt_engine_create(const char* config_proto)
{
    // based on the type of engine, we might choose to create a different concrete engine object
    try
    {
        return reinterpret_cast<nvllm_trt_engine_t>(new nvidia::nvllm::trt::StreamingEngine(std::string(config_proto)));
    } catch (const std::exception& e)
    {
        printf("Caught exception when initializing tensorrt_llm: %s\n", e.what());
        return nullptr;
    }
}

nvllm_trt_engine_t nvllm_trt_engine_unsafe_create_from_executor(void* engine)
{
    try
    {
        return reinterpret_cast<nvllm_trt_engine_t>(new nvidia::nvllm::trt::StreamingEngine(engine));
    } catch (const std::exception& e)
    {
        printf("Caught exception when initializing from raw pointer: %s\n", e.what());
        return nullptr;
    }
}

request_id_t nvllm_trt_engine_enqueue_request(nvllm_trt_engine_t engine, client_id_t client_id, const char* req_proto)
{
    // Call the enqueue_request method on the C++ class
    try
    {
        return reinterpret_cast<nvidia::nvllm::trt::StreamingEngine*>(engine)->enqueue_request(client_id,
                                                                                               std::string(req_proto));
    } catch (...)
    {
        return 0;
    }
}

char* nvllm_trt_engine_await_responses(nvllm_trt_engine_t engine)
{
    auto responses    = reinterpret_cast<nvidia::nvllm::trt::StreamingEngine*>(engine)->await_responses();
    char* c_responses = strdup(responses.c_str());  // Allocate memory and copy the string
    return c_responses;                             // Return the C string (remember to free this in the calling code)
}

char* nvllm_trt_engine_await_kv_events(nvllm_trt_engine_t engine)
{
    auto responses = reinterpret_cast<nvidia::nvllm::trt::StreamingEngine*>(engine)->await_kv_events();
    if (!responses)
    {
        return nullptr;
    }
    char* c_responses = strdup(responses->c_str());  // Allocate memory and copy the string
    return c_responses;                              // Return the C string (remember to free this in the calling code)
}

// Get basic iteration stats
char* nvllm_trt_engine_await_iter_stats(nvllm_trt_engine_t engine)
{
    auto responses = reinterpret_cast<nvidia::nvllm::trt::StreamingEngine*>(engine)->await_iter_stats();
    if (!responses)
    {
        return nullptr;
    }
    char* c_responses = strdup(responses->c_str());
    return c_responses;
}

void nvllm_trt_engine_free_responses(char* responses)
{
    free(responses);
}

void nvllm_trt_engine_cancel_request(nvllm_trt_engine_t engine, uint64_t request_id)
{
    reinterpret_cast<nvidia::nvllm::trt::StreamingEngine*>(engine)->cancel_request(request_id);
}

void nvllm_trt_engine_shutdown(nvllm_trt_engine_t engine)
{
    reinterpret_cast<nvidia::nvllm::trt::StreamingEngine*>(engine)->shutdown();
}

int nvllm_trt_engine_destroy(nvllm_trt_engine_t engine)
{
    auto* trtllm_engine = reinterpret_cast<nvidia::nvllm::trt::StreamingEngine*>(engine);
    delete trtllm_engine;
    return NVLLM_TRT_ENGINE_SUCCESS;
}

int nvllm_trt_engine_is_ready(nvllm_trt_engine_t engine)
{
    return reinterpret_cast<nvidia::nvllm::trt::StreamingEngine*>(engine)->is_ready();
}

int nvllm_trt_engine_has_completed(nvllm_trt_engine_t engine)
{
    return reinterpret_cast<nvidia::nvllm::trt::StreamingEngine*>(engine)->has_completed();
}

// int trtllm_version_major()
// {
//     return TRTLLM_VERSION_MAJOR;
// }

// int trtllm_version_minor()
// {
//     return TRTLLM_VERSION_MINOR;
// }

// int trtllm_version_patch()
// {
//     return TRTLLM_VERSION_PATCH;
// }
}
