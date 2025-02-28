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

#include "engine_trt/response.hpp"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <optional>
#include <string>
#include <vector>

using json   = nlohmann::json;
namespace ex = tensorrt_llm::executor;

namespace nvidia::nvllm::trt {

// Forward declarations
struct Response;
struct Output;

enum FinishReasonEnum
{
    FINISH_REASON_NOT_DONE = 0,
    FINISH_REASON_EOS      = 1,
    FINISH_REASON_STOP     = 2,
    FINISH_REASON_LENGTH   = 3,
};

// Output Struct
struct Output
{
    bool is_final;
    std::vector<int32_t> token_ids;
    std::optional<float> cum_log_prob;
    std::optional<std::vector<float>> log_probs;
    std::optional<FinishReasonEnum> finish_reason;
};

// Custom to_json function
void to_json(json& j, const Output& o)
{
    j = json{{"is_final", o.is_final}, {"token_ids", o.token_ids}};
    if (o.cum_log_prob)
    {
        j["cum_log_prob"] = *o.cum_log_prob;
    }
    if (o.log_probs)
    {
        j["log_probs"] = *o.log_probs;
    }
    if (o.finish_reason)
    {
        j["finish_reason"] = static_cast<int>(*o.finish_reason);
    }
}

void from_json(const json& j, Output& o)
{
    j.at("is_final").get_to(o.is_final);
    j.at("token_ids").get_to(o.token_ids);

    if (j.contains("cum_log_prob") && !j["cum_log_prob"].is_null())
    {
        o.cum_log_prob = j["cum_log_prob"].get<float>();
    }
    else
    {
        o.cum_log_prob = std::nullopt;
    }

    if (j.contains("log_probs") && !j["log_probs"].is_null())
    {
        o.log_probs = j["log_probs"].get<std::vector<float>>();
    }
    else
    {
        o.log_probs = std::nullopt;
    }

    if (j.contains("finish_reason") && !j["finish_reason"].is_null())
    {
        o.finish_reason = static_cast<FinishReasonEnum>(j["finish_reason"].get<int>());
    }
    else
    {
        o.finish_reason = std::nullopt;
    }
}

// Response Struct
struct Response
{
    uint64_t request_id;
    std::optional<uint64_t> client_id;  // Optional client ID.
    std::optional<std::string> error_msg;
    std::optional<Output> output;
};

inline void to_json(json& j, const Response& p)
{
    j = json{{"request_id", p.request_id}};
    if (p.client_id)
        j["client_id"] = p.client_id.value();
    if (p.error_msg)
        j["error_msg"] = p.error_msg.value();
    if (p.output)
        j["output"] = p.output.value();
}

inline void from_json(const json& j, Response& p)
{
    j.at("request_id").get_to(p.request_id);
    if (j.contains("client_id"))
        p.client_id = j.at("client_id").get<uint64_t>();
    if (j.contains("error_msg"))
        p.error_msg = j.at("error_msg").get<std::string>();
    if (j.contains("output"))
        p.output = j.at("output").get<Output>();
}

// Responses Struct
struct Responses
{
    std::vector<Response> responses;
    bool shutdown;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Responses, responses, shutdown)

Response convert(ex::Response&& response)
{
    auto request_id = response.getRequestId();
    auto client_id  = response.getClientId();

    if (response.hasError())
    {
        auto error_msg = response.getErrorMsg();
        return Response{request_id, client_id, {error_msg}, std::nullopt};
    }

    auto e_output = response.getResult();

    auto is_final = e_output.isFinal;

    assert(e_output.outputTokenIds.size() == 1);
    auto token_ids = std::move(e_output.outputTokenIds[0]);

    auto output = Output{is_final, std::move(token_ids), std::nullopt, std::nullopt, std::nullopt};

    if (e_output.cumLogProbs.has_value())
    {
        assert(e_output.cumLogProbs.value().size() == 1);
        output.cum_log_prob = {e_output.cumLogProbs.value()[0]};
    }

    if (e_output.logProbs.has_value())
    {
        assert(e_output.logProbs.value().size() == 1);
        output.log_probs = {std::move(e_output.logProbs.value()[0])};
    }

    if (e_output.finishReasons.size() > 0)
    {
        assert(e_output.finishReasons.size() == 1);
        auto finish_reason = static_cast<FinishReasonEnum>(e_output.finishReasons[0]);
        if (finish_reason != FinishReasonEnum::FINISH_REASON_NOT_DONE)
        {
            output.finish_reason = {finish_reason};
        }
    }

    return Response{request_id, client_id, std::nullopt, {output}};
}

std::string serialize_responses(std::deque<ex::Response> responses, bool shutdown)
{
    auto object     = Responses{};
    object.shutdown = shutdown;

    while (!responses.empty())
    {
        auto response = std::move(responses.front());
        responses.pop_front();

        auto r = convert(std::move(response));
        assert(r.output.has_value() || r.error_msg.has_value());
        object.responses.emplace_back(std::move(r));
    }

    return json(object).dump();
}

}  // namespace nvidia::nvllm::trt
