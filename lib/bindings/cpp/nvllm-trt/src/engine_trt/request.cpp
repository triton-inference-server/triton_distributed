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

#include "engine_trt/request.hpp"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

using json   = nlohmann::json;
namespace ex = tensorrt_llm::executor;

namespace nvidia::nvllm::trt {

// SamplingConfig Struct
struct SamplingConfig
{
    uint32_t beam_width = 1;
    std::optional<uint32_t> top_k;
    std::optional<float> top_p;
    std::optional<float> top_p_min;
    std::optional<uint32_t> top_p_reset_ids;
    std::optional<float> top_p_decay;
    std::optional<uint32_t> seed;
    std::optional<float> temperature;
    std::optional<uint32_t> min_tokens;
    std::optional<float> beam_search_diversity_rate;
    std::optional<float> repetition_penalty;
    std::optional<float> presence_penalty;
    std::optional<float> frequency_penalty;
    std::optional<float> length_penalty;
    std::optional<uint32_t> early_stopping;
    std::optional<uint32_t> no_repeat_ngram_size;
    std::optional<uint32_t> num_return_sequences;

    ex::SamplingConfig to_executor_config() const
    {
        return ex::SamplingConfig(beam_width,
                                  top_k,
                                  top_p,
                                  top_p_min,
                                  top_p_reset_ids,
                                  top_p_decay,
                                  seed,
                                  temperature,
                                  min_tokens,
                                  beam_search_diversity_rate,
                                  repetition_penalty,
                                  presence_penalty,
                                  frequency_penalty,
                                  length_penalty,
                                  early_stopping,
                                  no_repeat_ngram_size,
                                  num_return_sequences);
    }
};

// Custom to_json and from_json functions for SamplingConfig
inline void to_json(json& j, const SamplingConfig& s)
{
    j = json{{"beam_width", s.beam_width}};
    if (s.top_k)
        j["top_k"] = s.top_k.value();
    if (s.top_p)
        j["top_p"] = s.top_p.value();
    if (s.top_p_min)
        j["top_p_min"] = s.top_p_min.value();
    if (s.top_p_reset_ids)
        j["top_p_reset_ids"] = s.top_p_reset_ids.value();
    if (s.top_p_decay)
        j["top_p_decay"] = s.top_p_decay.value();
    if (s.seed)
        j["seed"] = s.seed.value();
    if (s.temperature)
        j["temperature"] = s.temperature.value();
    if (s.min_tokens)
        j["min_tokens"] = s.min_tokens.value();
    if (s.beam_search_diversity_rate)
        j["beam_search_diversity_rate"] = s.beam_search_diversity_rate.value();
    if (s.repetition_penalty)
        j["repetition_penalty"] = s.repetition_penalty.value();
    if (s.presence_penalty)
        j["presence_penalty"] = s.presence_penalty.value();
    if (s.frequency_penalty)
        j["frequency_penalty"] = s.frequency_penalty.value();
    if (s.length_penalty)
        j["length_penalty"] = s.length_penalty.value();
    if (s.early_stopping)
        j["early_stopping"] = s.early_stopping.value();
    if (s.no_repeat_ngram_size)
        j["no_repeat_ngram_size"] = s.no_repeat_ngram_size.value();
    if (s.num_return_sequences)
        j["num_return_sequences"] = s.num_return_sequences.value();
}

inline void from_json(const json& j, SamplingConfig& s)
{
    j.at("beam_width").get_to(s.beam_width);
    if (j.contains("top_k"))
        s.top_k = j.at("top_k").get<uint32_t>();
    if (j.contains("top_p"))
        s.top_p = j.at("top_p").get<float>();
    if (j.contains("top_p_min"))
        s.top_p_min = j.at("top_p_min").get<float>();
    if (j.contains("top_p_reset_ids"))
        s.top_p_reset_ids = j.at("top_p_reset_ids").get<uint32_t>();
    if (j.contains("top_p_decay"))
        s.top_p_decay = j.at("top_p_decay").get<float>();
    if (j.contains("seed"))
        s.seed = j.at("seed").get<uint32_t>();
    if (j.contains("temperature"))
        s.temperature = j.at("temperature").get<float>();
    if (j.contains("min_tokens"))
        s.min_tokens = j.at("min_tokens").get<uint32_t>();
    if (j.contains("beam_search_diversity_rate"))
        s.beam_search_diversity_rate = j.at("beam_search_diversity_rate").get<float>();
    if (j.contains("repetition_penalty"))
        s.repetition_penalty = j.at("repetition_penalty").get<float>();
    if (j.contains("presence_penalty"))
        s.presence_penalty = j.at("presence_penalty").get<float>();
    if (j.contains("frequency_penalty"))
        s.frequency_penalty = j.at("frequency_penalty").get<float>();
    if (j.contains("length_penalty"))
        s.length_penalty = j.at("length_penalty").get<float>();
    if (j.contains("early_stopping"))
        s.early_stopping = j.at("early_stopping").get<uint32_t>();
    if (j.contains("no_repeat_ngram_size"))
        s.no_repeat_ngram_size = j.at("no_repeat_ngram_size").get<uint32_t>();
    if (j.contains("num_return_sequences"))
        s.num_return_sequences = j.at("num_return_sequences").get<uint32_t>();
}

// OutputConfig Struct
struct OutputConfig
{
    bool return_log_probs;
    bool return_context_logits;
    bool return_generation_logits;
    bool exclude_input_from_output;
    bool return_encoder_output;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(OutputConfig,
                                   return_log_probs,
                                   return_context_logits,
                                   return_generation_logits,
                                   exclude_input_from_output,
                                   return_encoder_output)

// RetentionPriorityAndDuration Struct
struct RetentionPriorityAndDuration
{
    std::optional<uint32_t> retention_priority;
    std::optional<uint64_t> duration_ms;
};

inline void to_json(json& j, const RetentionPriorityAndDuration& r)
{
    if (r.retention_priority)
        j["retention_priority"] = r.retention_priority.value();
    if (r.duration_ms)
        j["duration_ms"] = r.duration_ms.value();
}

inline void from_json(const json& j, RetentionPriorityAndDuration& r)
{
    if (j.contains("retention_priority"))
        r.retention_priority = j.at("retention_priority").get<uint32_t>();
    if (j.contains("duration_ms"))
        r.duration_ms = j.at("duration_ms").get<uint64_t>();
}

// TokenRangeRetentionConfig Struct
struct TokenRangeRetentionConfig
{
    uint32_t token_start;
    std::optional<uint32_t> token_end;
    uint32_t priority;
    std::optional<uint64_t> duration_ms;
};

inline void to_json(json& j, const TokenRangeRetentionConfig& t)
{
    j = json{{"token_start", t.token_start}, {"priority", t.priority}};
    if (t.token_end)
        j["token_end"] = t.token_end.value();
    if (t.duration_ms)
        j["duration_ms"] = t.duration_ms.value();
}

inline void from_json(const json& j, TokenRangeRetentionConfig& t)
{
    j.at("token_start").get_to(t.token_start);
    j.at("priority").get_to(t.priority);
    if (j.contains("token_end"))
        t.token_end = j.at("token_end").get<uint32_t>();
    if (j.contains("duration_ms"))
        t.duration_ms = j.at("duration_ms").get<uint64_t>();
}

// KvCacheRetentionConfig Struct
struct KvCacheRetentionConfig
{
    std::vector<TokenRangeRetentionConfig> token_range_retention_configs;
    uint32_t decode_retention_priority;
    std::optional<uint64_t> decode_duration_ms;
};

inline void to_json(json& j, const KvCacheRetentionConfig& k)
{
    j = json{{"token_range_retention_configs", k.token_range_retention_configs},
             {"decode_retention_priority", k.decode_retention_priority}};
    if (k.decode_duration_ms)
        j["decode_duration_ms"] = k.decode_duration_ms.value();
}

inline void from_json(const json& j, KvCacheRetentionConfig& k)
{
    j.at("token_range_retention_configs").get_to(k.token_range_retention_configs);
    j.at("decode_retention_priority").get_to(k.decode_retention_priority);
    if (j.contains("decode_duration_ms"))
        k.decode_duration_ms = j.at("decode_duration_ms").get<uint64_t>();
}

// Request Struct
struct Request
{
    std::vector<int32_t> input_token_ids;
    uint32_t max_tokens;
    bool streaming;
    std::optional<SamplingConfig> sampling_config;
    std::optional<OutputConfig> output_config;
    std::optional<uint32_t> end_id;
    // std::optional<uint32_t> pad_id;
    // std::vector<uint32_t> position_ids;
    // std::vector<uint32_t> bad_words;
    // std::vector<uint32_t> stop_words;
    // std::vector<uint8_t> embedding_bias;  // bytes
    // // TODO: Add ExternalDraftTokensConfig external_draft_tokens_config;
    // // TODO: Add PromptTuningConfig prompt_tuning_config;
    // // TODO: Add LoraConfig lora_config;
    // // TODO: Add LookaheadDecodingConfig lookahead_config;
    // KvCacheRetentionConfig kv_cache_retention_config;
    // std::string logits_post_processor_name;
    // std::vector<uint32_t> encoder_input_token_ids;
    // std::optional<uint64_t> client_id;
    // bool return_all_generated_tokens;
    // float priority;
    // uint32_t request_type;
    // // TODO: Add ContextPhaseParams context_phase_params;
    // std::vector<uint8_t> encoder_input_features;  // bytes
    // std::optional<uint32_t> encoder_output_length;
    // std::vector<uint8_t> cross_attention_mask;  // bytes
    // uint32_t num_return_sequences;
    // // TODO: Add EagleConfig eagle_config;
    // std::vector<uint8_t> skip_cross_attn_blocks;  // bytes
};

// Custom to_json and from_json functions for Request
inline void to_json(json& j, const Request& r)
{
    j = json{
        {"input_token_ids", r.input_token_ids},
        {"max_tokens", r.max_tokens},
        {"streaming", r.streaming},
        // {"sampling_config", r.sampling_config},
        // {"output_config", r.output_config},
        //  {"position_ids", r.position_ids},
        //  {"bad_words", r.bad_words},
        //  {"stop_words", r.stop_words},
        //  {"kv_cache_retention_config", r.kv_cache_retention_config},
        //  {"logits_post_processor_name", r.logits_post_processor_name},
        //  {"encoder_input_token_ids", r.encoder_input_token_ids},
        //  {"return_all_generated_tokens", r.return_all_generated_tokens},
        //  {"priority", r.priority},
        //  {"request_type", r.request_type},
        //  {"num_return_sequences", r.num_return_sequences}
    };

    if (r.sampling_config)
        j["sampling_config"] = r.sampling_config.value();
    if (r.output_config)
        j["output_config"] = r.output_config.value();

    if (r.end_id)
        j["end_id"] = r.end_id.value();
    // if (r.pad_id)
    //     j["pad_id"] = r.pad_id.value();
    // if (!r.embedding_bias.empty())
    //     j["embedding_bias"] = r.embedding_bias;
    // if (r.client_id)
    //     j["client_id"] = r.client_id.value();
    // if (!r.encoder_input_features.empty())
    //     j["encoder_input_features"] = r.encoder_input_features;
    // if (r.encoder_output_length)
    //     j["encoder_output_length"] = r.encoder_output_length.value();
    // if (!r.cross_attention_mask.empty())
    //     j["cross_attention_mask"] = r.cross_attention_mask;
    // if (!r.skip_cross_attn_blocks.empty())
    //     j["skip_cross_attn_blocks"] = r.skip_cross_attn_blocks;
}

inline void from_json(const json& j, Request& r)
{
    j.at("input_token_ids").get_to(r.input_token_ids);
    j.at("max_tokens").get_to(r.max_tokens);
    j.at("streaming").get_to(r.streaming);

    if (j.contains("sampling_config"))
        r.sampling_config = j.at("sampling_config").get<SamplingConfig>();

    if (j.contains("output_config"))
        r.output_config = j.at("output_config").get<OutputConfig>();

    // j.at("sampling_config").get_to(r.sampling_config);
    // j.at("output_config").get_to(r.output_config);
    // j.at("position_ids").get_to(r.position_ids);
    // j.at("bad_words").get_to(r.bad_words);
    // j.at("stop_words").get_to(r.stop_words);
    // j.at("kv_cache_retention_config").get_to(r.kv_cache_retention_config);
    // j.at("logits_post_processor_name").get_to(r.logits_post_processor_name);
    // j.at("encoder_input_token_ids").get_to(r.encoder_input_token_ids);
    // j.at("return_all_generated_tokens").get_to(r.return_all_generated_tokens);
    // j.at("priority").get_to(r.priority);
    // j.at("request_type").get_to(r.request_type);
    // j.at("num_return_sequences").get_to(r.num_return_sequences);

    if (j.contains("end_id"))
        r.end_id = j.at("end_id").get<uint32_t>();

    // if (j.contains("pad_id"))
    //     r.pad_id = j.at("pad_id").get<uint32_t>();
    // if (j.contains("embedding_bias"))
    //     r.embedding_bias = j.at("embedding_bias").get<std::vector<uint8_t>>();
    // if (j.contains("client_id"))
    //     r.client_id = j.at("client_id").get<uint64_t>();
    // if (j.contains("encoder_input_features"))
    //     r.encoder_input_features = j.at("encoder_input_features").get<std::vector<uint8_t>>();
    // if (j.contains("encoder_output_length"))
    //     r.encoder_output_length = j.at("encoder_output_length").get<uint32_t>();
    // if (j.contains("cross_attention_mask"))
    //     r.cross_attention_mask = j.at("cross_attention_mask").get<std::vector<uint8_t>>();
    // if (j.contains("skip_cross_attn_blocks"))
    //     r.skip_cross_attn_blocks = j.at("skip_cross_attn_blocks").get<std::vector<uint8_t>>();
}

tensorrt_llm::executor::Request deserialize_request(const std::string& request_proto)
{
    spdlog::trace("Deserializing request json: {}", request_proto);

    auto j      = json::parse(request_proto);
    auto req_in = j.get<Request>();

    spdlog::trace("constructing request with {} input tokens; max tokens: {}",
                  req_in.input_token_ids.size(),
                  req_in.max_tokens);
    tensorrt_llm::executor::Request request(std::move(req_in.input_token_ids), req_in.max_tokens, true);

    if (req_in.sampling_config)
    {
        spdlog::trace("Setting sampling_config");
        request.setSamplingConfig(req_in.sampling_config->to_executor_config());
    }

    if (req_in.end_id)
    {
        spdlog::trace("Setting end_id: {}", req_in.end_id.value());
        request.setEndId(req_in.end_id.value());
    }

    return request;
}

}  // namespace nvidia::nvllm::trt
