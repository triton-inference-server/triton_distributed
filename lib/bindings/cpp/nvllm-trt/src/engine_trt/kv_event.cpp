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

#include "engine_trt/kv_event.hpp"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <xxhash.h>

#include <optional>
#include <string>
#include <vector>

using json   = nlohmann::json;
namespace ex = tensorrt_llm::executor;

namespace tensorrt_llm::executor {

// Serialization for KVCacheRemovedData
void to_json(json& j, const KVCacheRemovedData& data)
{
    j = json{{"block_hashes", data.blockHashes}};
}

void from_json(const json& j, KVCacheRemovedData& data)
{
    j.at("block_hashes").get_to(data.blockHashes);
}

}  // namespace tensorrt_llm::executor

namespace nvidia::nvllm::trt {

using IdType      = ex::IdType;
using TokenIdType = ex::TokenIdType;

struct KVCacheStoredBlockData
{
    KVCacheStoredBlockData() = default;
    KVCacheStoredBlockData(const ex::KVCacheStoredBlockData& data)
    {
        std::vector<TokenIdType> tokens;
        for (auto& token : data.tokens)
        {
            tokens.push_back(token.tokenId);
        }
        auto size = tokens.size() * sizeof(TokenIdType);
        auto hash = XXH3_64bits_withSeed(tokens.data(), size, 1337);

        this->block_hash  = data.blockHash;
        this->tokens_hash = hash;
        this->lora_id     = data.loraId;
    }

    /// @brief The hash of the block
    IdType block_hash;

    /// @brief The tokens in the block
    IdType tokens_hash;

    /// @brief The Lora ID of the block
    IdType lora_id;
};

// Serialization for KVCacheStoredBlockData
void to_json(json& j, const KVCacheStoredBlockData& data)
{
    j = json{
        {"block_hash", data.block_hash},
        {"tokens_hash", data.tokens_hash},
        {"lora_id", data.lora_id},
    };
}

void from_json(const json& j, KVCacheStoredBlockData& data)
{
    j.at("block_hash").get_to(data.block_hash);
    j.at("tokens_hash").get_to(data.tokens_hash);
    j.at("lora_id").get_to(data.lora_id);
}

struct KVCacheStoredData
{
    KVCacheStoredData() = default;
    KVCacheStoredData(ex::KVCacheStoredData&& data) : parent_hash(std::move(data.parentHash))
    {
        for (auto& block : data.blocks)
        {
            blocks.emplace_back(block);
        }
    }

    /// @brief The parent of this sequence of stored blocks
    std::optional<IdType> parent_hash;

    /// @brief A sequence of blocks. The parent of block `i` is block `i-1`
    std::vector<KVCacheStoredBlockData> blocks;
};

using KVCacheRemovedData = ex::KVCacheRemovedData;

// Serialization for KVCacheStoredData
void to_json(json& j, const KVCacheStoredData& data)
{
    j = json{{"blocks", data.blocks}};

    if (data.parent_hash)
    {
        j["parent_hash"] = data.parent_hash.value();
    }
}

void from_json(const json& j, KVCacheStoredData& data)
{
    j.at("blocks").get_to(data.blocks);

    if (j.contains("parent_hash"))
    {
        data.parent_hash = j.at("parent_hash").get<IdType>();
    }
}

struct KVCacheEventData
{
    KVCacheEventData() = default;
    explicit KVCacheEventData(ex::KVCacheEventData&& data)
    {
        if (std::holds_alternative<ex::KVCacheStoredData>(data))
        {
            stored = KVCacheStoredData(std::move(std::get<ex::KVCacheStoredData>(data)));
        }
        else if (std::holds_alternative<ex::KVCacheRemovedData>(data))
        {
            removed = std::move(std::get<ex::KVCacheRemovedData>(data));
        }
    }

    std::optional<KVCacheStoredData> stored;
    std::optional<KVCacheRemovedData> removed;
};

// Serialization for KVCacheEventData
void to_json(json& j, const KVCacheEventData& data)
{
    if (data.stored)
    {
        j["stored"] = data.stored.value();
    }
    else if (data.removed)
    {
        j["removed"] = data.removed.value();
    }
}

void from_json(const json& j, KVCacheEventData& data)
{
    if (j.contains("stored"))
    {
        data.stored = {j.at("stored").get<KVCacheStoredData>()};
    }
    else if (j.contains("removed"))
    {
        data.removed = {j.at("removed").get<KVCacheRemovedData>()};
    }
}

struct KVCacheEvent
{
    KVCacheEvent(IdType eventId, KVCacheEventData data);
    KVCacheEvent(ex::KVCacheEvent&& event) : event_id(std::move(event.eventId)), data(std::move(event.data)) {}

    /// @brief The unique id of this event
    IdType event_id;
    /// @brief The data corresponding to this event
    KVCacheEventData data;
};

inline void to_json(json& j, const KVCacheEvent& event)
{
    j = json{{"event_id", event.event_id}, {"data", event.data}};
}

inline void from_json(const json& j, KVCacheEvent& event)
{
    j.at("event_id").get_to(event.event_id);
    j.at("data").get_to(event.data);
}

struct KVCacheEvents
{
    std::vector<KVCacheEvent> events;
    bool shutdown;
};

inline void to_json(json& j, const KVCacheEvents& events)
{
    j = json{{"events", events.events}, {"shutdown", events.shutdown}};
}

// inline void from_json(const json& j, KVCacheEvents& events)
// {
//     j.at("events").get_to(events.events);
//     j.at("shutdown").get_to(events.shutdown);
// }

std::string serialize_kv_events(std::deque<tensorrt_llm::executor::KVCacheEvent> events_in, bool shutdown)
{
    std::vector<KVCacheEvent> events_out;

    while (!events_in.empty())
    {
        auto event = events_in.front();
        events_in.pop_front();

        if (std::holds_alternative<ex::KVCacheCreatedData>(event.data) ||
            std::holds_alternative<ex::KVCacheUpdatedData>(event.data))
        {
            continue;
        }

        events_out.emplace_back(std::move(event));
    }

    KVCacheEvents events{std::move(events_out), shutdown};

    return json(events).dump();
}

}  // namespace nvidia::nvllm::trt
