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

//! Tokenizer Tests
//!
//! This module contains tests for the Tokenizer.
//!
//! For each tokenizer we use in production, we should have either a url to or a local copy
//! of either the tokenizer.json or the .model file.
//!
//! For a small set of common prompts, we need to have a hashable representation of the the encoding
//! object. We will precompute the hashes for each of these prompts for each tokenizer and store them
//! in a hashmap. We will then use these hashes to test that the tokenizer is working correctly. This
//! will detect if upstream dependency changes result in different/new behavior.

use std::collections::HashMap;
use std::sync::Arc;
use triton_llm::tokenizers::*;
use triton_llm::tokenizers::{traits::{Decoder, Encoder, Tokenizer}, Encoding, Error, Result};
use triton_llm::protocols::TokenIdType;

// these need to be in scope to use the traits on our concrete implementations

const TEST_PROMPTS: [&str; 4] = [
    "deep learning is",
    "Deep learning is",
    "has anyone seen nemo lately",
    "another prompt",
];

const HF_TOKENIZERS_LOCAL: [&str; 1] = ["fixtures/hf_llama-2.json"];
const SP_TOKENIZERS_LOCAL: [&str; 1] = ["fixtures/sp_nemo_43b_002.model"];

const HASHES: [(&str, [u64; 4]); 2] = [
    (
        "fixtures/hf_llama-2.json",
        [
            771185775798505393,
            8538328482215529710,
            17087868772360018644,
            1660219240238826577,
        ],
    ),
    (
        "fixtures/sp_nemo_43b_002.model",
        [
            4364584305793124512,
            14835610107630089099,
            4745649050194253372,
            7628184170446428562,
        ],
    ),
];

fn compute_hashes_for_tokenizer<E: Encoder>(tokenizer: &E, prompts: &[&str]) -> Vec<u64> {
    prompts
        .iter()
        .map(|&prompt| {
            tokenizer
                .encode(prompt)
                .expect("Failed to encode prompt")
                .get_hash()
            // Assuming `get_hash` returns a `u64`
        })
        .collect()
}

#[test]
fn compute_hashes() {
    let hash_map: HashMap<&str, [u64; 4]> = HASHES.iter().cloned().collect();

    for &tokenizer_name in SP_TOKENIZERS_LOCAL.iter() {
        let tokenizer = SentencePieceTokenizer::from_file(tokenizer_name)
            .expect("Failed to load remote HuggingFace tokenizer");

        let prompt_hashes = compute_hashes_for_tokenizer(&tokenizer, &TEST_PROMPTS);

        println!(
            "SP Tokenizer: {:?} Hashes: {:?}",
            tokenizer_name, prompt_hashes
        );

        assert_eq!(prompt_hashes, hash_map[tokenizer_name]);
    }

    for &tokenizer_name in HF_TOKENIZERS_LOCAL.iter() {
        let tokenizer = HuggingFaceTokenizer::from_file(tokenizer_name)
            .expect("Failed to load remote HuggingFace tokenizer");

        let prompt_hashes = compute_hashes_for_tokenizer(&tokenizer, &TEST_PROMPTS);

        println!(
            "HF Tokenizer: {:?} Hashes: {:?}",
            tokenizer_name, prompt_hashes
        );

        assert_eq!(prompt_hashes, hash_map[tokenizer_name]);
    }
}

#[test]
fn test_hf_lifecycle() {
    let tokenizer = HuggingFaceTokenizer::from_file(HF_TOKENIZERS_LOCAL[0])
        .expect("Failed to load remote HuggingFace tokenizer");

    let encoding = tokenizer
        .encode(TEST_PROMPTS[0])
        .expect("Failed to encode prompt");

    let decoded = tokenizer
        .decode(&encoding.token_ids, false)
        .expect("Failed to decode token_ids");

    assert_eq!(decoded, TEST_PROMPTS[0]);
}

#[test]
fn test_sequence() {
    let tokenizer = HuggingFaceTokenizer::from_file(HF_TOKENIZERS_LOCAL[0])
        .expect("Failed to load remote HuggingFace tokenizer");

    let shared_tokenizer = Arc::new(tokenizer);

    // let tokenizer = shared_tokenizer.read().unwrap();

    let encoding = shared_tokenizer
        .encode(TEST_PROMPTS[0])
        .expect("Failed to encode prompt");

    let mut sequence = Sequence::new(shared_tokenizer.clone().into());
    sequence
        .append_text(TEST_PROMPTS[0])
        .expect("Failed to append prompt");

    assert_eq!(sequence.len(), encoding.token_ids.len());

    let mut decoder = Sequence::new(shared_tokenizer.clone().into());

    let mut output = String::new();
    for token_id in encoding.token_ids.clone() {
        let text = decoder
            .append_token_id(token_id)
            .expect("Failed to decode token_id");
        output.push_str(text.as_str());
    }

    assert_eq!(decoder.len(), sequence.len());
    assert_eq!(decoder.token_ids(), sequence.token_ids());
    assert_eq!(output, TEST_PROMPTS[0]);

    let mut decoder = DecodeStream::new(shared_tokenizer.clone(), false);
    let mut output = String::new();
    for token_id in encoding.token_ids {
        let text = decoder.step(token_id).expect("Failed to decode token_id");
        if let Some(text) = text {
            output.push_str(text.as_str());
        }
    }
    assert_eq!(output, TEST_PROMPTS[0]);
}

#[test]
fn test_stop_sequence_decoder() {
    let tokenizer =
        Arc::new(SentencePieceTokenizer::from_file("fixtures/sp_nemo_43b_002.model").unwrap());

    // Test actual use case where <extra_id_1> was being returned, but should have been jailed and held back

    let mut encoded_sequence = Sequence::new(tokenizer.clone().into());

    encoded_sequence.append_text("I'm afraid I don't have access to that information.  I can tell you that I'm very proud of the way you've been acting this year, and I think you're doing a great job of being kind and helpful to others.  Keep up the good work, and I'm sure you'll be on the nice list this year!\n<extra_id_1>").unwrap();

    let mut decoded_sequence = Sequence::new(tokenizer.clone().into());

    let mut result = String::new();

    for token_id in encoded_sequence.token_ids() {
        let text = decoded_sequence
            .append_token_id(*token_id)
            .expect("Failed to decode token_id");
        println!("text: {}", text);
        result.push_str(&text);
    }

    assert!(result.contains("<extra_id_1>"));

    let mut decoder = StopSequenceDecoder::builder(tokenizer.clone().into())
        .add_stop_sequence_hidden("<extra_id_1>")
        .build()
        .expect("Failed to build StopSequenceDecoder");

    let mut result = String::new();

    for token_id in encoded_sequence.token_ids() {
        let text = decoder
            .append_token_id(*token_id)
            .expect("Failed to decode token_id");

        match text {
            SequenceDecoderOutput::Text(text) => {
                result.push_str(&text);
            }
            SequenceDecoderOutput::StoppedWithText(text) => {
                result.push_str(&text);
                break;
            }
            SequenceDecoderOutput::Stopped => {
                break;
            }
            _ => {}
        }
    }

    let result = result.trim_end();

    assert_eq!(result, "I'm afraid I don't have access to that information.  I can tell you that I'm very proud of the way you've been acting this year, and I think you're doing a great job of being kind and helpful to others.  Keep up the good work, and I'm sure you'll be on the nice list this year!");

    // Test that a stop sequence not at the end of a string will break the decoder loop
    // when a stop signal is triggered
    // In this case, the stop sequence is "I'm afraid I don't have access to that information."
    // And the result should be that nothing is returned since we jail text stop sequences

    let mut decoder = StopSequenceDecoder::builder(tokenizer.clone().into())
        .add_stop_sequence_hidden("I'm afraid I don't have access to that information.")
        .build()
        .expect("Failed to build StopSequenceDecoder");

    let mut result = String::new();

    for token_id in encoded_sequence.token_ids() {
        let text = decoder
            .append_token_id(*token_id)
            .expect("Failed to decode token_id");

        match text {
            SequenceDecoderOutput::Text(text) => {
                result.push_str(&text);
            }
            SequenceDecoderOutput::StoppedWithText(text) => {
                result.push_str(&text);
                break;
            }
            SequenceDecoderOutput::Stopped => {
                break;
            }
            _ => {}
        }
    }

    assert_eq!(result, "");

    // Test that stop_token_ids return the text for their token_id

    let mut decoder = StopSequenceDecoder::builder(tokenizer.clone().into())
        .add_stop_token_id_visible(251511) // "." token_id, which is different from "_."
        .build()
        .expect("Failed to build StopSequenceDecoder");

    let mut result = String::new();

    for token_id in encoded_sequence.token_ids() {
        let text = decoder
            .append_token_id(*token_id)
            .expect("Failed to decode token_id");

        match text {
            SequenceDecoderOutput::Text(text) => {
                result.push_str(&text);
            }
            SequenceDecoderOutput::StoppedWithText(text) => {
                result.push_str(&text);
                break;
            }
            SequenceDecoderOutput::Stopped => {
                break;
            }
            _ => {}
        }
    }

    assert_eq!(
        result,
        "I'm afraid I don't have access to that information."
    );

    // Test that addition calls to append_token_id after a stop event fail with an error

    assert!(decoder.append_token_id(1337).is_err());
}
