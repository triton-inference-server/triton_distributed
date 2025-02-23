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

use tempfile::tempdir;
use triton_llm::model_card::model::{
    ModelDeploymentCard, ModelInfoType, PromptFormatterArtifact, TokenizerKind,
};

#[tokio::test]
async fn test_model_info_from_hf_like_local_repo() {
    let path = "tests/data/sample-models/mock-llama-3.1-8b-instruct";
    let mdc = ModelDeploymentCard::from_local_path(path).await.unwrap();
    let info = mdc.model_info.get_model_info().await.unwrap();
    assert_eq!(info.model_type(), "llama");
    assert_eq!(info.bos_token_id(), 128000);
    assert_eq!(info.eos_token_ids(), vec![128009]);
    assert_eq!(info.max_position_embeddings(), 8192);
    assert_eq!(info.vocab_size(), 128256);
}

#[tokio::test]
async fn test_model_info_from_non_existent_local_repo() {
    let path = "tests/data/sample-models/this-model-does-not-exist";
    let result = ModelDeploymentCard::from_local_path(path).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_tokenizer_from_hf_like_local_repo() {
    let path = "tests/data/sample-models/mock-llama-3.1-8b-instruct";
    let mdc = ModelDeploymentCard::from_local_path(path).await.unwrap();
    // Verify tokenizer file was found
    match mdc.tokenizer {
        TokenizerKind::HfTokenizerJson(_) => (),
        _ => panic!("Expected HfTokenizerJson"),
    }
}

#[tokio::test]
async fn test_prompt_formatter_from_hf_like_local_repo() {
    let path = "tests/data/sample-models/mock-llama-3.1-8b-instruct";
    let mdc = ModelDeploymentCard::from_local_path(path).await.unwrap();
    // Verify prompt formatter was found
    match mdc.prompt_formatter {
        Some(PromptFormatterArtifact::HfTokenizerConfigJson(_)) => (),
        _ => panic!("Expected HfTokenizerConfigJson prompt formatter"),
    }
}

#[tokio::test]
async fn test_missing_required_files() {
    // Create empty temp directory
    let temp_dir = tempdir().unwrap();
    let result = ModelDeploymentCard::from_local_path(temp_dir.path()).await;
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    // Should fail because config.json is missing
    assert!(err.contains("unable to extract"));
}
