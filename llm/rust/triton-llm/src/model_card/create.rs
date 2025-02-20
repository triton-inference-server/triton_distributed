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

use std::collections::HashMap;

use std::path::Path;
use std::fs;
use crate::model_card::model::ModelDeploymentCard;
use anyhow::{Context, Result};

use crate::model_card::model::{ModelInfoType, TokenizerKind, PromptFormatterArtifact, File};

impl ModelDeploymentCard {
    /// Attempt to create a MDC from a local directory
    pub async fn from_local_path(local_root_dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        let local_root_dir = local_root_dir.as_ref();
        check_valid_local_repo_path(local_root_dir)?;
        let repo_id = format!("{}", local_root_dir.canonicalize()?.display());
        let model_name = local_root_dir.file_name().unwrap().to_str().unwrap();
        Self::from_repo(&repo_id, model_name).await
    }

    /// Attempt to auto-detect model type and construct an MDC from a NGC repo
    pub async fn from_ngc_repo(_: &str) -> anyhow::Result<Self> {
        Err(anyhow::anyhow!("ModelDeploymentCard::from_ngc_repo is not implemented in model-card-local"))
    }

    pub async fn from_repo(repo_id: &str, model_name: &str) -> anyhow::Result<Self> {
        Ok(Self {
            display_name: model_name.to_string(),
            service_name: model_name.to_string(),
            model_info: ModelInfoType::from_repo(repo_id).await?,
            tokenizer: TokenizerKind::from_repo(repo_id).await?,
            prompt_formatter: PromptFormatterArtifact::from_repo(repo_id).await?,
            prompt_context: None, // TODO - auto-detect prompt context
            revision: 0,
            last_published: None,
            requires_preprocessing: true,
        })
    }
}

impl ModelInfoType {
    pub async fn from_repo(repo_id: &str) -> Result<Self> {
        Self::try_is_hf_repo(repo_id)
            .await
            .with_context(|| format!("unable to extract model info from repo {}", repo_id))
    }

    async fn try_is_hf_repo(repo: &str) -> anyhow::Result<Self> {
        Ok(Self::HfConfigJson(
            check_for_file(repo, "config.json").await?,
        ))
    }
}

impl PromptFormatterArtifact {
    pub async fn from_repo(repo_id: &str) -> Result<Option<Self>> {
        // we should only error if we expect a prompt formatter and it's not found
        // right now, we don't know when to expect it, so we just return Ok(Some/None)
        Ok(Self::try_is_hf_repo(repo_id)
            .await
            .with_context(|| format!("unable to extract prompt format from repo {}", repo_id))
            .ok())
    }

    async fn try_is_hf_repo(repo: &str) -> anyhow::Result<Self> {
        Ok(Self::HfTokenizerConfigJson(
            check_for_file(repo, "tokenizer_config.json").await?,
        ))
    }
}

impl TokenizerKind {
    pub async fn from_repo(repo_id: &str) -> Result<Self> {
        Self::try_is_hf_repo(repo_id)
            .await
            .with_context(|| format!("unable to extract tokenizer kind from repo {}", repo_id))
    }

    async fn try_is_hf_repo(repo: &str) -> anyhow::Result<Self> {
        Ok(Self::HfTokenizerJson(
            check_for_file(repo, "tokenizer.json").await?,
        ))
    }
}

async fn check_for_file(repo_id: &str, file: &str) -> anyhow::Result<File> {
    let mut files = check_for_files(repo_id, vec![file.to_string()]).await?;
    let file = files
        .remove(file)
        .ok_or(anyhow::anyhow!("file {} not found", file))?;
    Ok(file)
}

async fn check_for_files(repo_id: &str, files: Vec<String>) -> Result<HashMap<String, File>> {
    let files: HashMap<String, File> = fs::read_dir(repo_id)?
        .map(|entry| {
            let path = entry.unwrap().path();
            let file_name = path.file_name().unwrap().to_str().unwrap();
            (file_name.to_string(), path.display().to_string())
        })
        .filter(|(file_name, _)| files.contains(&file_name))
        .collect();
    Ok(files)
}

/// Check if the path is a valid local repo path
fn check_valid_local_repo_path(path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    if !path.exists() {
        return Err(anyhow::anyhow!("path does not exist"));
    }

    if !path.is_dir() {
        return Err(anyhow::anyhow!("path is not a directory"));
    }
    Ok(())
}