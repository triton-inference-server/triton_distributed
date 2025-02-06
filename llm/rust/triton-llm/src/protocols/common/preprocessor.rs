use derive_builder::Builder;
use serde::{Deserialize, Serialize};

use super::{SamplingOptions, StopConditions};
use crate::protocols::TokenIdType;

/// [`PreprocessedRequest`] is the internal representation of an LLM request. The [`nim-llm-preprocessor`]
/// crate is responsible for converting request from the public APIs to this internal representation.
#[derive(Serialize, Deserialize, Debug, Clone, Builder)]
pub struct PreprocessedRequest {
    /// Type of prompt
    pub token_ids: Vec<TokenIdType>,

    /// StopConditions are conditions that the inference engine will use to stop generation.
    pub stop_conditions: StopConditions,

    /// SamplingOptions directs the inference engine to use sampling instead of greedy decoding.
    /// More documentation on how and on the order in which sampling options are applied
    /// are needed.
    pub sampling_options: SamplingOptions,

    /// The EOS token ID(s) for the Model
    /// Not every backend needs this, but those that do can find it here.
    /// TODO - refactor this to a better location
    #[builder(default)]
    pub eos_token_ids: Vec<TokenIdType>,

    /// The computed checksum of the Model Deployment Card (MDC).
    #[builder(default)]
    pub mdc_sum: Option<String>,

    /// User requested annotations for the request
    #[builder(default)]
    pub annotations: Vec<String>,
}

impl PreprocessedRequest {
    pub fn has_annotation(&self, annotation: &str) -> bool {
        self.annotations.contains(&annotation.to_string())
    }
}

impl PreprocessedRequest {
    pub fn builder() -> PreprocessedRequestBuilder {
        PreprocessedRequestBuilder::default()
    }
}
