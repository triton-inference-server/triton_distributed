//! Prompt Formatting Module
//!
//! This module is responsible for formatting the prompt of an LLM request/turn.

// TODO: query if `add_generation_prompt` is in the prompt template
// TOOD: only models which support add_generation_prompt can support:
//       - PALS
//       - Continuation - Continuation is detected if the request is a user turn.
//         We could send back the partial assistant response, do not enable
//         add_generation_prompt, and let the LLM continue generating the response.

use anyhow::Result;
use minijinja::value::Value;
use std::sync::Arc;

mod template;

pub use template::ContextMixins;

/// Trait that defines a request that can map to an OpenAI-like request.
pub trait OAIChatLikeRequest {
    fn messages(&self) -> Value;
    fn tools(&self) -> Option<Value> {
        None
    }
    fn tool_choice(&self) -> Option<Value> {
        None
    }

    fn should_add_generation_prompt(&self) -> bool;
}

pub trait OAIPromptFormatter: Send + Sync + 'static {
    fn supports_add_generation_prompt(&self) -> bool;
    fn render(&self, req: &dyn OAIChatLikeRequest) -> Result<String>;
}

pub enum PromptFormatter {
    OAI(Arc<dyn OAIPromptFormatter>),
}
