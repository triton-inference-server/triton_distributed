use async_trait::async_trait;
use clap::{Args, Subcommand};
use serde::{Deserialize, Serialize};
use tracing as log;
use triton_distributed::{
    actions::{Action, ExecuteAction},
    error, protocols, DistributedRuntime, Result,
};

use crate::http::service::discovery::ModelState;

#[derive(Debug, Clone, Serialize, Deserialize, Subcommand)]
#[serde(rename_all = "snake_case")]
pub enum HttpAction {
    #[command(name = "create-route")]
    CreateRoute(CreateRouteOptions),

    #[command(name = "remove-route")]
    RemoveRoute(RemoveRouteOptions),

    #[command(name = "mark-ready")]
    MarkReady(MarkReadyOptions),

    #[command(name = "mark-unavailable")]
    MarkUnavailable(MarkUnavailableOptions),
}

#[derive(Debug, Clone, Serialize, Deserialize, clap::ValueEnum)]
pub enum HttpModelRequestFormat {
    #[serde(rename = "openai_chat")]
    OpenAIChat,

    #[serde(rename = "openai_cmpl")]
    OpenAICompletion,
}

impl std::fmt::Display for HttpModelRequestFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OpenAIChat => write!(f, "openai_chat"),
            Self::OpenAICompletion => write!(f, "openai_cmpl"),
        }
    }
}

#[derive(Debug, Clone, Args, Serialize, Deserialize)]
pub struct CreateRouteOptions {
    #[arg(long)]
    pub name: String,

    #[arg(long)]
    pub endpoint: protocols::EndpointAddress,

    #[arg(long, value_name = "FORMAT", default_value = "openai_chat")]
    pub format: HttpModelRequestFormat,
}

#[derive(Debug, Clone, Args, Serialize, Deserialize)]
pub struct RemoveRouteOptions {
    #[arg(long)]
    pub name: String,
}

#[derive(Debug, Clone, Args, Serialize, Deserialize)]
pub struct MarkReadyOptions {
    #[arg(long)]
    pub name: String,
}

#[derive(Debug, Clone, Args, Serialize, Deserialize)]
pub struct MarkUnavailableOptions {
    #[arg(long)]
    pub name: String,
}

pub struct HttpModelActionExecutor {
    drt: DistributedRuntime,
    ns: String,
}

impl HttpModelActionExecutor {
    pub fn new(drt: DistributedRuntime, ns: String) -> Self {
        Self { drt, ns }
    }

    async fn create_route(&self, opts: &CreateRouteOptions) -> Result<()> {
        // todo - add component "http" as a build options
        let component = self.drt.namespace(&self.ns)?.component("http")?;
        let etcd_client = self.drt.etcd_client();

        let request_format = opts.format.to_string();
        let path = format!("{}/models/{}", component.etcd_path(), opts.name);

        log::debug!(
            "Creating route for model {0} --> {1} stored at {path}",
            opts.name,
            opts.endpoint
        );

        // state
        let state_key = format!("{path}/state");
        let state_val = serde_json::to_vec_pretty(&ModelState::Ready)?;
        etcd_client.kv_put(state_key, state_val, None).await?;

        // route
        let route_key = format!("{path}/{request_format}.address");
        let route_val = opts.endpoint.as_str().as_bytes().to_vec();
        etcd_client.kv_create(route_key, route_val, None).await?;

        Ok(())
    }

    async fn remove_route(&self, opts: &RemoveRouteOptions) -> Result<()> {
        let component = self.drt.namespace(&self.ns)?.component("http")?;
        let etcd_client = self.drt.etcd_client();

        let path = format!("{}/models/{}", component.etcd_path(), opts.name);

        let mut client = self.drt.etcd_client().etcd_client().kv_client();
        let _deleted = client.delete(path, None).await?;
        Ok(())
    }
}

#[async_trait]
impl ExecuteAction<HttpAction> for HttpModelActionExecutor {
    async fn execute(&self, action: &HttpAction) -> Result<()> {
        match action {
            HttpAction::CreateRoute(opts) => self.create_route(opts).await,
            HttpAction::RemoveRoute(opts) => todo!(),
            HttpAction::MarkReady(opts) => todo!(),
            HttpAction::MarkUnavailable(opts) => todo!(),
        }
    }
}

// Action Creation Helpers
impl HttpAction {
    pub fn create_route(
        name: impl Into<String>,
        endpoint: impl AsRef<str>,
        format: Option<HttpModelRequestFormat>,
    ) -> Result<Self> {
        let endpoint_addr = endpoint
            .as_ref()
            .parse::<protocols::EndpointAddress>()
            .map_err(|_| error!("invalid endpoint address"))?;

        Ok(Self::CreateRoute(CreateRouteOptions {
            name: name.into(),
            endpoint: endpoint_addr.into(),
            format: format.unwrap_or(HttpModelRequestFormat::OpenAIChat),
        }))
    }

    pub fn remove_route(name: impl Into<String>) -> Self {
        Self::RemoveRoute(RemoveRouteOptions { name: name.into() })
    }

    pub fn mark_ready(name: impl Into<String>) -> Self {
        Self::MarkReady(MarkReadyOptions { name: name.into() })
    }

    pub fn mark_unavailable(name: impl Into<String>) -> Self {
        Self::MarkUnavailable(MarkUnavailableOptions { name: name.into() })
    }
}

impl HttpAction {
    pub fn to_action(&self, action: &str) -> Action {
        Action {
            action: action.to_string(),
            params: serde_json::to_value(self).unwrap(),
        }
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;
    use triton_distributed::actions::ActionRegistry;

    use super::*;
    use std::sync::{Arc, Mutex};

    #[derive(Debug, Clone, Parser)]
    #[command(name = "http")]
    pub struct HttpCommand {
        #[command(subcommand)]
        action: HttpAction,
    }

    impl From<HttpCommand> for HttpAction {
        fn from(cmd: HttpCommand) -> Self {
            cmd.action
        }
    }

    #[derive(Clone)]
    struct MockHttpModelExecutor {
        created_routes: Arc<Mutex<Vec<(String, protocols::Endpoint)>>>,
        removed_routes: Arc<Mutex<Vec<String>>>,
        ready_models: Arc<Mutex<Vec<String>>>,
        unavailable_models: Arc<Mutex<Vec<String>>>,
    }

    impl MockHttpModelExecutor {
        fn new() -> Self {
            Self {
                created_routes: Arc::new(Mutex::new(Vec::new())),
                removed_routes: Arc::new(Mutex::new(Vec::new())),
                ready_models: Arc::new(Mutex::new(Vec::new())),
                unavailable_models: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    #[async_trait]
    impl ExecuteAction<HttpAction> for MockHttpModelExecutor {
        async fn execute(&self, params: &HttpAction) -> Result<()> {
            match params {
                HttpAction::CreateRoute(opts) => {
                    self.created_routes
                        .lock()
                        .unwrap()
                        .push((opts.name.clone(), opts.endpoint.clone().into()));
                    Ok(())
                }
                HttpAction::RemoveRoute(opts) => {
                    self.removed_routes.lock().unwrap().push(opts.name.clone());
                    Ok(())
                }
                HttpAction::MarkReady(opts) => {
                    self.ready_models.lock().unwrap().push(opts.name.clone());
                    Ok(())
                }
                HttpAction::MarkUnavailable(opts) => {
                    self.unavailable_models
                        .lock()
                        .unwrap()
                        .push(opts.name.clone());
                    Ok(())
                }
            }
        }
    }

    #[tokio::test]
    async fn test_http_model_actions() {
        let mut registry = ActionRegistry::new();
        let executor = MockHttpModelExecutor::new();

        registry.register_typed::<HttpAction, _>("http", executor.clone());

        // Test programmatic usage
        registry
            .execute(
                "http",
                HttpAction::create_route(
                    "gpt-4",
                    "llm.inference.text-generation",
                    Some(HttpModelRequestFormat::OpenAIChat),
                )
                .unwrap(),
            )
            .await
            .unwrap();

        registry
            .execute("http", HttpAction::mark_ready("gpt-4"))
            .await
            .unwrap();

        // Test CLI usage
        let cmd = HttpCommand::parse_from([
            "http",
            "create-route",
            "--name",
            "gpt-5",
            "--endpoint",
            "llm.inference.chat",
        ]);

        // todo add action test
        registry
            .execute_action(cmd.action.to_action("http"))
            .await
            .unwrap();

        let created = executor.created_routes.lock().unwrap();
        assert_eq!(created.len(), 2);
        assert_eq!(created[0].0, "gpt-4");
        assert_eq!(created[0].1.to_string(), "llm.inference.text-generation");
        assert_eq!(created[1].0, "gpt-5");
        assert_eq!(created[1].1.to_string(), "llm.inference.chat");

        let ready = executor.ready_models.lock().unwrap();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0], "gpt-4");
    }
}
