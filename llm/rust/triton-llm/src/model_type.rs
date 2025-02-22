use serde::{Deserialize, Serialize};

// Macro to define model types
macro_rules! define_model_types {
    ($(($variant:ident, $str_name:expr)),* $(,)?) => {
        #[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
        pub enum ModelType {
            $($variant),*
        }

        impl ModelType {
            pub fn as_str(&self) -> &'static str {
                match self {
                    $(Self::$variant => $str_name),*
                }
            }

            pub fn all() -> Vec<Self> {
                vec![$(Self::$variant),*]
            }
        }
    }
}

// Define all model types in one place
define_model_types!(
    (Chat, "chat"),
    (Completion, "completion"),
); 