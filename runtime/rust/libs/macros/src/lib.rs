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

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn, ReturnType};

/// A proc-macro which restructures code like:
///
/// ```rust,ignore
/// fn main() -> anyhow::Result<()> {
///     let worker = Worker::from_settings()?;
///     worker.execute(app)
/// }
///
/// async fn app(runtime: Runtime) -> anyhow::Result<()> {
///     # App logic here
/// }
/// ```
///
/// into a single-proc macro like:
///
/// ```rust,ignore
/// #[triton_distributed::main]
/// async fn app(runtime: Runtime) -> anyhow::Result<()> {
///    # App logic here
/// }
/// ```
#[proc_macro_attribute]
pub fn main(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // TODO: Moving forward, we may want to add some configuration options to this.

    // Parse the function we are annotating (the annotated item).
    let input_fn = parse_macro_input!(item as ItemFn);

    // Make sure it’s an `async fn`.
    if input_fn.sig.asyncness.is_none() {
        return syn::Error::new_spanned(
            &input_fn.sig.ident,
            "`#[triton_distributed::main]` can only be applied to `async fn`",
        )
        .to_compile_error()
        .into();
    }

    // Pull out pieces of the function signature and body
    let vis = &input_fn.vis; // e.g., `pub`
    let sig = &input_fn.sig; // the signature
    let ident = &sig.ident; // function name (e.g. `app`)
    let block = &input_fn.block; // the async fn body

    // Determine the return type, e.g. `-> Result<()>`.
    let output = match &sig.output {
        ReturnType::Default => quote! { () },
        ReturnType::Type(_, ty) => quote! { #ty },
    };

    // Generate code:
    // 1) A synchronous main() that calls worker.execute(...).
    // 2) The original async fn.
    //
    // We'll expand to something like:
    //
    // fn main() -> ResultType {
    //     let worker = Worker::from_settings()?;
    //     worker.execute(app)
    // }
    //
    // async fn app(runtime: Runtime) -> ResultType {
    //     ...
    // }
    let expanded = quote! {
        fn main() -> #output {
            let worker = triton_distributed::Worker::from_settings()?;
            worker.execute(#ident)
        }

        #vis #sig {
            #block
        }
    };

    TokenStream::from(expanded)
}
