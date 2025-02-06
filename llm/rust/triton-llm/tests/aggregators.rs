// use futures::StreamExt;
// use triton_llm::protocols::{
//     codec::{create_message_stream, Message},
//     error::SseCodecError,
//     openai::{chat_completions::ChatCompletionResponse, completions::CompletionResponse},
//     ContentProvider, DataStream,
// };

// const CMPL_ROOT_PATH: &str = "tests/data/replays/meta/llama-3.1-8b-instruct/completions";
// const CHAT_ROOT_PATH: &str = "tests/data/replays/meta/llama-3.1-8b-instruct/chat_completions";

// fn create_stream(root_path: &str, file_name: &str) -> DataStream<Result<Message, SseCodecError>> {
//     let data = std::fs::read_to_string(format!("{}/{}", root_path, file_name)).unwrap();
//     create_message_stream(&data)
// }

// #[tokio::test]
// async fn test_openai_chat_stream() {
//     let data = std::fs::read_to_string("tests/data/replays/meta/llama-3.1-8b-instruct/chat_completions/chat-completion.streaming.1").unwrap();

//     // note: we are only taking the first 16 messages to keep the size of the response small
//     let stream = create_message_stream(&data).take(16);
//     let result = ChatCompletionResponse::from_sse_stream(Box::pin(stream))
//         .await
//         .unwrap();

//     // todo: provide a cleaner way to extract the content from choices
//     assert_eq!(
//         result.choices.first().unwrap().content(),
//         "Deep learning is a subfield of machine learning that involves the use of artificial"
//     );
// }

// #[tokio::test]
// async fn test_openai_chat_edge_case_multi_line_data() {
//     let stream = create_stream(CHAT_ROOT_PATH, "edge_cases/valid-multi-line-data");
//     let result = ChatCompletionResponse::from_sse_stream(Box::pin(stream))
//         .await
//         .unwrap();

//     assert_eq!(result.choices.first().unwrap().content(), "Deep learning");
// }

// #[tokio::test]
// async fn test_openai_chat_edge_case_comments_per_response() {
//     let stream = create_stream(CHAT_ROOT_PATH, "edge_cases/valid-comments_per_response");
//     let result = ChatCompletionResponse::from_sse_stream(Box::pin(stream))
//         .await
//         .unwrap();

//     assert_eq!(result.choices.first().unwrap().content(), "Deep learning");
// }

// #[tokio::test]
// async fn test_openai_chat_edge_case_invalid_deserialize_error() {
//     let stream = create_stream(CHAT_ROOT_PATH, "edge_cases/invalid-deserialize_error");
//     let result = ChatCompletionResponse::from_sse_stream(Box::pin(stream)).await;

//     assert!(result.is_err());
//     // insta::assert_debug_snapshot!(result);
// }

// // =============================
// // Completions (/v1/completions)
// // =============================

// #[tokio::test]
// async fn test_openai_cmpl_stream() {
//     let stream = create_stream(CMPL_ROOT_PATH, "completion.streaming.1").take(16);
//     let result = CompletionResponse::from_sse_stream(Box::pin(stream))
//         .await
//         .unwrap();

//     // todo: provide a cleaner way to extract the content from choices
//     assert_eq!(
//         result.choices.first().unwrap().content(),
//         " This is a question that is often asked by those outside of AI research and development"
//     );
// }
