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

//! Codec Module
//!
//! Codec map structure into blobs of bytes and streams of bytes.
//!
//! In this module, we define three primary codec used to issue single, two-part or multi-part messages,
//! on a byte stream.

use tokio_util::{
    bytes::{Buf, BufMut, BytesMut},
    codec::{Decoder, Encoder},
};

mod two_part;

pub use two_part::{TwoPartCodec, TwoPartMessage, TwoPartMessageType};

// // Custom codec that reads a u64 length header and the message of that length
// #[derive(Default)]
// pub struct LengthPrefixedCodec;

// impl LengthPrefixedCodec {
//     pub fn new() -> Self {
//         LengthPrefixedCodec {}
//     }
// }

// impl Decoder for LengthPrefixedCodec {
//     type Item = Vec<u8>;
//     type Error = tokio::io::Error;

//     fn decode(&mut self, src: &mut BytesMut) -> Result<Option<Self::Item>, Self::Error> {
//         // Check if enough bytes are available to read the length (u64 = 8 bytes)
//         if src.len() < 8 {
//             return Ok(None); // Not enough data to read the length
//         }

//         // Read the u64 length header
//         let len = src.get_u64() as usize;

//         // Check if enough bytes are available to read the full message
//         if src.len() < len {
//             src.reserve(len - src.len()); // Reserve space for the remaining bytes
//             return Ok(None);
//         }

//         // Read the actual message bytes of the specified length
//         let data = src.split_to(len).to_vec();
//         Ok(Some(data))
//     }
// }

// impl Encoder<Vec<u8>> for LengthPrefixedCodec {
//     type Error = tokio::io::Error;

//     fn encode(&mut self, item: Vec<u8>, dst: &mut BytesMut) -> Result<(), Self::Error> {
//         // Write the length of the message as a u64 header
//         dst.put_u64(item.len() as u64);

//         // Write the actual message bytes
//         dst.put_slice(&item);
//         Ok(())
//     }
// }
