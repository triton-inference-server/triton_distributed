//! ZMQ Router Bridge
//!
//! Links the `inproc` sockets with the `ipc` sockets.
//!
//! Components names are used as the identities for `Dealer` sockets
//! which are prefixed as either `inproc-` or `ipc-` depending on their
//! location.
//!
//! Location should be etcd discoverable. Python components/endpoints will
//! always be `ipc-` but will have a Rust proxy.

use std::thread::JoinHandle;

use async_zmq::zmq;
use tokio_util::sync::CancellationToken;
use zmq::{PollItem, POLLIN};

use crate::{discovery::Lease, Result};

pub fn inproc_endpoint() -> String {
    "inproc://router_inproc".to_string()
}

pub fn ipc_endpoint(lease_id: i64) -> String {
    format!("ipc:///tmp/router_{lease_id}.ipc")
}

pub struct ThreadExecutionHandle {
    pub token: CancellationToken,
    pub handle: JoinHandle<()>,
}

#[derive(Debug, thiserror::Error)]
pub enum RouterBridgeError {
    #[error(transparent)]
    SendError(zmq::Error),

    #[error("invalid envelope")]
    InvalidEnvelope,

    #[error("unknown target prefix")]
    UnknownTargetPrefix,
}

/// The router bridge binds to two endpoints and polls both. It inspects the target identity’s prefix
/// to decide on which socket to forward the message.
pub fn run_router_bridge(
    ctx: zmq::Context,
    lease_id: i64,
    token: CancellationToken,
) -> Result<ThreadExecutionHandle> {
    // Bind a ROUTER for inproc dealers.
    let router_inproc = ctx.socket(zmq::ROUTER)?;
    router_inproc.bind(&inproc_endpoint())?;

    // Bind a ROUTER for ipc dealers.
    let router_ipc = ctx.socket(zmq::ROUTER)?;
    router_ipc.bind(&ipc_endpoint(lease_id))?;

    let child_token = token.child_token();
    let child_token_clone = child_token.clone();

    let handle = std::thread::spawn(move || {
        match router_bridge_thread(router_inproc, router_ipc, child_token_clone) {
            Ok(()) => (),
            Err(e) => {
                log::error!("[Router Bridge] Error: {}", e);
                token.cancel();
            }
        }
    });

    Ok(ThreadExecutionHandle {
        token: child_token,
        handle,
    })
}

fn router_bridge_thread(
    router_inproc: zmq::Socket,
    router_ipc: zmq::Socket,
    token: CancellationToken,
) -> Result<(), RouterBridgeError> {
    println!("[Router Bridge] Running...");

    let mut poll_items = [
        router_inproc.as_poll_item(zmq::POLLIN),
        router_ipc.as_poll_item(zmq::POLLIN),
    ];

    loop {
        // Poll both sockets with a 100ms timeout.
        let poll_result = zmq::poll(&mut poll_items, 100).unwrap();

        if token.is_cancelled() {
            log::info!("[Router Bridge] Lease cancelled, shutting down...");
            break;
        }

        fn process_messages(
            source: &zmq::Socket,
            target_inproc: &zmq::Socket,
            target_ipc: &zmq::Socket,
            side: &str,
        ) -> Result<(), RouterBridgeError> {
            let msg = source.recv_multipart(0).unwrap();
            if msg.len() != 6 {
                return Err(RouterBridgeError::InvalidEnvelope);
            }
            let sender = &msg[0]; // the router adds this to the envelope - id of sender
            let remote = &msg[1]; // b"" if local
            let target = &msg[2]; // identity of dealer
            let action = &msg[3]; // action to take on dealer
            let stream = &msg[4]; // stream_id of task on dealer
            let payload = &msg[5]; // payload

            println!(
                "[Router Bridge] {side} side: {} -> {} : {} : {}",
                String::from_utf8_lossy(sender),
                String::from_utf8_lossy(target),
                String::from_utf8_lossy(action),
                String::from_utf8_lossy(stream),
            );

            // Forward based on target's prefix.
            if target.starts_with(b"inproc-") {
                println!("sending to inproc");
                target_inproc
                    .send_multipart(&[target, remote, sender, action, stream, payload], 0)
                    .map_err(RouterBridgeError::SendError)
            } else if target.starts_with(b"ipc-") {
                println!("sending to ipc");
                target_ipc
                    .send_multipart(&[target, remote, sender, action, stream, payload], 0)
                    .map_err(RouterBridgeError::SendError)
            } else {
                Err(RouterBridgeError::UnknownTargetPrefix)
            }
        }

        // Process messages from the inproc side.
        if poll_items[0].is_readable() {
            println!("inproc side is readable");
            process_messages(&router_inproc, &router_inproc, &router_ipc, "inproc")?;
        }

        // Process messages from the ipc side.
        if poll_items[1].is_readable() {
            println!("ipc side is readable");
            process_messages(&router_ipc, &router_inproc, &router_ipc, "ipc")?;
        }
    }

    Ok(())
}

// fn send_with_envelope(
//     socket: &zmq::Socket,
//     target: &[u8],
//     sender: &[u8],
//     stream: &[u8],
//     action: &[u8],
//     payload: &[u8],
// ) {
//     let envelope = vec![target, sender, stream, action, payload];

//     match socket.send_multipart(&envelope, 0) {
//         Ok(()) => {
//             // Successfully sent.
//         }
//         Err(e) => {
//             if e == zmq::Error::EAGAIN {
//                 // The send would block.
//                 eprintln!("Socket send would block. Consider queuing the message for a retry.");
//                 // TODO: Push envelope to a retry queue and process later.
//             } else {
//                 eprintln!("Send failed with error: {}", e);
//                 // Send a cancellation signal back to the sender if needed:
//                 send_cancel_signal(socket, target, sender, stream_id);
//             }
//         }
//     }
// }
