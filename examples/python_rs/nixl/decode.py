import argparse
from typing import List

import nixl_bindings as nixl
import torch
import zmq
import time
import pickle

GPU_ID = 1
AGENT_NAME = "AgentDecode"

_ctx = None
_socket = None

def init_zmq(hostname, port):
    global _ctx, _socket
    _ctx = zmq.Context()
    _socket = _ctx.socket(zmq.PAIR)
    _socket.connect(f"tcp://{hostname}:{port}")


def init_nixl():
    print("Initializing NIXL")
    name = AGENT_NAME # Should be unique per GPU
    devices = nixl.nixlDeviceMD()
    init = nixl.nixlUcxInitParams()

    agent = nixl.nixlAgent(name, devices)
    ucx = agent.createBackend(init)

    return agent, ucx

def allocate_memory() -> torch.Tensor:
    print("Allocating memory")
    device = torch.device(f"cuda:{GPU_ID}")
    tensors = [torch.zeros(10, device=device, dtype=torch.float32) for _ in range(2)]
    return tensors


def register_memory(agent, backend, tensors: List[torch.Tensor]):
    print("Registering memory")
    reg_list = nixl.nixlDescList(nixl.VRAM_SEG, True, False)
    for tensor in tensors:
        base_addr = tensor.data_ptr()
        region_len = tensor.numel() * tensor.element_size()
        reg_list.addDesc(nixl.nixlBasicDesc(base_addr, region_len, GPU_ID))
    ret = agent.registerMem(reg_list, backend)
    assert ret == 0

    return reg_list


def sync_agent_meta(agent):
    # Send and recv metedata over zmq sockets
    print("Sending metadata to prefill")
    decode_meta = agent.getLocalMD()
    _socket.send(decode_meta)
    print(f"Sent metadata to prefill: {decode_meta}")

    # Recv metadata from prefill
    print("Receiving metadata from prefill")
    prefill_meta = _socket.recv()
    print(f"Received metadata from prefill: {prefill_meta}")

    # Load metadata from prefill
    print("Loading metadata from prefill")
    ret = agent.loadRemoteMD(prefill_meta)
    print(f"Loaded metadata from prefill: {ret}")
    return ret

def send_mem_desc(reg_list):
    print("Sending memory description")
    mem_desc = pickle.dumps(reg_list)
    _socket.send(mem_desc)
    print(f"Sent memory description: {mem_desc}")

def wait_for_notif(tensors,agent):
    print("Waiting for notification")
    notifMap = {}
    while len(notifMap) == 0:
        notifMap = agent.getNotifs(notifMap)
    print(f"Notif received: {notifMap}")
    print(f"Tensors: {tensors}")


def release_memory(agent, backend, reg_list):
    print("Releasing memory")
    ret = agent.deregisterMem(reg_list, backend)
    assert ret == 0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill-host", type=str, default="localhost")
    parser.add_argument("--prefill-port", type=int, default=5432)
    return parser.parse_args()

def main_decode(args):
    init_zmq(args.prefill_host, args.prefill_port)
    agent, backend = init_nixl()
    tensors = allocate_memory()
    print(f"Tensors: {tensors}")
    reg_list = register_memory(agent, backend, tensors)

    prefill_agent = sync_agent_meta(agent)
    send_mem_desc(reg_list)
    wait_for_notif(tensors, agent)
    release_memory(agent, backend, reg_list)


if __name__ == "__main__":
    args = parse_args()
    main_decode(args)

