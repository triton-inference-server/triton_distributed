import argparse
from typing import List
import pickle
import nixl_bindings as nixl
import torch
import zmq

GPU_ID = 0 
AGENT_NAME = "AgentPrefill"

_ctx = None
_socket = None


def init_zmq(hostname, port):
    global _ctx, _socket
    _ctx = zmq.Context()
    _socket = _ctx.socket(zmq.PAIR)
    _socket.bind(f"tcp://{hostname}:{port}")


def init_nixl():
    print("Initializing NIXL")
    name = AGENT_NAME  # Should be unique per GPU
    devices = nixl.nixlDeviceMD()
    init = nixl.nixlUcxInitParams()

    agent = nixl.nixlAgent(name, devices)
    ucx = agent.createBackend(init)

    return agent, ucx


def allocate_memory() -> torch.Tensor:
    print("Allocating memory")
    device = torch.device(f"cuda:{GPU_ID}")
    tensors = [torch.arange(10, device=device, dtype=torch.float32) for _ in range(2)]
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
    # Recv and send metadata over zmq sockets
    print("Receiving metadata from decode")
    decode_meta = _socket.recv()
    print(f"Received metadata from decode: {decode_meta}")

    # Send metadata to decode
    print("Sending metadata to decode")
    prefill_meta = agent.getLocalMD()
    _socket.send(prefill_meta)
    print(f"Sent metadata to decode: {prefill_meta}")

    # Load metadata from decode
    print("Loading metadata from decode")
    ret = agent.loadRemoteMD(decode_meta)
    print(f"Loaded metadata from decode: {ret}")
    return ret


def recv_mem_desc():
    print("Receiving memory description")
    mem_desc = _socket.recv()
    mem_desc = pickle.loads(mem_desc)
    print(f"Received memory description: {mem_desc}")
    return mem_desc


def transfer_mem(prefill_agent, decode_agent, prefill_mem_desc, decode_mem_desc):
    print("Transferring memory")
    handle = prefill_agent.createXferReq(
        prefill_mem_desc, decode_mem_desc, decode_agent, "OK", nixl.NIXL_WR_NOTIF)
    assert handle != None
    
    ret = prefill_agent.postXferReq(handle)
    assert ret != nixl.NIXL_XFER_ERR, f"Transfer failed with error: {ret}"

    status = 0
    while status != nixl.NIXL_XFER_DONE:
        status = prefill_agent.getXferStatus(handle)
        assert status != nixl.NIXL_XFER_ERR, f"Transfer failed with error: {status}"

    print("Transferred memory")

    return handle


def release_memory(agent, backend, reg_list, decode_agent, handle):
    print("Releasing memory")
    agent.invalidateXferReq(handle)
    ret = agent.deregisterMem(reg_list, backend)
    assert ret == 0
    agent.invalidateRemoteMD(decode_agent)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill-host", type=str, default="localhost")
    parser.add_argument("--prefill-port", type=int, default=5432)
    return parser.parse_args()


def main_prefill(args):
    init_zmq(args.prefill_host, args.prefill_port)
    agent, backend = init_nixl()
    tensors = allocate_memory()
    reg_list = register_memory(agent, backend, tensors)

    decode_agent = sync_agent_meta(agent)
    decode_mem_desc = recv_mem_desc()
    handle = transfer_mem(agent, decode_agent, reg_list, decode_mem_desc)
    release_memory(agent, backend, reg_list, decode_agent, handle)


if __name__ == "__main__":
    args = parse_args()
    main_prefill(args)