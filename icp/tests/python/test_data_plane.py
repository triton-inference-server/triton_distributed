# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import multiprocessing
import uuid
from multiprocessing import Process, Queue
from typing import Sequence

import cupy
import numpy
import pytest
import ucp
from cupy_backends.cuda.api.runtime import CUDARuntimeError
from tdist.icp.data_plane import DataPlaneError
from tdist.icp.ucp_data_plane import (
    UcpDataPlane,
    get_icp_tensor_uri,
    set_icp_tensor_uri,
)
from tritonserver import DataType, MemoryType, Tensor
from tritonserver._api._datautils import TRITON_TO_NUMPY_DTYPE


def _cuda_available():
    # Note: cuda.is_avalailable initializes cuda
    #       and can't be called when forking subprocesses
    try:
        return cupy.cuda.is_available()
    except CUDARuntimeError:
        return False


def data_plane_reader(
    input_tensor_queue: Queue,
    tensor_descriptor_queue: Queue,
    output_tensor_queue: Queue,
    memory_type: MemoryType,
    memory_type_id: int,
):
    ucp.reset()
    data_plane = UcpDataPlane()
    data_plane.connect()
    output_tensor = None
    get_error = None
    release_error = None
    while True:
        input_tensor = tensor_descriptor_queue.get()
        if input_tensor is None:
            break
        try:
            output_tensor = data_plane.get_tensor(
                input_tensor,
                requested_memory_type=memory_type,
                requested_memory_type_id=memory_type_id,
            )
        except DataPlaneError as e:
            get_error = e

        if not get_error:
            if output_tensor.data_type == DataType.BYTES:
                output_tensor = output_tensor.to_bytes_array()
            else:
                if memory_type == MemoryType.GPU and _cuda_available():
                    output_tensor = cupy.from_dlpack(output_tensor)
                else:
                    output_tensor = numpy.from_dlpack(output_tensor)
        try:
            data_plane.release_tensor(input_tensor)
        except DataPlaneError as e:
            release_error = e

        if get_error:
            output_tensor_queue.put((get_error, release_error))
        else:
            output_tensor_queue.put((output_tensor, release_error))

    output_tensor_queue.put((None, None))
    data_plane.close()


def data_plane_writer(
    input_tensor_queue: Queue,
    tensor_descriptor_queue: Queue,
    output_tensor_queue: Queue,
    memory_type: MemoryType,
    memory_type_id: int,
    use_invalid_descriptor: bool = False,
    timeout=30,
    use_tensor_contents: bool = False,
):
    ucp.reset()
    data_plane = UcpDataPlane()
    data_plane.connect()
    while True:
        input_tensor = input_tensor_queue.get()
        if input_tensor is None:
            tensor_descriptor_queue.put(None)
            break

        input_tensor = Tensor._from_object(input_tensor)

        input_tensor_descriptor = data_plane.put_input_tensor(
            input_tensor, use_tensor_contents=use_tensor_contents
        )

        if use_invalid_descriptor and not use_tensor_contents:
            tensor_uri = get_icp_tensor_uri(input_tensor_descriptor)
            invalid_tensor_id = str(uuid.uuid1())
            tensor_uri = tensor_uri[: -len(invalid_tensor_id)]
            tensor_uri = tensor_uri + invalid_tensor_id
            set_icp_tensor_uri(input_tensor_descriptor, tensor_uri)

        tensor_descriptor_queue.put(input_tensor_descriptor)

    if not use_invalid_descriptor and timeout:
        data_plane.close(wait_for_release=timeout)

    data_plane.close()


@pytest.fixture
def tensors():
    tensors = [
        numpy.random.randint(0, 10, size=(2, 3)),
        numpy.random.randint(0, 10, size=(100)),
        numpy.random.randint(2, size=(1), dtype=bool),
    ]
    return tensors


@pytest.mark.timeout(60, method="thread")
def test_data_plane_error_invalid_tensor_uri(request):
    input_tensor_queue = Queue()
    tensor_descriptor_queue = Queue()
    output_tensor_queue = Queue()
    input_tensors = []
    memory_type = MemoryType.CPU
    memory_type_id = 0

    reader = Process(
        target=data_plane_reader,
        args=(
            input_tensor_queue,
            tensor_descriptor_queue,
            output_tensor_queue,
            memory_type,
            memory_type_id,
        ),
    )
    writer = Process(
        target=data_plane_writer,
        args=(
            input_tensor_queue,
            tensor_descriptor_queue,
            output_tensor_queue,
            memory_type,
            memory_type_id,
            True,
            30,
        ),
    )

    reader.start()
    writer.start()

    tensors = request.getfixturevalue("tensors")

    for tensor in tensors:
        input_tensors.append(Tensor.from_dlpack(tensor))

    for input_tensor in input_tensors:
        if input_tensor.memory_type == MemoryType.CPU or not _cuda_available():
            input_tensor_queue.put(numpy.from_dlpack(input_tensor))
        else:
            input_tensor_queue.put(cupy.from_dlpack(input_tensor))

    input_tensor_queue.put(None)

    reader.join()
    writer.join()

    while True:
        output_tensor, release_error = output_tensor_queue.get()
        if output_tensor is None:
            break
        assert isinstance(output_tensor, DataPlaneError)
        assert isinstance(release_error, DataPlaneError)


@pytest.mark.timeout(30, method="thread")
@pytest.mark.parametrize(
    "memory_type,memory_type_id", [(MemoryType.CPU, 0), (MemoryType.GPU, 0)]
)
def test_requested_memory_type(memory_type, memory_type_id, request):
    ctx = multiprocessing.get_context("spawn")
    input_tensor_queue = ctx.Queue()
    tensor_descriptor_queue = ctx.Queue()
    output_tensor_queue = ctx.Queue()
    input_tensors = []
    output_tensors = []

    reader = ctx.Process(
        target=data_plane_reader,
        args=(
            input_tensor_queue,
            tensor_descriptor_queue,
            output_tensor_queue,
            memory_type,
            memory_type_id,
        ),
    )
    writer = ctx.Process(
        target=data_plane_writer,
        args=(
            input_tensor_queue,
            tensor_descriptor_queue,
            output_tensor_queue,
            memory_type,
            memory_type_id,
            False,
            30,
        ),
    )

    reader.start()
    writer.start()

    tensors = request.getfixturevalue("tensors")

    for tensor in tensors:
        input_tensors.append(Tensor.from_dlpack(tensor))

    for input_tensor in input_tensors:
        if input_tensor.memory_type == MemoryType.CPU or not _cuda_available():
            input_tensor_queue.put(numpy.from_dlpack(input_tensor))
        else:
            input_tensor_queue.put(cupy.from_dlpack(input_tensor))

    input_tensor_queue.put(None)

    reader.join()
    writer.join()

    while True:
        output_tensor, release_error = output_tensor_queue.get()
        if output_tensor is None:
            break

        assert not isinstance(output_tensor, DataPlaneError)

        output_tensors.append(Tensor.from_dlpack(output_tensor))

    for input_tensor, output_tensor in zip(input_tensors, output_tensors):
        expected_memory_type = memory_type
        if not _cuda_available():
            expected_memory_type = MemoryType.CPU

        assert output_tensor.memory_type == expected_memory_type
        assert output_tensor.memory_type_id == memory_type_id

        input_comparison = numpy.from_dlpack(input_tensor.to_host())
        output_comparison = numpy.from_dlpack(output_tensor.to_host())
        numpy.testing.assert_equal(input_comparison, output_comparison)
        print(input_tensor, output_tensor)


def _get_random_tensor(data_type: DataType, size: Sequence[int]):
    dtype = TRITON_TO_NUMPY_DTYPE[data_type]
    value = numpy.random.rand(*size)
    return value.astype(dtype)


@pytest.mark.timeout(30, method="thread")
@pytest.mark.parametrize(
    "data_type",
    [
        data_type
        for data_type in DataType.__members__.values()
        if data_type not in [DataType.INVALID, DataType.BF16]
    ],
    ids=[
        data_type
        for data_type in DataType.__members__.keys()
        if data_type not in ["INVALID", "BF16"]
    ],
)
def test_tensor_types(request, data_type):
    input_tensor_queue = Queue()
    tensor_descriptor_queue = Queue()
    output_tensor_queue = Queue()
    input_tensors = []
    output_tensors = []
    memory_type = MemoryType.CPU
    memory_type_id = 0

    reader = Process(
        target=data_plane_reader,
        args=(
            input_tensor_queue,
            tensor_descriptor_queue,
            output_tensor_queue,
            memory_type,
            memory_type_id,
        ),
    )
    writer = Process(
        target=data_plane_writer,
        args=(
            input_tensor_queue,
            tensor_descriptor_queue,
            output_tensor_queue,
            memory_type,
            memory_type_id,
            False,
            30,
        ),
    )

    reader.start()
    writer.start()

    tensors = []

    tensors.append(_get_random_tensor(data_type, [1, 4]))

    for tensor in tensors:
        if data_type == DataType.BYTES:
            input_tensors.append(Tensor._from_object(tensor))
        else:
            input_tensors.append(Tensor.from_dlpack(tensor))

    for input_tensor in input_tensors:
        if input_tensor.data_type != DataType.BYTES:
            input_tensor_queue.put(numpy.from_dlpack(input_tensor))
        else:
            input_tensor_queue.put(input_tensor.to_bytes_array())
    input_tensor_queue.put(None)

    reader.join()
    writer.join()

    while True:
        output_tensor, release_error = output_tensor_queue.get()
        if output_tensor is None:
            break

        assert not isinstance(output_tensor, DataPlaneError)

        output_tensors.append(Tensor._from_object(output_tensor))

    for input_tensor, output_tensor in zip(input_tensors, output_tensors):
        expected_memory_type = memory_type
        if not _cuda_available():
            expected_memory_type = MemoryType.CPU

        assert output_tensor.memory_type == expected_memory_type
        assert output_tensor.memory_type_id == memory_type_id

        if input_tensor.data_type == DataType.BYTES:
            input_comparison = input_tensor.to_bytes_array()
            output_comparison = output_tensor.to_bytes_array()
        else:
            input_comparison = numpy.from_dlpack(input_tensor.to_host())
            output_comparison = numpy.from_dlpack(output_tensor.to_host())
        numpy.testing.assert_equal(input_comparison, output_comparison)
        print(input_tensor, output_tensor)


@pytest.mark.timeout(30, method="thread")
@pytest.mark.parametrize(
    "data_type",
    [
        data_type
        for data_type in DataType.__members__.values()
        if data_type not in [DataType.INVALID, DataType.BF16]
    ],
    ids=[
        data_type
        for data_type in DataType.__members__.keys()
        if data_type not in ["INVALID", "BF16"]
    ],
)
def test_use_tensor_contents(request, data_type):
    input_tensor_queue = Queue()
    tensor_descriptor_queue = Queue()
    output_tensor_queue = Queue()
    input_tensors = []
    output_tensors = []
    memory_type = MemoryType.CPU
    memory_type_id = 0

    reader = Process(
        target=data_plane_reader,
        args=(
            input_tensor_queue,
            tensor_descriptor_queue,
            output_tensor_queue,
            memory_type,
            memory_type_id,
        ),
    )
    writer = Process(
        target=data_plane_writer,
        args=(
            input_tensor_queue,
            tensor_descriptor_queue,
            output_tensor_queue,
            memory_type,
            memory_type_id,
            True,
            30,
            True,
        ),
    )

    reader.start()
    writer.start()

    tensors = []
    tensors.append(_get_random_tensor(data_type, [2, 4]))

    for tensor in tensors:
        if data_type == DataType.BYTES:
            input_tensors.append(Tensor._from_object(tensor))
        else:
            input_tensors.append(Tensor.from_dlpack(tensor))

    for input_tensor in input_tensors:
        if input_tensor.data_type != DataType.BYTES:
            input_tensor_queue.put(numpy.from_dlpack(input_tensor))
        else:
            input_tensor_queue.put(input_tensor.to_bytes_array())
    input_tensor_queue.put(None)

    reader.join()
    writer.join()

    while True:
        output_tensor, release_error = output_tensor_queue.get()
        if output_tensor is None:
            break

        assert not isinstance(output_tensor, DataPlaneError)

        output_tensors.append(Tensor._from_object(output_tensor))

    for input_tensor, output_tensor in zip(input_tensors, output_tensors):
        expected_memory_type = memory_type
        if not _cuda_available():
            expected_memory_type = MemoryType.CPU

        assert output_tensor.memory_type == expected_memory_type
        assert output_tensor.memory_type_id == memory_type_id

        if input_tensor.data_type == DataType.BYTES:
            input_comparison = input_tensor.to_bytes_array()
            output_comparison = output_tensor.to_bytes_array()
        else:
            input_comparison = numpy.from_dlpack(input_tensor.to_host())
            output_comparison = numpy.from_dlpack(output_tensor.to_host())
        numpy.testing.assert_equal(input_comparison, output_comparison)
        print(input_tensor, output_tensor)
