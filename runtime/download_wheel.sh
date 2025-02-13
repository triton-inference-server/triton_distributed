#!/bin/bash


set -ex

cd /workspace/runtime/rust && cargo build --release

cd /workspace/runtime/rust/python-wheel && maturin build

pip uninstall triton_distributed_rs
pip install /workspace/runtime/rust/python-wheel/target/wheels/triton_distributed_rs-0.1.3-cp312-cp312-manylinux_2_34_x86_64.whl --force-reinstall