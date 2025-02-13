#!/bin/bash


set -ex

cd rust && cargo build --release

cd python-wheel && maturin build

pip uninstall triton_distributed_rs
pip install target/wheels/triton_distributed_rs-0.1.3-cp312-cp312-manylinux_2_34_x86_64.whl --force-reinstall