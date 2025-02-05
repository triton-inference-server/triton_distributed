# Python Bindings for Rust

This README is under construction and will have more details added over time.

```bash
cd triton_distributed/runtime/rust/python-wheel

# Build python package for bindings (requires cargo and maturin)
maturin build

# Install the python package for binded rust code
pip install target/wheels/triton_distributed_rs-*.whl

# Sanity check the package installed successfully
python3 -c "import triton_distributed_rs"
```
