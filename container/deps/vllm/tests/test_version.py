import pytest

try:
    import vllm
except ImportError:
    vllm = None

pytestmark = pytest.mark.pre_merge


# TODO: Consider `pytest.mark.vllm` and running tests based on environment
@pytest.mark.skipif(vllm is None, reason="Skipping vllm tests, vllm not installed")
def test_version():
    # Verify that the image has the patched version of vllm
    assert vllm.__version__ == "0.6.3.post2.dev16+gf61960ce"
