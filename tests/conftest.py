import os
import pathlib
import sys
import pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def native():
    from native_components import load
    root = os.environ.get("COMFYUI_ROOT")
    if not root:
        pytest.skip("Set COMFYUI_ROOT to the audited ComfyUI checkout for native integration tests")
    return load(root)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    import torch
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)
