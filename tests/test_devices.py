from types import SimpleNamespace
from unittest.mock import Mock
import sys

import pytest

from semif_phase1.core import (
    load_causal_model,
    resolve_device,
    synchronize,
    torch_accelerator_metadata,
)


@pytest.fixture
def torch_stub(monkeypatch):
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=Mock(return_value=False), device_count=Mock(return_value=0),
                             synchronize=Mock()),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=Mock(return_value=True))),
        mps=SimpleNamespace(synchronize=Mock()),
        device=lambda name: SimpleNamespace(type=name.split(":")[0], name=name),
        version=SimpleNamespace(cuda="12.8", hip=None),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def test_auto_uses_mps_without_cuda(torch_stub):
    assert resolve_device().name == "mps"


def test_cpu_must_be_selected_explicitly(torch_stub):
    torch_stub.backends.mps.is_available.return_value = False
    assert resolve_device("cpu").name == "cpu"
    with pytest.raises(ValueError, match="MPS is unavailable"):
        resolve_device()


def test_cuda_requires_exactly_one_gpu(torch_stub):
    torch_stub.cuda.is_available.return_value = True
    torch_stub.cuda.device_count.return_value = 1
    assert resolve_device().name == "cuda:0"
    torch_stub.cuda.device_count.return_value = 2
    with pytest.raises(ValueError, match="exactly one CUDA"):
        resolve_device()
    with pytest.raises(ValueError, match="exactly one CUDA"):
        resolve_device("cuda")


def test_rocm_uses_torch_cuda_namespace_and_visibility_guidance(torch_stub):
    torch_stub.version.cuda = None
    torch_stub.version.hip = "7.1"
    torch_stub.cuda.is_available.return_value = True
    torch_stub.cuda.device_count.return_value = 1
    assert resolve_device("cuda").name == "cuda:0"
    assert torch_accelerator_metadata(
        torch_stub, SimpleNamespace(type="cuda")
    ) == {
        "torch_accelerator": "rocm",
        "torch_accelerator_version": "7.1",
    }

    torch_stub.cuda.device_count.return_value = 2
    with pytest.raises(ValueError, match="ROCR_VISIBLE_DEVICES=0"):
        resolve_device("cuda")


@pytest.mark.parametrize(
    "device,cuda_version,hip_version,expected",
    [
        ("cuda", "12.8", None, {"torch_accelerator": "cuda", "torch_accelerator_version": "12.8"}),
        ("cuda", None, "7.1", {"torch_accelerator": "rocm", "torch_accelerator_version": "7.1"}),
        ("mps", None, None, {"torch_accelerator": "mps"}),
        ("cpu", None, None, {"torch_accelerator": "cpu"}),
    ],
)
def test_torch_accelerator_metadata_all_backends(torch_stub, device, cuda_version, hip_version, expected):
    torch_stub.version.cuda = cuda_version
    torch_stub.version.hip = hip_version
    assert torch_accelerator_metadata(torch_stub, SimpleNamespace(type=device)) == expected


def test_mps_availability_is_checked(torch_stub):
    torch_stub.backends.mps.is_available.return_value = False
    with pytest.raises(ValueError, match="MPS is unavailable"):
        resolve_device("mps")


@pytest.mark.parametrize("backend", ["cuda", "mps", "cpu"])
def test_synchronization_dispatch(torch_stub, backend):
    device = SimpleNamespace(type=backend)
    synchronize(device)
    if backend == "cuda":
        torch_stub.cuda.synchronize.assert_called_once_with(device)
    if backend == "mps":
        torch_stub.mps.synchronize.assert_called_once_with()


@pytest.mark.parametrize("device,dtype,target", [
    ("cuda", "bfloat16", "cuda:0"),
    ("mps", "float16", "mps"),
    ("cpu", "float32", "cpu"),
])
def test_loader_passes_device_dtype_and_revision(torch_stub, monkeypatch, device, dtype, target):
    model = Mock()
    factory = SimpleNamespace(from_pretrained=Mock(return_value=(model, {})))
    transformers = SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=Mock(return_value=SimpleNamespace(model_type="other"))),
        AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=object())),
        AutoModelForCausalLM=factory, __version__="test",
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    torch_stub.cuda.is_available.return_value = True
    torch_stub.cuda.device_count.return_value = 1
    torch_stub.device = lambda name: name
    torch_stub.bfloat16 = "bfloat16"
    torch_stub.float16 = "float16"
    torch_stub.float32 = "float32"
    torch_stub.__version__ = "test"
    revision = "a" * 40
    _, _, metadata = load_causal_model("test/model", revision, device, dtype)
    kwargs = factory.from_pretrained.call_args.kwargs
    assert kwargs["device_map"] == {"": target}
    assert kwargs["dtype"] == dtype
    assert kwargs["revision"] == revision
    assert kwargs["trust_remote_code"] is False
    assert metadata["device"] == target and metadata["dtype"] == dtype
    assert metadata["torch_accelerator"] == device
    if device == "cuda":
        assert metadata["torch_accelerator_version"] == "12.8"
    else:
        assert "torch_accelerator_version" not in metadata
    model.eval.assert_called_once()
