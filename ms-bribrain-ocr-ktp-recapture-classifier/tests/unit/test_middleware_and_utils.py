"""Tests for middleware and device utilities."""

import sys
import pytest
from unittest.mock import MagicMock, patch
from fastapi import Request, Response
from starlette.datastructures import Headers


class TestRequestIdMiddleware:
    """Test cases for RequestIdMiddleware."""

    @pytest.mark.asyncio
    async def test_request_id_added_to_response(self):
        """Test that request ID is added to response headers."""
        from src.middleware.add_requestid import RequestIdMiddleware

        app = MagicMock()
        middleware = RequestIdMiddleware(app)

        request = MagicMock(spec=Request)
        request.headers = Headers({})

        async def call_next(request):
            return Response(content="test", status_code=200)

        response = await middleware.dispatch(request, call_next)

        assert "x-request-id" in response.headers
        assert len(response.headers["x-request-id"]) > 0

    @pytest.mark.asyncio
    async def test_request_id_from_existing_header(self):
        """Test that existing request ID from header is used."""
        from src.middleware.add_requestid import RequestIdMiddleware

        app = MagicMock()
        middleware = RequestIdMiddleware(app)

        existing_id = "existing-request-id-12345"
        request = MagicMock(spec=Request)
        request.headers = Headers({"x-request-id": existing_id})

        async def call_next(request):
            return Response(content="test", status_code=200)

        response = await middleware.dispatch(request, call_next)

        assert response.headers["x-request-id"] == existing_id

    @pytest.mark.asyncio
    async def test_request_id_context_reset_when_call_next_raises(self):
        """The `finally: request_id_ctx.reset(token)` branch must execute on exception."""
        from src.middleware.add_requestid import RequestIdMiddleware
        from src.core.logging import request_id_ctx

        app = MagicMock()
        middleware = RequestIdMiddleware(app)

        request = MagicMock(spec=Request)
        request.headers = Headers({"x-request-id": "req-xyz"})

        async def call_next(request):
            raise RuntimeError("downstream failure")

        # Capture the context value BEFORE dispatch so we can assert it isn't leaked afterwards.
        sentinel = request_id_ctx.get()

        with pytest.raises(RuntimeError, match="downstream failure"):
            await middleware.dispatch(request, call_next)

        # After dispatch, context should be restored to its pre-dispatch value.
        assert request_id_ctx.get() == sentinel

    @pytest.mark.asyncio
    async def test_request_id_context_var_set(self):
        """Test that request ID is set in context variable."""
        from src.middleware.add_requestid import RequestIdMiddleware
        from src.core.logging import request_id_ctx

        app = MagicMock()
        middleware = RequestIdMiddleware(app)

        request = MagicMock(spec=Request)
        request.headers = Headers({})

        captured_request_id = None

        async def call_next(request):
            nonlocal captured_request_id
            captured_request_id = request_id_ctx.get()
            return Response(content="test", status_code=200)

        await middleware.dispatch(request, call_next)

        assert captured_request_id is not None
        assert len(captured_request_id) > 0


class TestDeviceUtils:
    """Test cases for device selection utilities."""

    def test_get_device_returns_device(self):
        """Test that get_device returns a valid device."""
        from src.core.device import get_device
        import torch

        device = get_device()

        assert device is not None
        assert isinstance(device, torch.device)

    def test_get_device_force_cpu(self):
        """Test device selection when CPU is forced."""
        from src.core.device import get_device

        device = get_device(force_cpu=True)

        assert str(device) == "cpu"

    def test_get_device_default(self):
        """Test device selection with default settings."""
        from src.core.device import get_device

        device = get_device()

        assert str(device) in ["cpu", "cuda:0", "cuda"]

    def test_get_device_cuda_path_logs_gpu_info(self):
        """If CUDA is reported available, the GPU info log branch runs end-to-end."""
        from src.core import device as device_mod
        import torch

        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "get_device_name", return_value="FakeGPU"), \
             patch.object(torch.cuda, "get_device_properties", return_value=MagicMock(total_memory=8 * 1024**3)), \
             patch.object(torch.version, "cuda", "12.1", create=True):
            dev = device_mod.get_device()
        assert dev.type == "cuda"

    def test_get_device_info_cpu(self):
        from src.core.device import get_device_info
        import torch

        with patch.object(torch.cuda, "is_available", return_value=False):
            info = get_device_info()
        assert info["device_type"] == "cpu"
        assert info["cuda_available"] is False

    def test_get_device_info_cuda_multi_gpu(self):
        """Multi-GPU host should still populate all keys — device 0 values used."""
        from src.core import device as device_mod
        import torch

        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "get_device_name", return_value="GPU0"), \
             patch.object(torch.cuda, "get_device_properties", return_value=MagicMock(total_memory=16 * 1024**3)), \
             patch.object(torch.cuda, "device_count", return_value=4), \
             patch.object(torch.version, "cuda", "12.1", create=True):
            info = device_mod.get_device_info()

        assert info["device_type"] == "cuda"
        assert info["gpu_name"] == "GPU0"
        assert info["gpu_memory_gb"] == 16.0
        assert info["device_count"] == 4


class TestModelLoading:
    """Test cases for ML model loading utilities (torchvision mocked)."""

    @pytest.fixture(autouse=True)
    def _mock_torchvision(self):
        """Mock torchvision at module level for all tests in this class."""
        import torch
        import torch.nn as nn

        # Create a real nn.Module mock for resnet50
        mock_resnet = MagicMock(spec=nn.Module)
        mock_resnet.fc = nn.Linear(2048, 1000)
        mock_resnet.load_state_dict = MagicMock()
        mock_resnet.to = MagicMock(return_value=mock_resnet)
        mock_resnet.eval = MagicMock(return_value=mock_resnet)

        mock_models_mod = MagicMock()
        mock_models_mod.resnet50.return_value = mock_resnet

        mock_transforms_mod = MagicMock()
        mock_compose_instance = MagicMock()
        mock_transforms_mod.Compose.return_value = mock_compose_instance
        mock_transforms_mod.CenterCrop = MagicMock()
        mock_transforms_mod.ToTensor = MagicMock()
        mock_transforms_mod.Normalize = MagicMock()

        mock_tv = MagicMock()
        mock_tv.models = mock_models_mod
        mock_tv.transforms = mock_transforms_mod

        self._resnet = mock_resnet
        self._transforms_mod = mock_transforms_mod

        # Patch sys.modules and clear cached src.models
        modules_to_clean = [k for k in sys.modules if k.startswith('src.models')]
        saved = {k: sys.modules.pop(k) for k in modules_to_clean}

        with patch.dict(sys.modules, {
            'torchvision': mock_tv,
            'torchvision.models': mock_models_mod,
            'torchvision.transforms': mock_transforms_mod,
        }):
            yield

        # Restore cleaned modules
        for k in [k for k in sys.modules if k.startswith('src.models')]:
            del sys.modules[k]
        sys.modules.update(saved)

    @patch('pathlib.Path.exists', return_value=True)
    @patch('torch.load', return_value={})
    def test_load_model_success(self, mock_torch_load, mock_exists):
        """Test successful model loading."""
        import torch
        from src.models.ml_model import load_model

        device = torch.device("cpu")
        model = load_model("./models/test.pth", device, num_classes=2)

        assert model is not None
        self._resnet.load_state_dict.assert_called_once()

    @patch('pathlib.Path.exists', return_value=False)
    def test_load_model_file_not_found(self, mock_exists):
        """Test model loading when file doesn't exist."""
        import torch
        from src.models.ml_model import load_model

        device = torch.device("cpu")

        with pytest.raises(FileNotFoundError):
            load_model("./models/nonexistent.pth", device, num_classes=2)

    def test_get_transform(self):
        """Test transform creation."""
        from src.models.ml_model import get_transform

        transform = get_transform(crop_size=224)
        assert transform is not None
        assert callable(transform)

    def test_get_transform_default_crop_size(self):
        """Test transform with default crop size."""
        from src.models.ml_model import get_transform

        transform = get_transform()
        assert transform is not None

    @patch('pathlib.Path.exists', return_value=True)
    @patch('torch.load', return_value={})
    def test_load_model_cuda_compile_path(self, mock_torch_load, mock_exists):
        """When use_compile=True and device is cuda, torch.compile is invoked."""
        import torch
        from src.core.config import config
        from src.models.ml_model import load_model

        real_get = config.get

        def patched(key, default=None):
            if key == "model.use_compile":
                return True
            if key == "model.compile_mode":
                return "reduce-overhead"
            return real_get(key, default)

        compiled_marker = MagicMock(name="compiled")
        with patch.object(config, "get", side_effect=patched), \
             patch("torch.compile", return_value=compiled_marker) as mock_compile:
            device = torch.device("cuda")
            model = load_model("./models/test.pth", device, num_classes=2)
            mock_compile.assert_called_once()
            assert model is compiled_marker

    @patch('pathlib.Path.exists', return_value=True)
    @patch('torch.load', return_value={})
    def test_load_model_cpu_jit_path(self, mock_torch_load, mock_exists):
        """CPU path should call torch.jit.trace and optimize_for_inference."""
        import torch
        from src.core.config import config
        from src.models.ml_model import load_model

        real_get = config.get

        def patched(key, default=None):
            if key == "model.use_compile":
                return True
            if key == "model.crop_size":
                return 224
            return real_get(key, default)

        traced = MagicMock(name="traced")
        optimized = MagicMock(name="optimized")
        with patch.object(config, "get", side_effect=patched), \
             patch("torch.jit.trace", return_value=traced) as mock_trace, \
             patch("torch.jit.optimize_for_inference", return_value=optimized) as mock_opt:
            device = torch.device("cpu")
            model = load_model("./models/test.pth", device, num_classes=2)
            mock_trace.assert_called_once()
            mock_opt.assert_called_once_with(traced)
            assert model is optimized

    @patch('pathlib.Path.exists', return_value=True)
    @patch('torch.load', return_value={})
    def test_load_model_optimization_failure_falls_back_eager(self, mock_torch_load, mock_exists):
        """If optimization raises, load_model logs a warning and returns the eager model."""
        import torch
        from src.core.config import config
        from src.models.ml_model import load_model

        real_get = config.get

        def patched(key, default=None):
            if key == "model.use_compile":
                return True
            if key == "model.crop_size":
                return 224
            return real_get(key, default)

        with patch.object(config, "get", side_effect=patched), \
             patch("torch.jit.trace", side_effect=RuntimeError("trace failed")):
            device = torch.device("cpu")
            model = load_model("./models/test.pth", device, num_classes=2)
            # Should still return a model (eager fallback), not raise
            assert model is not None

    @patch('pathlib.Path.exists', return_value=True)
    @patch('torch.load', side_effect=RuntimeError("corrupt checkpoint"))
    def test_load_model_wraps_exception_as_runtime_error(self, mock_torch_load, mock_exists):
        """Any failure inside the try block is wrapped as RuntimeError with prefix."""
        import torch
        from src.models.ml_model import load_model

        with pytest.raises(RuntimeError, match="Model loading failed"):
            load_model("./models/test.pth", torch.device("cpu"), num_classes=2)


class TestMainApp:
    """Test cases for main application functions."""

    @pytest.fixture(autouse=True)
    def _mock_torchvision(self):
        """Mock torchvision to prevent import errors."""
        mock_tv = MagicMock()

        modules_to_clean = [k for k in sys.modules
                            if k.startswith('src.models') or k == 'src.main']
        saved = {k: sys.modules.pop(k) for k in modules_to_clean}

        with patch.dict(sys.modules, {
            'torchvision': mock_tv,
            'torchvision.models': mock_tv.models,
            'torchvision.transforms': mock_tv.transforms,
        }):
            yield

        for k in [k for k in sys.modules
                   if k.startswith('src.models') or k == 'src.main']:
            del sys.modules[k]
        sys.modules.update(saved)

    def test_configure_cpu_threading(self):
        """Test CPU threading configuration."""
        from src.main import _configure_cpu_threading
        import os

        _configure_cpu_threading()

        assert "OMP_NUM_THREADS" in os.environ
        assert "MKL_NUM_THREADS" in os.environ

    def test_warmup_model(self):
        """Test model warmup function."""
        import torch
        from src.main import _warmup_model

        mock_model = MagicMock()
        mock_model.return_value = torch.zeros((1, 2))
        mock_device = torch.device("cpu")

        def mock_transform_fn(img):
            return torch.zeros((3, 224, 224))

        _warmup_model(mock_model, mock_device, mock_transform_fn)

    def test_warmup_model_handles_exceptions(self):
        """Test that warmup model handles exceptions gracefully."""
        import torch
        from src.main import _warmup_model

        mock_model = MagicMock()
        mock_device = torch.device("cpu")

        def failing_transform(img):
            raise Exception("Transform error")

        _warmup_model(mock_model, mock_device, failing_transform)


class TestLifespan:
    """Tests for FastAPI lifespan startup/shutdown behavior.

    Kept outside TestMainApp because that class mocks torchvision in
    sys.modules which destabilizes other torch imports when src.main is
    re-imported here.
    """

    def test_lifespan_exits_on_startup_failure(self):
        """Any exception inside lifespan startup must call sys.exit(1)."""
        import asyncio
        import src.main as main_mod
        from fastapi import FastAPI

        app = FastAPI()

        with patch("src.core.validation.validate_startup_configuration"), \
             patch.object(main_mod, "get_device", side_effect=RuntimeError("device boom")), \
             patch.object(main_mod.sys, "exit", side_effect=SystemExit(1)) as mock_exit:

            async def run():
                async with main_mod.lifespan(app):
                    pass

            with pytest.raises(SystemExit):
                asyncio.get_event_loop().run_until_complete(run())
            mock_exit.assert_called_with(1)

    def test_warmup_model_cuda_cleanup_path(self):
        """When device.type == 'cuda', warmup synchronizes and empties cache."""
        import torch
        from src.main import _warmup_model

        fake_device = MagicMock()
        fake_device.type = "cuda"

        mock_model = MagicMock()
        mock_model.return_value = torch.zeros((1, 2))

        # Use a MagicMock for the tensor chain so we don't need a real CUDA runtime.
        fake_input = MagicMock()
        fake_input.unsqueeze.return_value.to.return_value = torch.zeros((1, 3, 224, 224))

        def transform_fn(img):
            return fake_input

        with patch.object(torch.cuda, "synchronize") as mock_sync, \
             patch.object(torch.cuda, "empty_cache") as mock_empty:
            _warmup_model(mock_model, fake_device, transform_fn)

        mock_sync.assert_called_once()
        mock_empty.assert_called_once()

    def test_lifespan_cloud_downloads_from_gcs(self):
        """When APP_ENVIRO != 'onprem', lifespan should call download_model_gcs."""
        import asyncio
        import torch
        import src.main as main_mod
        from fastapi import FastAPI

        app = FastAPI()

        async def fake_init():
            return None

        async def fake_dispose():
            return None

        with patch.object(main_mod, "APP_ENVIRO", "cloud"), \
             patch("src.core.validation.validate_startup_configuration"), \
             patch.object(main_mod, "get_device", return_value=torch.device("cpu")), \
             patch.object(main_mod, "download_model_minio") as mock_minio, \
             patch.object(main_mod, "download_model_gcs") as mock_gcs, \
             patch.object(main_mod, "load_model", return_value=MagicMock()), \
             patch.object(main_mod, "get_transform", return_value=lambda x: torch.zeros((3, 224, 224))), \
             patch.object(main_mod, "set_model_globals"), \
             patch.object(main_mod, "_warmup_model"), \
             patch.object(main_mod, "init_engine", side_effect=fake_init), \
             patch.object(main_mod, "dispose_engine", side_effect=fake_dispose):

            async def run():
                async with main_mod.lifespan(app):
                    pass

            asyncio.get_event_loop().run_until_complete(run())

        mock_gcs.assert_called_once()
        mock_minio.assert_not_called()

    def test_lifespan_disposes_engine_on_shutdown(self):
        """dispose_engine must run when the lifespan context exits cleanly."""
        import asyncio
        import torch
        import src.main as main_mod
        from fastapi import FastAPI

        app = FastAPI()
        disposed = {"called": False}

        async def fake_dispose():
            disposed["called"] = True

        async def fake_init():
            return None

        with patch("src.core.validation.validate_startup_configuration"), \
             patch.object(main_mod, "get_device", return_value=torch.device("cpu")), \
             patch.object(main_mod, "download_model_minio"), \
             patch.object(main_mod, "download_model_gcs"), \
             patch.object(main_mod, "load_model", return_value=MagicMock()), \
             patch.object(main_mod, "get_transform", return_value=lambda x: torch.zeros((3, 224, 224))), \
             patch.object(main_mod, "set_model_globals"), \
             patch.object(main_mod, "_warmup_model"), \
             patch.object(main_mod, "init_engine", side_effect=fake_init), \
             patch.object(main_mod, "dispose_engine", side_effect=fake_dispose):

            async def run():
                async with main_mod.lifespan(app):
                    pass

            asyncio.get_event_loop().run_until_complete(run())

        assert disposed["called"] is True

