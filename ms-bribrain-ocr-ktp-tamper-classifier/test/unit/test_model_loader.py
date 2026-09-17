"""
Unit tests for model_loader service
"""

import os
import pytest
from unittest.mock import Mock, patch

from src.services.model_loader import (
    get_optimal_num_threads,
    configure_cpu_threading,
    load_model
)


@pytest.mark.unit
class TestGetOptimalNumThreads:
    """Tests for get_optimal_num_threads function"""
    
    def test_uses_omp_num_threads_env_var(self, monkeypatch):
        """Test that OMP_NUM_THREADS env var takes priority"""
        monkeypatch.setenv("OMP_NUM_THREADS", "8")
        
        result = get_optimal_num_threads()
        
        assert result == 8
    
    def test_invalid_omp_num_threads_falls_back(self, monkeypatch):
        """Test fallback when OMP_NUM_THREADS is invalid"""
        monkeypatch.setenv("OMP_NUM_THREADS", "invalid")
        
        result = get_optimal_num_threads()
        
        # Should fall back to os.cpu_count()
        assert isinstance(result, int)
        assert result >= 1
    
    @patch("os.cpu_count")
    def test_fallback_to_cpu_count(self, mock_cpu_count, monkeypatch):
        """Test fallback to os.cpu_count()"""
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        mock_cpu_count.return_value = 4
        
        result = get_optimal_num_threads()
        
        assert result == 4
    
    @patch("os.cpu_count")
    def test_returns_1_when_cpu_count_is_none(self, mock_cpu_count, monkeypatch):
        """Test that function returns 1 when cpu_count returns None"""
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        mock_cpu_count.return_value = None
        
        result = get_optimal_num_threads()
        
        assert result == 1


@pytest.mark.unit
class TestConfigureCPUThreading:
    """Tests for configure_cpu_threading function"""
    
    @patch("torch.set_num_threads")
    @patch("torch.set_num_interop_threads")
    @patch("src.services.model_loader.get_optimal_num_threads")
    def test_configures_torch_threads(self, mock_get_threads, mock_inter_op, mock_intra_op):
        """Test that PyTorch threading is configured correctly"""
        mock_get_threads.return_value = 8
        
        result = configure_cpu_threading()
        
        assert result == 8
        mock_intra_op.assert_called_once_with(8)
        mock_inter_op.assert_called_once_with(4)  # Half of intra-op
    
    @patch("torch.set_num_threads")
    @patch("torch.set_num_interop_threads")
    @patch("src.services.model_loader.get_optimal_num_threads")
    def test_sets_environment_variables(self, mock_get_threads, mock_inter_op, mock_intra_op):
        """Test that environment variables are set correctly"""
        mock_get_threads.return_value = 6
        
        configure_cpu_threading()
        
        assert os.environ["OMP_NUM_THREADS"] == "6"
        assert os.environ["MKL_NUM_THREADS"] == "6"
        assert os.environ["NUMEXPR_NUM_THREADS"] == "6"
    
    @patch("torch.set_num_threads")
    @patch("torch.set_num_interop_threads")
    @patch("src.services.model_loader.get_optimal_num_threads")
    def test_inter_op_is_half_of_intra_op(self, mock_get_threads, mock_inter_op, mock_intra_op):
        """Test that inter-op threads is half of intra-op threads"""
        mock_get_threads.return_value = 8
        
        configure_cpu_threading()
        
        mock_inter_op.assert_called_once_with(4)
    
    @patch("torch.set_num_threads")
    @patch("torch.set_num_interop_threads")
    @patch("src.services.model_loader.get_optimal_num_threads")
    def test_minimum_inter_op_threads_is_one(self, mock_get_threads, mock_inter_op, mock_intra_op):
        """Test that inter-op threads is at least 1"""
        mock_get_threads.return_value = 1
        
        configure_cpu_threading()
        
        mock_inter_op.assert_called_once_with(1)  # max(1, 1//2) = 1


@pytest.mark.unit
class TestLoadModel:
    """Tests for load_model function"""
    
    @patch("src.services.model_loader.AutoModelForImageClassification.from_pretrained")
    @patch("src.services.model_loader.AutoImageProcessor.from_pretrained")
    @patch("src.services.model_loader.configure_cpu_threading")
    @patch("torch.cuda.is_available")
    def test_load_model_with_cpu(self, mock_cuda, mock_config_threads, 
                                  mock_processor, mock_model):
        """Test loading model with CPU"""
        mock_cuda.return_value = False
        mock_config_threads.return_value = 4
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        mock_processor_instance = Mock()
        mock_processor.return_value = mock_processor_instance
        
        model, processor, device = load_model("/fake/path", force_cpu=True)
        
        assert model == mock_model_instance
        assert processor == mock_processor_instance
        assert device == "cpu"
        mock_config_threads.assert_called_once()
    
    @patch("src.services.model_loader.AutoModelForImageClassification.from_pretrained")
    @patch("src.services.model_loader.AutoImageProcessor.from_pretrained")
    @patch("src.services.model_loader.configure_cpu_threading")
    @patch("torch.cuda.is_available")
    def test_load_model_force_cpu_overrides_cuda(self, mock_cuda, mock_config_threads,
                                                   mock_processor, mock_model):
        """Test that force_cpu=True uses CPU even when CUDA is available"""
        mock_cuda.return_value = True  # CUDA available
        mock_config_threads.return_value = 4
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        mock_processor_instance = Mock()
        mock_processor.return_value = mock_processor_instance
        
        model, processor, device = load_model("/fake/path", force_cpu=True)
        
        assert device == "cpu"
    
    @patch("src.services.model_loader.AutoModelForImageClassification.from_pretrained")
    @patch("src.services.model_loader.AutoImageProcessor.from_pretrained")
    @patch("torch.cuda.is_available")
    def test_load_model_with_cuda(self, mock_cuda, mock_processor, mock_model):
        """Test loading model with CUDA"""
        mock_cuda.return_value = True
        mock_model_instance = Mock()
        mock_model_instance.to = Mock(return_value=mock_model_instance)
        mock_model_instance.eval = Mock(return_value=mock_model_instance)
        mock_model.return_value = mock_model_instance
        mock_processor_instance = Mock()
        mock_processor.return_value = mock_processor_instance
        
        model, processor, device = load_model("/fake/path", force_cpu=False)
        
        # Model should attempt to use cuda (to was called)
        mock_model_instance.to.assert_called()
        mock_model_instance.eval.assert_called()
    
    @patch("src.services.model_loader.AutoModelForImageClassification.from_pretrained")
    def test_load_model_raises_on_invalid_path(self, mock_model):
        """Test that load_model raises exception for invalid path"""
        mock_model.side_effect = Exception("Model not found")

        with pytest.raises(Exception):
            load_model("/invalid/path", force_cpu=True)


@pytest.mark.unit
class TestGetOptimalNumThreadsCgroup:
    """Tests for cgroup detection in get_optimal_num_threads."""

    def test_cgroup_v2_detection(self, monkeypatch):
        """Test cgroup v2 CPU limit detection."""
        from io import StringIO
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)

        mock_file = StringIO("200000 100000")
        mock_open = Mock(return_value=mock_file)
        setattr(mock_file, "close", Mock())  # prevent actual close

        with patch("builtins.open", mock_open):
            result = get_optimal_num_threads()
        assert result == 2

    def test_cgroup_v2_max_falls_through(self, monkeypatch):
        """Test cgroup v2 with 'max' value falls through to cpu_count."""
        from io import StringIO
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)

        mock_file = StringIO("max 100000")
        setattr(mock_file, "close", Mock())

        with patch("builtins.open", Mock(side_effect=[mock_file, FileNotFoundError, FileNotFoundError])), \
             patch("os.cpu_count", return_value=4):
            result = get_optimal_num_threads()
        assert result == 4


@pytest.mark.unit
class TestJITClasses:
    """Tests for JIT model wrapper classes."""

    def test_jit_model_wrapper_forward(self):
        """Test JITModelWrapper forward pass."""
        import torch
        from src.services.model_loader import JITModelWrapper

        inner = Mock()
        output = Mock()
        output.logits = torch.tensor([[0.1, 0.9]])
        inner.return_value = output

        wrapper = JITModelWrapper(inner)
        pixel_values = torch.randn(1, 3, 32, 32)
        result = wrapper(pixel_values)
        assert torch.equal(result, output.logits)

    def test_jit_optimized_model_call(self):
        """Test JITOptimizedModel call returns model output."""
        import torch
        from src.services.model_loader import JITOptimizedModel

        jit_model = Mock()
        jit_model.return_value = torch.tensor([[0.2, 0.8]])
        config = Mock()

        model = JITOptimizedModel(jit_model, config)
        result = model(torch.randn(1, 3, 32, 32))
        assert hasattr(result, "logits")

    def test_jit_optimized_model_eval_and_to(self):
        """Test JITOptimizedModel eval() and to() return self."""
        from src.services.model_loader import JITOptimizedModel

        model = JITOptimizedModel(Mock(), Mock())
        assert model.eval() is model
        assert model.to("cpu") is model

    def test_model_output_has_logits(self):
        """Test _ModelOutput stores logits."""
        import torch
        from src.services.model_loader import _ModelOutput

        logits = torch.tensor([[0.5, 0.5]])
        output = _ModelOutput(logits)
        assert torch.equal(output.logits, logits)


@pytest.mark.unit
class TestApplyJitOptimization:
    """Tests for apply_jit_optimization function."""

    def test_skips_non_cpu(self):
        """Test JIT optimization is skipped for non-CPU devices."""
        from src.services.model_loader import apply_jit_optimization

        model = Mock()
        result = apply_jit_optimization(model, "cuda:0")
        assert result is model

    @patch("torch.jit.trace")
    @patch("torch.jit.freeze")
    def test_applies_on_cpu(self, mock_freeze, mock_trace):
        """Test JIT optimization is applied on CPU."""
        import torch
        from src.services.model_loader import apply_jit_optimization

        model = Mock()
        model.config = Mock()
        mock_frozen = Mock()
        mock_frozen.return_value = torch.tensor([[0.1, 0.9]])
        mock_trace.return_value = Mock()
        mock_freeze.return_value = mock_frozen

        result = apply_jit_optimization(model, "cpu")
        assert result is not model  # should be JITOptimizedModel

    def test_falls_back_on_jit_failure(self):
        """Test JIT optimization falls back to original on error."""
        from src.services.model_loader import apply_jit_optimization

        model = Mock()
        model.config = Mock()
        # JITModelWrapper will fail when calling .eval() on wrapper
        with patch("torch.jit.trace", side_effect=RuntimeError("trace failed")):
            result = apply_jit_optimization(model, "cpu")
        assert result is model
