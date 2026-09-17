"""
Unit tests for core.device module
"""
import os
from unittest.mock import patch, MagicMock

import pytest

from src.core.device import (
    is_running_in_container,
    get_cpu_count,
    get_device,
    get_optimal_worker_count,
    log_device_info,
    configure_cpu_threading,
    _configure_openvino_threading,
)


@pytest.mark.unit
class TestDeviceDetection:
    """Test device detection functions"""
    
    def test_is_running_in_container_kubernetes(self):
        """Test container detection for Kubernetes"""
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            assert is_running_in_container()
    
    def test_is_running_in_container_docker(self):
        """Test container detection for Docker"""
        with patch('os.path.exists') as mock_exists:
            def side_effect(path):
                if path == "/.dockerenv":
                    return True
                return False
            mock_exists.side_effect = side_effect
            assert is_running_in_container()
    
    def test_is_running_in_container_cgroup(self):
        """Test container detection via cgroup"""
        with patch('os.path.exists', return_value=True):
            with patch('pathlib.Path.read_text', return_value='docker/container123'):
                assert is_running_in_container()
    
    def test_is_running_in_container_env_var(self):
        """Test container detection via environment variable"""
        with patch.dict(os.environ, {'KUBERNETES_SERVICE_HOST': 'kubernetes'}):
            assert is_running_in_container()
    
    def test_is_not_running_in_container(self):
        """Test when not running in container"""
        with patch('os.path.exists', return_value=False):
            with patch.dict(os.environ, {}, clear=True):
                assert not is_running_in_container()
    
    def test_get_cpu_count_from_env(self, clean_env):
        """Test CPU count from environment variable"""
        os.environ['CPU_LIMIT'] = '4'
        assert get_cpu_count() == 4
    
    def test_get_cpu_count_from_env_millicores(self, clean_env):
        """Test CPU count from environment variable with millicores"""
        os.environ['CPU_LIMIT'] = '500m'
        # 500 millicores = 0 full cores, but minimum is 1
        assert get_cpu_count() == 1
        
        os.environ['CPU_LIMIT'] = '2500m'
        # 2500 millicores = 2 full cores
        assert get_cpu_count() == 2
    
    def test_get_cpu_count_cgroup_v2(self, clean_env):
        """Test CPU count from cgroup v2"""
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.read_text', return_value='200000 100000'):
                # 200000 quota / 100000 period = 2 CPUs
                count = get_cpu_count()
                assert count == 2
    
    def test_get_cpu_count_cgroup_v2_unlimited(self, clean_env):
        """Test CPU count when cgroup v2 shows unlimited"""
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.read_text', return_value='max 100000'):
                with patch('os.cpu_count', return_value=8):
                    # Should fall back to os.cpu_count
                    count = get_cpu_count()
                    assert count >= 1
    
    def test_get_cpu_count_fallback(self, clean_env):
        """Test CPU count fallback to os.cpu_count"""
        with patch('pathlib.Path.exists', return_value=False):
            with patch('os.cpu_count', return_value=4):
                assert get_cpu_count() == 4
    
    def test_get_cpu_count_minimum_one(self, clean_env):
        """Test that CPU count is at least 1"""
        with patch('os.cpu_count', return_value=None):
            assert get_cpu_count() >= 1


@pytest.mark.unit
class TestGetDevice:
    """Test get_device function"""
    
    def test_get_device_gpu_available(self):
        """Test device selection when GPU is available"""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_name', return_value='NVIDIA Tesla'):
                with patch('torch.cuda.get_device_properties') as mock_props:
                    mock_props.return_value = MagicMock(total_memory=8589934592)
                    with patch('src.core.device.config') as mock_config:
                        with patch('src.core.device.configure_cpu_threading'):
                            mock_config.get.side_effect = lambda key, default=None: {
                                'device.force_cpu': False,
                                'device.prefer_gpu': True
                            }.get(key, default)
                            
                            device, info = get_device()
                            
                            assert device == 'cuda'
                            assert info['type'] == 'cuda'
    
    def test_get_device_gpu_preferred_but_unavailable(self):
        """Test device selection when GPU preferred but not available"""
        with patch('torch.cuda.is_available', return_value=False):
            with patch('src.core.device.config') as mock_config:
                with patch('src.core.device.configure_cpu_threading'):
                    with patch('os.cpu_count', return_value=8):
                        mock_config.get.side_effect = lambda key, default=None: {
                            'device.prefer_gpu': True,
                            'device.force_cpu': False
                        }.get(key, default)
                        
                        device, info = get_device()
                        
                        assert device == 'cpu'
                        assert info['type'] == 'cpu'
    
    def test_get_device_force_cpu(self):
        """Test device selection when CPU is forced"""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('src.core.device.config') as mock_config:
                with patch('src.core.device.configure_cpu_threading'):
                    with patch('os.cpu_count', return_value=8):
                        mock_config.get.side_effect = lambda key, default=None: {
                            'device.force_cpu': True
                        }.get(key, default)
                        
                        device, info = get_device()
                        
                        assert device == 'cpu'
                        assert info['type'] == 'cpu'
    
    def test_get_device_cpu_info(self):
        """Test CPU device info"""
        with patch('torch.cuda.is_available', return_value=False):
            with patch('src.core.device.config') as mock_config:
                with patch('src.core.device.configure_cpu_threading'):
                    mock_config.get.side_effect = lambda key, default=None: {
                        'device.force_cpu': False,
                        'device.prefer_gpu': False
                    }.get(key, default)
                    with patch('os.sched_getaffinity', side_effect=OSError):
                        with patch('os.cpu_count', return_value=8):

                            device, info = get_device()

                            assert device == 'cpu'
                            assert info['type'] == 'cpu'
                            assert info['cpu_count'] == 8


@pytest.mark.unit
class TestOptimalWorkerCount:
    """Test get_optimal_worker_count function"""
    
    def test_optimal_worker_count_for_inference_cpu(self, mock_torch_cpu):
        """Test worker count for inference on CPU"""
        with patch('src.core.device.get_cpu_count', return_value=4):
            count = get_optimal_worker_count(for_inference=True)
            # For inference on CPU, should return CPU count
            assert count == 4
    
    def test_optimal_worker_count_for_inference_gpu(self, mock_torch_cuda):
        """Test worker count for inference on GPU"""
        with patch('src.core.config.config') as mock_config:
            mock_config.get.return_value = True
            with patch('src.core.device.get_cpu_count', return_value=8):
                with patch('torch.cuda.is_available', return_value=True):
                    count = get_optimal_worker_count(for_inference=True)
                    # With GPU available, still uses CPU count / 2 for inference workers
                    assert count >= 1
    
    def test_optimal_worker_count_for_io(self, mock_torch_cpu):
        """Test worker count for I/O operations"""
        with patch('src.core.device.get_cpu_count', return_value=4):
            count = get_optimal_worker_count(for_inference=False)
            # For I/O, should be 2x CPU count (capped at 8)
            assert count == 8
    
    def test_optimal_worker_count_io_capped(self, mock_torch_cpu):
        """Test worker count for I/O operations"""
        with patch('src.core.device.get_cpu_count', return_value=16):
            count = get_optimal_worker_count(for_inference=False)
            # Should use CPU count * 2 for I/O
            assert count >= 8
    
    def test_optimal_worker_count_minimum(self, mock_torch_cpu):
        """Test worker count has minimum of 1"""
        with patch('src.core.device.get_cpu_count', return_value=0):
            count = get_optimal_worker_count(for_inference=True)
            assert count >= 1


@pytest.mark.unit
class TestLogDeviceInfo:
    """Test log_device_info function"""
    
    def test_log_device_info_gpu(self):
        """Test logging GPU device info"""
        device = 'cuda:0'
        info = {
            'type': 'GPU',
            'name': 'NVIDIA GeForce GTX 1080',
            'memory_gb': 8.0,
            'cuda_available': True,
            'count': 12,
            'workers': 1,
            'in_container': False
        }
        
        # Should not raise any exceptions
        log_device_info(device, info)
    
    def test_log_device_info_cpu(self):
        """Test logging CPU device info"""
        device = 'cpu'
        info = {
            'type': 'CPU',
            'count': 8,
            'name': 'CPU',
            'cuda_available': False,
            'workers': 1,
            'in_container': False
        }

        # Should not raise any exceptions
        log_device_info(device, info)


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIsRunningInContainerEdgeCases:
    """Edge-case tests for is_running_in_container."""

    def test_cgroup_permission_error_returns_false(self):
        """PermissionError when reading cgroup should not crash, returns False."""
        with patch('os.path.exists', return_value=False):
            with patch('pathlib.Path.exists', return_value=True):
                with patch('pathlib.Path.read_text', side_effect=PermissionError):
                    with patch.dict(os.environ, {}, clear=True):
                        assert not is_running_in_container()

    def test_cgroup_containerd_indicator(self):
        """Cgroup content containing 'containerd' should be detected."""
        with patch('os.path.exists', return_value=False):
            with patch('pathlib.Path.exists', return_value=True):
                with patch('pathlib.Path.read_text', return_value='/system.slice/containerd.service'):
                    assert is_running_in_container()


@pytest.mark.unit
class TestGetCpuCountEdgeCases:
    """Edge-case tests for get_cpu_count."""

    def test_cpu_limit_invalid_value_falls_through(self, clean_env):
        """Invalid CPU_LIMIT (e.g. 'abc') should fall through to next method."""
        os.environ['CPU_LIMIT'] = 'abc'
        with patch('pathlib.Path.exists', return_value=False):
            with patch('os.sched_getaffinity', side_effect=OSError):
                with patch('os.cpu_count', return_value=6):
                    assert get_cpu_count() == 6

    def test_cpu_limit_float_parsed_as_int(self, clean_env):
        """Float CPU_LIMIT (e.g. '2.5') should be parsed as int(float(...)) = 2."""
        os.environ['CPU_LIMIT'] = '2.5'
        assert get_cpu_count() == 2

    def test_config_override_valid_int(self, clean_env):
        """Config override with a valid positive int should be used."""
        with patch('src.core.device.config') as mock_config:
            mock_config.get.side_effect = lambda key, default=None: {
                'performance.override_cpu_count': 3,
            }.get(key, default)
            assert get_cpu_count() == 3

    def test_cgroup_v2_value_error_falls_through(self, clean_env):
        """ValueError when parsing cgroup v2 should fall through."""
        with patch('src.core.device.config') as mock_config:
            mock_config.get.return_value = None
            with patch('pathlib.Path.exists', return_value=True):
                with patch('pathlib.Path.read_text', return_value='not_a_number 100000'):
                    with patch('os.sched_getaffinity', side_effect=OSError):
                        with patch('os.cpu_count', return_value=5):
                            assert get_cpu_count() == 5

    def test_cgroup_v2_index_error_falls_through(self, clean_env):
        """IndexError when parsing cgroup v2 should fall through."""
        with patch('src.core.device.config') as mock_config:
            mock_config.get.return_value = None
            with patch('pathlib.Path.exists', return_value=True):
                with patch('pathlib.Path.read_text', return_value=''):
                    with patch('os.cpu_count', return_value=7):
                        result = get_cpu_count()
                        assert result >= 1

    def test_cgroup_v1_quota_positive(self, clean_env):
        """cgroup v1 with positive quota should compute CPU count."""
        with patch('src.core.device.config') as mock_config:
            mock_config.get.return_value = None
            cpu_max = MagicMock()
            cpu_max.exists.return_value = False  # skip v2

            quota_path = MagicMock()
            quota_path.exists.return_value = True
            quota_path.read_text.return_value = '400000'

            period_path = MagicMock()
            period_path.exists.return_value = True
            period_path.read_text.return_value = '100000'

            def make_path(p):
                if 'cpu.max' in str(p):
                    return cpu_max
                if 'quota' in str(p):
                    return quota_path
                if 'period' in str(p):
                    return period_path
                m = MagicMock()
                m.exists.return_value = False
                return m

            with patch('src.core.device.Path', side_effect=make_path):
                assert get_cpu_count() == 4

    def test_cgroup_v1_permission_error_falls_through(self, clean_env):
        """PermissionError reading cgroup v1 should fall through."""
        with patch('src.core.device.config') as mock_config:
            mock_config.get.return_value = None
            cpu_max = MagicMock()
            cpu_max.exists.return_value = False  # skip v2

            quota_path = MagicMock()
            quota_path.exists.return_value = True
            quota_path.read_text.side_effect = PermissionError

            period_path = MagicMock()
            period_path.exists.return_value = True

            def make_path(p):
                if 'cpu.max' in str(p):
                    return cpu_max
                if 'quota' in str(p):
                    return quota_path
                if 'period' in str(p):
                    return period_path
                m = MagicMock()
                m.exists.return_value = False
                return m

            with patch('src.core.device.Path', side_effect=make_path):
                with patch('os.cpu_count', return_value=10):
                    result = get_cpu_count()
                    assert result >= 1

    def test_sched_getaffinity_oserror_falls_through(self, clean_env):
        """OSError from sched_getaffinity should fall through."""
        with patch('src.core.device.config') as mock_config:
            mock_config.get.return_value = None
            with patch('pathlib.Path.exists', return_value=False):
                with patch('os.sched_getaffinity', side_effect=OSError, create=True):
                    with patch('os.cpu_count', return_value=12):
                        assert get_cpu_count() == 12


@pytest.mark.unit
class TestConfigureCpuThreading:
    """Tests for configure_cpu_threading."""

    def test_sets_pytorch_threads(self, clean_env):
        """Should set PyTorch num_threads and interop_threads."""
        with patch('src.core.device.get_cpu_count', return_value=4):
            with patch('src.core.device._get_uvicorn_worker_count', return_value=1):
                with patch('src.core.device.is_running_in_container', return_value=False):
                    with patch('src.core.device._configure_openvino_threading'):
                        with patch('torch.set_num_threads') as mock_threads:
                            with patch('torch.set_num_interop_threads') as mock_interop:
                                configure_cpu_threading()
                                mock_threads.assert_called_once_with(4)
                                mock_interop.assert_called_once_with(2)

    def test_sets_omp_mkl_openblas_env_vars(self, clean_env):
        """Should set OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS."""
        os.environ.pop('OMP_NUM_THREADS', None)
        os.environ.pop('MKL_NUM_THREADS', None)
        os.environ.pop('OPENBLAS_NUM_THREADS', None)
        with patch('src.core.device.get_cpu_count', return_value=4):
            with patch('src.core.device._get_uvicorn_worker_count', return_value=1):
                with patch('src.core.device.is_running_in_container', return_value=False):
                    with patch('src.core.device._configure_openvino_threading'):
                        with patch('torch.set_num_threads'):
                            with patch('torch.set_num_interop_threads'):
                                configure_cpu_threading()
                                assert os.environ['OMP_NUM_THREADS'] == '4'
                                assert os.environ['MKL_NUM_THREADS'] == '4'
                                assert os.environ['OPENBLAS_NUM_THREADS'] == '4'

    def test_does_not_override_existing_omp_num_threads(self, clean_env):
        """Should not overwrite OMP_NUM_THREADS if already set."""
        os.environ['OMP_NUM_THREADS'] = '2'
        with patch('src.core.device.get_cpu_count', return_value=8):
            with patch('src.core.device._get_uvicorn_worker_count', return_value=1):
                with patch('src.core.device.is_running_in_container', return_value=False):
                    with patch('src.core.device._configure_openvino_threading'):
                        with patch('torch.set_num_threads'):
                            with patch('torch.set_num_interop_threads'):
                                configure_cpu_threading()
                                # Must stay at original value
                                assert os.environ['OMP_NUM_THREADS'] == '2'


@pytest.mark.unit
class TestConfigureOpenvinoThreading:
    """Tests for _configure_openvino_threading."""

    def test_throughput_mode_sets_streams_and_bind(self, clean_env):
        """Throughput mode should set OPENVINO_THROUGHPUT_STREAMS and CPU_BIND_THREAD."""
        with patch('src.core.device.config') as mock_config:
            mock_config.get.side_effect = lambda key, default=None: {
                'performance.openvino_mode': 'throughput',
                'performance.openvino_streams': 'AUTO',
            }.get(key, default)
            _configure_openvino_threading(4)
            assert os.environ['OPENVINO_THROUGHPUT_STREAMS'] == 'AUTO'
            assert os.environ['OPENVINO_CPU_BIND_THREAD'] == 'YES'
            assert os.environ['OPENVINO_TELEMETRY'] == '0'
            assert os.environ['OPENVINO_LOG_LEVEL'] == '0'
            assert os.environ['OPENVINO_INFERENCE_PRECISION_HINT'] == 'f32'

    def test_latency_mode_does_not_set_throughput_streams(self, clean_env):
        """Latency mode should NOT set OPENVINO_THROUGHPUT_STREAMS."""
        with patch('src.core.device.config') as mock_config:
            mock_config.get.side_effect = lambda key, default=None: {
                'performance.openvino_mode': 'latency',
                'performance.openvino_streams': 'AUTO',
            }.get(key, default)
            _configure_openvino_threading(4)
            assert 'OPENVINO_THROUGHPUT_STREAMS' not in os.environ
            assert 'OPENVINO_CPU_BIND_THREAD' not in os.environ


@pytest.mark.unit
class TestGetDeviceEdgeCases:
    """Edge-case tests for get_device."""

    def test_cuda_visible_devices_empty_forces_cpu(self):
        """CUDA_VISIBLE_DEVICES='' should force CPU mode."""
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}):
            with patch('src.core.device.config') as mock_config:
                mock_config.get.side_effect = lambda key, default=None: {
                    'device.force_cpu': False,
                    'device.prefer_gpu': True,
                }.get(key, default)
                with patch('src.core.device.configure_cpu_threading'):
                    with patch('src.core.device.get_cpu_count', return_value=4):
                        with patch('src.core.device.is_running_in_container', return_value=False):
                            with patch('src.core.device.get_optimal_worker_count', return_value=4):
                                device, info = get_device()
                                assert device == 'cpu'
                                assert info['type'] == 'cpu'

    def test_cuda_init_failure_falls_back_to_cpu(self):
        """If CUDA init raises an exception, should fall back to CPU."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_name', side_effect=RuntimeError('CUDA init failed')):
                with patch('src.core.device.config') as mock_config:
                    mock_config.get.side_effect = lambda key, default=None: {
                        'device.force_cpu': False,
                        'device.prefer_gpu': True,
                    }.get(key, default)
                    with patch('src.core.device.configure_cpu_threading'):
                        with patch('src.core.device.get_cpu_count', return_value=4):
                            with patch('src.core.device.is_running_in_container', return_value=False):
                                with patch('src.core.device.get_optimal_worker_count', return_value=4):
                                    device, info = get_device()
                                    assert device == 'cpu'
                                    assert info['type'] == 'cpu'
                                    assert 'CUDA failed' in info['name']
