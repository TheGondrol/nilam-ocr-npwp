"""
Device Detection Module
Detects GPU availability, CPU resources, and provides dynamic threading configuration.
Supports Kubernetes pod resource limits via cgroup detection.
"""

import torch
import os
from pathlib import Path
from typing import Tuple

from .logging import logger
from .config import config


def is_running_in_container() -> bool:
    """
    Detect if running inside a container (Docker/Kubernetes).

    Returns:
        True if running in a container, False otherwise
    """
    # Check for Kubernetes service account
    if os.path.exists("/var/run/secrets/kubernetes.io"):
        return True

    # Check for Docker indicator
    if os.path.exists("/.dockerenv"):
        return True

    # Check cgroup for container indicators
    try:
        cgroup_path = Path("/proc/1/cgroup")
        if cgroup_path.exists():
            content = cgroup_path.read_text()
            if any(
                indicator in content
                for indicator in ["docker", "kubepods", "containerd"]
            ):
                return True
    except (PermissionError, OSError):
        pass

    # Check for Kubernetes environment variables
    if os.getenv("KUBERNETES_SERVICE_HOST"):
        return True

    return False


def get_cpu_count() -> int:
    """
    Get the effective CPU count, respecting container/Kubernetes limits.

    Priority:
        1. Environment variable override (CPU_LIMIT from K8s Downward API)
        2. cgroup v2 limits (/sys/fs/cgroup/cpu.max)
        3. cgroup v1 limits (/sys/fs/cgroup/cpu/cpu.cfs_quota_us)
        4. os.sched_getaffinity (Linux process affinity)
        5. os.process_cpu_count (Python 3.13+)
        6. os.cpu_count() fallback

    Returns:
        Number of available CPUs (minimum 1)
    """
    # Check for environment variable override (K8s Downward API)
    cpu_limit_env = os.getenv("CPU_LIMIT")
    if cpu_limit_env:
        try:
            # Kubernetes may return millicores (e.g., "500m" or "2")
            if cpu_limit_env.endswith("m"):
                cpu_count = max(1, int(cpu_limit_env[:-1]) // 1000)
            else:
                cpu_count = max(1, int(float(cpu_limit_env)))
            logger.info(f"CPU count from CPU_LIMIT env: {cpu_count}")
            return cpu_count
        except ValueError:
            pass

    # Check config override
    config_override = config.get("performance.override_cpu_count", None)
    if config_override and isinstance(config_override, int) and config_override > 0:
        logger.info(f"CPU count from config override: {config_override}")
        return config_override

    # Try cgroup v2 (modern Kubernetes)
    cpu_max_path = Path("/sys/fs/cgroup/cpu.max")
    if cpu_max_path.exists():
        try:
            content = cpu_max_path.read_text().strip()
            parts = content.split()
            if parts[0] != "max":  # "max" means unlimited
                quota = int(parts[0])
                period = int(parts[1])
                cpu_count = max(1, quota // period)
                logger.info(f"CPU count from cgroup v2: {cpu_count}")
                return cpu_count
        except (ValueError, IndexError, PermissionError) as e:
            logger.debug(f"Failed to read cgroup v2: {e}")

    # Try cgroup v1 (older Kubernetes)
    quota_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota_path.exists() and period_path.exists():
        try:
            quota = int(quota_path.read_text().strip())
            period = int(period_path.read_text().strip())
            if quota > 0:  # -1 means unlimited
                cpu_count = max(1, quota // period)
                logger.info(f"CPU count from cgroup v1: {cpu_count}")
                return cpu_count
        except (ValueError, PermissionError) as e:
            logger.debug(f"Failed to read cgroup v1: {e}")

    # Try sched_getaffinity (Linux)
    if hasattr(os, "sched_getaffinity"):
        try:
            cpu_count = len(os.sched_getaffinity(0))
            logger.info(f"CPU count from sched_getaffinity: {cpu_count}")
            return cpu_count
        except (OSError, AttributeError) as e:
            logger.debug(f"Failed to get sched_getaffinity: {e}")

    # Try process_cpu_count (Python 3.13+)
    if hasattr(os, "process_cpu_count"):
        count = os.process_cpu_count()
        if count is not None:
            logger.info(f"CPU count from process_cpu_count: {count}")
            return count

    # Fallback to host CPU count
    cpu_count = os.cpu_count() or 1
    logger.info(f"CPU count from os.cpu_count (fallback): {cpu_count}")
    return cpu_count


def get_optimal_worker_count(for_inference: bool = True) -> int:
    """
    Calculate optimal worker thread count based on workload and available CPUs.

    Args:
        for_inference: If True, optimizes for ML inference (CPU-bound).
                       If False, optimizes for I/O-bound workloads.

    Returns:
        Recommended number of worker threads
    """
    cpu_count = get_cpu_count()
    
    # Check if running under multi-worker uvicorn
    worker_count = _get_uvicorn_worker_count()

    if for_inference:
        # For ML inference (CPU-bound): allocate CPUs per worker
        if torch.cuda.is_available():
            workers = max(2, min(cpu_count, 4))  # 2-4 threads for GPU inference
        else:
            # Divide resources if multiple uvicorn workers
            per_worker_cpus = max(1, cpu_count // worker_count) if worker_count > 1 else cpu_count
            workers = max(1, per_worker_cpus)  # Use all allocated CPUs
    else:
        # For I/O-bound: can use more threads
        workers = min(cpu_count * 2, 32)  # Cap at 32

    logger.info(
        f"Optimal worker count: {workers} (inference={for_inference}, cpus={cpu_count}, uvicorn_workers={worker_count})"
    )
    return workers


def _get_uvicorn_worker_count() -> int:
    """
    Detect number of uvicorn workers from environment.
    Returns 1 if not detected (single worker mode).
    """
    # Uvicorn sets WEB_CONCURRENCY for worker processes
    worker_env = os.getenv("WEB_CONCURRENCY")
    if worker_env:
        try:
            return max(1, int(worker_env))
        except ValueError:
            pass
    return 1


def configure_cpu_threading():
    """
    Configure CPU threading for optimal performance.
    Sets PyTorch, OpenMP, MKL, and OpenVINO thread counts based on detected CPU resources.

    Should be called early in application startup.
    """
    cpu_count = get_cpu_count()
    worker_count = _get_uvicorn_worker_count()
    in_container = is_running_in_container()
    
    # Allocate CPUs per worker if running in multi-worker mode
    per_worker_cpus = max(1, cpu_count // worker_count) if worker_count > 1 else cpu_count

    logger.info(
        f"Configuring CPU threading: {cpu_count} CPUs, {worker_count} workers, {per_worker_cpus} CPUs/worker, container={in_container}"
    )

    # Set PyTorch threads (use all allocated CPUs per worker)
    torch.set_num_threads(per_worker_cpus)
    logger.info(f"PyTorch threads set to: {per_worker_cpus}")

    # Set interop threads (for parallel regions)
    interop_threads = max(1, per_worker_cpus // 2)
    torch.set_num_interop_threads(interop_threads)
    logger.info(f"PyTorch interop threads set to: {interop_threads}")

    # Set OpenMP threads (only if not already set)
    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = str(per_worker_cpus)
        logger.info(f"OMP_NUM_THREADS set to: {per_worker_cpus}")

    # Set MKL threads (Intel Math Kernel Library)
    if "MKL_NUM_THREADS" not in os.environ:
        os.environ["MKL_NUM_THREADS"] = str(per_worker_cpus)
        logger.info(f"MKL_NUM_THREADS set to: {per_worker_cpus}")

    # Set OpenBLAS threads
    if "OPENBLAS_NUM_THREADS" not in os.environ:
        os.environ["OPENBLAS_NUM_THREADS"] = str(per_worker_cpus)
        logger.info(f"OPENBLAS_NUM_THREADS set to: {per_worker_cpus}")
    
    # Configure OpenVINO for throughput optimization
    _configure_openvino_threading(per_worker_cpus)


def _configure_openvino_threading(cpu_count: int):
    """
    Configure OpenVINO-specific environment variables for optimal CPU utilization.
    
    Args:
        cpu_count: Number of CPUs allocated to this worker
    """
    openvino_mode = config.get("performance.openvino_mode", "throughput")
    openvino_streams = config.get("performance.openvino_streams", "AUTO")
    
    # Disable OpenVINO telemetry to prevent potential signal handler interference
    if "OPENVINO_TELEMETRY" not in os.environ:
        os.environ["OPENVINO_TELEMETRY"] = "0"
    
    # Suppress OpenVINO's internal verbose logging
    if "OPENVINO_LOG_LEVEL" not in os.environ:
        os.environ["OPENVINO_LOG_LEVEL"] = "0"
    
    # Set performance hint
    if "OPENVINO_INFERENCE_PRECISION_HINT" not in os.environ:
        os.environ["OPENVINO_INFERENCE_PRECISION_HINT"] = "f32"  # Use FP32 for CPU
        logger.info("OpenVINO precision hint set to: f32")
    
    # Set performance mode (throughput is better for concurrent requests)
    if openvino_mode == "throughput":
        if "OPENVINO_THROUGHPUT_STREAMS" not in os.environ:
            streams = str(openvino_streams) if openvino_streams != "AUTO" else "AUTO"
            os.environ["OPENVINO_THROUGHPUT_STREAMS"] = streams
            logger.info(f"OpenVINO throughput streams set to: {streams}")
        
        # Enable async execution for better CPU utilization
        if "OPENVINO_CPU_BIND_THREAD" not in os.environ:
            os.environ["OPENVINO_CPU_BIND_THREAD"] = "YES"
            logger.info("OpenVINO CPU thread binding enabled")
    
    logger.info(f"OpenVINO configured for {openvino_mode} mode")


def get_device() -> Tuple[str, dict]:
    """
    Detect and return the appropriate device for PyTorch.
    Also configures CPU threading if using CPU.

    Returns:
        Tuple of (device_string, device_info_dict)
    """
    cpu_count = get_cpu_count()
    in_container = is_running_in_container()

    device_info = {
        "type": "cpu",
        "name": "CPU",
        "cuda_available": False,
        "cpu_count": cpu_count,
        "in_container": in_container,
        "worker_threads": get_optimal_worker_count(for_inference=True),
    }

    # Check if CPU is forced
    force_cpu = config.get("device.force_cpu", False)
    if force_cpu:
        logger.info("CPU mode forced by configuration")
        configure_cpu_threading()
        return "cpu", device_info

    # Check for CUDA_VISIBLE_DEVICES environment variable
    cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible == "":
        logger.info("CPU mode forced by CUDA_VISIBLE_DEVICES environment variable")
        configure_cpu_threading()
        return "cpu", device_info

    # Check CUDA availability
    prefer_gpu = config.get("device.prefer_gpu", True)

    if prefer_gpu and torch.cuda.is_available():
        try:
            device_info["type"] = "cuda"
            device_info["cuda_available"] = True
            device_info["name"] = torch.cuda.get_device_name(0)
            device_info["memory_gb"] = (
                torch.cuda.get_device_properties(0).total_memory / 1024**3
            )
            device_info["cuda_version"] = torch.version.cuda
            device_info["device_count"] = torch.cuda.device_count()

            logger.info(f"Using GPU: {device_info['name']}")
            logger.info(f"GPU Memory: {device_info['memory_gb']:.2f} GB")
            logger.info(f"CUDA Version: {device_info['cuda_version']}")
            logger.info(f"Device Count: {device_info['device_count']}")

            return "cuda", device_info

        except Exception as e:
            logger.warning(f"Failed to initialize CUDA: {str(e)}")
            logger.warning("Falling back to CPU")
            device_info["type"] = "cpu"
            device_info["name"] = "CPU (CUDA failed)"
            configure_cpu_threading()
            return "cpu", device_info
    else:
        if not prefer_gpu:
            logger.info("GPU disabled by configuration, using CPU")
        else:
            logger.warning("CUDA not available, using CPU")

        configure_cpu_threading()
        return "cpu", device_info


def log_device_info(device: str, device_info: dict):
    """
    Log detailed device information.

    Args:
        device: Device string ('cuda' or 'cpu')
        device_info: Device information dictionary
    """
    logger.info(f"Device: {device}")
    logger.info(f"Device Type: {device_info['type']}")
    logger.info(f"Device Name: {device_info['name']}")
    logger.info(f"CPU Count: {device_info.get('cpu_count', 'unknown')}")
    logger.info(f"Worker Threads: {device_info.get('worker_threads', 'unknown')}")
    logger.info(f"Running in Container: {device_info.get('in_container', False)}")

    if device_info["cuda_available"]:
        logger.info("CUDA Available: Yes")
        if "memory_gb" in device_info:
            logger.info(f"GPU Memory: {device_info['memory_gb']:.2f} GB")
        if "cuda_version" in device_info:
            logger.info(f"CUDA Version: {device_info['cuda_version']}")
    else:
        logger.info("CUDA Available: No")
