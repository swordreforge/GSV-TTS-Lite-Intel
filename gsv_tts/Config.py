import logging
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


def get_cuda_device_info(idx: int):
    """获取 CUDA 设备信息"""
    if not torch.cuda.is_available() or idx >= torch.cuda.device_count():
        return None

    try:
        props = torch.cuda.get_device_properties(idx)
    except Exception:
        return None

    name = props.name
    major, minor = props.major, props.minor
    sm_version = major + minor / 10.0
    mem_gb = props.total_memory / (1024**3)

    if sm_version < 5.3:
        return None

    device = torch.device(f"cuda:{idx}")

    # 针对旧架构或特殊系列强制使用 float32
    is_16_series = (major == 7 and minor == 5) and ("16" in name)
    if sm_version == 6.1 or is_16_series:
        return device, torch.float32, sm_version, mem_gb

    # 针对 Volta (sm 7.0) 和 Turing (sm 7.5，除16系列外) 及以上架构使用 float16
    if sm_version >= 7.0:
        return device, torch.float16, sm_version, mem_gb

    # 其他情况兜底使用 float32
    return device, torch.float32, sm_version, mem_gb

def get_xpu_device_info(idx: int = 0):
    """获取 Intel XPU (Arc GPU / NPU) 设备信息"""
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        return None
    if idx >= torch.xpu.device_count():
        return None

    try:
        props = torch.xpu.get_device_properties(idx)
        mem_gb = props.total_memory / (1024 ** 3)
    except Exception:
        return None

    device = torch.device(f"xpu:{idx}")

    # Intel Arc 对 float16 / bfloat16 均有良好支持，
    # 沿用 CUDA 侧的 float16 快路径即可。
    return device, torch.float16, 0.0, mem_gb


def get_mps_device_info():
    """获取 Apple Silicon MPS 设备信息"""
    if not torch.backends.mps.is_available():
        return None

    try:
        # MPS 设备
        device = torch.device("mps")
        # Apple Silicon 上 MPS 使用 float32 更稳定
        # 虽然 MPS 支持 float16，但在某些模型上可能有精度问题
        return device, torch.float32, 0.0, 0.0  # sm_version 和 mem_gb 对 MPS 不适用
    except Exception:
        return None

def choose_attention_backend(batch=1, heads=8, seq=128, head_dim=64, dtype=torch.float16):
    # 选择探测所用的加速设备：CUDA -> Intel XPU
    if torch.cuda.is_available():
        probe_device = "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        probe_device = "xpu"
    else:
        logging.info("SDPBackend: MATH")
        return SDPBackend.MATH

    k = torch.randn(batch, heads, seq, head_dim, device=probe_device, dtype=dtype)
    v = torch.randn(batch, heads, seq, head_dim, device=probe_device, dtype=dtype)

    probes = []
    for q_len in (seq, 1):
        q = torch.randn(batch, heads, q_len, head_dim, device=probe_device, dtype=dtype)
        mask = torch.zeros(batch, heads, q_len, seq, device=probe_device, dtype=torch.bool)
        probes.append((q, mask))

    def is_usable(backend):
        try:
            for q, mask in probes:
                with sdpa_kernel(backend):
                    F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            return True
        except Exception:
            return False

    # CUDA 优先尝试 cuDNN / 内存高效注意力；
    # Intel XPU 上 cuDNN / FlashAttention 不可用（仅 PVC/BMG 支持），
    # 实际会回退到 EFFICIENT_ATTENTION（oneDNN 内存高效注意力）。
    for backend in (SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION):
        if is_usable(backend):
            logging.info(f"SDPBackend: {backend.name}")
            return backend

    logging.info("SDPBackend: MATH")
    return SDPBackend.MATH


# 检测设备类型和配置
device = None
dtype = None

# 优先尝试 CUDA
if torch.cuda.is_available():
    GPU_COUNT = torch.cuda.device_count()
    available_devices = []
    for i in range(GPU_COUNT):
        info = get_cuda_device_info(i)
        if info is not None:
            available_devices.append(info)

    if available_devices:
        best_info = max(available_devices, key=lambda x: (x[2], x[3]))
        device = best_info[0]
        dtype = best_info[1]

# 如果没有 CUDA，尝试 Intel XPU (Arc GPU)
if device is None and hasattr(torch, "xpu") and torch.xpu.is_available():
    xpu_info = get_xpu_device_info(0)
    if xpu_info is not None:
        device = xpu_info[0]
        dtype = xpu_info[1]

# 如果没有 CUDA，尝试 MPS (Apple Silicon)
if device is None:
    mps_info = get_mps_device_info()
    if mps_info is not None:
        device = mps_info[0]
        dtype = torch.float32  # MPS 使用 float32

# 如果没有可用的 GPU，使用 CPU
if device is None:
    device = torch.device("cpu")
    dtype = torch.float32  # CPU 使用 float32


class Config:
    def __init__(self):
        self.dtype = dtype
        self.device = device

        self.use_flash_attn = True

        self.gpt_cache = None
        self.sovits_cache = None

        self.cnroberta = None


class GlobalConfig:
    def __init__(self):
        self.models_dir = None

        self.use_jieba_fast = None

        self.chinese_g2p = None
        self.japanese_g2p = None
        self.english_g2p = None

global_config = GlobalConfig()

SDPBACKEND = choose_attention_backend()