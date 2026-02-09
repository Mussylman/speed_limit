# config.py
# Paths, constants, config loading

import os
import sys
import yaml

# Headless mode (Docker without display)
HEADLESS = os.environ.get("HEADLESS", "0") == "1" or os.environ.get("DISPLAY_OFF", "0") == "1"

# Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ensure src/ and nomeroff-net/ are on sys.path
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "nomeroff-net"))

# Config file paths
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")
CONFIG_CAM_PATH = os.path.join(BASE_DIR, "config", "config_cam.yaml")


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_configs() -> tuple:
    """Load both config.yaml and config_cam.yaml. Returns (cfg, cam_cfg)."""
    cfg = load_yaml(CONFIG_PATH)
    cam_cfg = load_yaml(CONFIG_CAM_PATH)
    return cfg, cam_cfg


def print_gpu_info():
    """Print GPU information."""
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("CUDA not available, using CPU")
