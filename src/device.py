import os
import sys
import shutil
import subprocess
from typing import Dict, Any, List, Tuple, Optional

def get_gpu_hardware_info() -> Dict[str, Any]:
    """
    Query system GPU hardware details via nvidia-smi or torch.
    Returns:
        dict: {
            "has_nvidia_gpu": bool,
            "name": str,
            "driver_version": str,
            "vram_mb": int,
            "is_blackwell_rtx50": bool,
            "arch_note": str
        }
    """
    info = {
        "has_nvidia_gpu": False,
        "name": "None",
        "driver_version": "Unknown",
        "vram_mb": 0,
        "is_blackwell_rtx50": False,
        "arch_note": ""
    }

    # 1. Try querying via nvidia-smi
    if shutil.which("nvidia-smi"):
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=3
            )
            if res.returncode == 0 and res.stdout.strip():
                lines = res.stdout.strip().splitlines()
                first_gpu = lines[0].split(",")
                if len(first_gpu) >= 3:
                    info["has_nvidia_gpu"] = True
                    info["name"] = first_gpu[0].strip()
                    info["driver_version"] = first_gpu[1].strip()
                    try:
                        info["vram_mb"] = int(float(first_gpu[2].strip()))
                    except ValueError:
                        info["vram_mb"] = 0
        except Exception:
            pass

    # 2. Check if PyTorch can provide CUDA info as fallback
    if not info["has_nvidia_gpu"]:
        try:
            import torch
            if torch.cuda.is_available():
                info["has_nvidia_gpu"] = True
                info["name"] = torch.cuda.get_device_name(0)
                try:
                    info["vram_mb"] = int(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
                except Exception:
                    pass
        except Exception:
            pass

    # 3. Detect Blackwell / RTX 50 series (sm_120) and other architectures
    name_upper = info["name"].upper()
    if any(k in name_upper for k in ["5090", "5080", "5070", "5060", "BLACKWELL", "B200", "B100"]):
        info["is_blackwell_rtx50"] = True
        info["arch_note"] = "NVIDIA Blackwell 架構 (sm_120 / sm_121)"
    elif any(k in name_upper for k in ["4090", "4080", "4070", "4060", "ADA"]):
        info["arch_note"] = "NVIDIA Ada Lovelace 架構 (sm_89)"
    elif any(k in name_upper for k in ["3090", "3080", "3070", "3060", "AMPERE"]):
        info["arch_note"] = "NVIDIA Ampere 架構 (sm_86)"
    elif any(k in name_upper for k in ["2080", "2070", "2060", "TURING"]):
        info["arch_note"] = "NVIDIA Turing 架構 (sm_75)"
    elif info["has_nvidia_gpu"]:
        info["arch_note"] = "NVIDIA GPU"

    return info

def get_onnxruntime_providers() -> List[str]:
    """Return available ONNX Runtime execution providers."""
    try:
        import onnxruntime as ort
        return ort.get_available_providers()
    except Exception:
        return []

def resolve_device(device_choice: str = "auto") -> Tuple[str, str]:
    """
    Resolve requested device choice ('auto', 'cuda', 'dml', 'cpu')
    Returns:
        tuple[resolved_device ('cuda'|'dml'|'cpu'), description_string]
    """
    choice = (device_choice or "auto").strip().lower()
    gpu_info = get_gpu_hardware_info()
    ort_providers = get_onnxruntime_providers()
    has_cuda_ep = "CUDAExecutionProvider" in ort_providers
    has_dml_ep = "DmlExecutionProvider" in ort_providers

    if choice == "cuda":
        if not has_cuda_ep:
            return "cuda", (
                f"⚠️ 指定使用 CUDA，但 onnxruntime 尚未偵測到 CUDAExecutionProvider "
                f"(目前可用 Provider: {ort_providers})。若為 RTX 5090 等顯卡，請安裝 onnxruntime-gpu。"
            )
        desc = f"NVIDIA GPU: {gpu_info['name']} (CUDA 加速 / {gpu_info['arch_note'] or 'sm_compat'})"
        return "cuda", desc

    elif choice == "dml":
        if not has_dml_ep:
            return "dml", (
                f"⚠️ 指定使用 DirectML，但 onnxruntime 尚未偵測到 DmlExecutionProvider "
                f"(目前可用: {ort_providers})。"
            )
        desc = f"DirectML GPU 加速 (DmlExecutionProvider - 支援 Windows DirectX 12 顯卡)"
        return "dml", desc

    elif choice == "cpu":
        return "cpu", "CPU 模式 (未啟用硬體加速)"

    else:  # "auto"
        if has_cuda_ep and gpu_info["has_nvidia_gpu"]:
            desc = f"NVIDIA GPU: {gpu_info['name']} (CUDA 自動加速 / {gpu_info['arch_note'] or 'CUDA EP'})"
            return "cuda", desc
        elif has_dml_ep:
            desc = f"DirectML GPU 自動加速 ({gpu_info['name'] if gpu_info['has_nvidia_gpu'] else 'DirectX 12'})"
            return "dml", desc
        elif gpu_info["has_nvidia_gpu"]:
            desc = (
                f"CPU 模式 (偵測到實體顯卡 {gpu_info['name']}，但環境中為 CPU 版 onnxruntime；"
                f"若需在 RTX 5090 上啟用 GPU 加速，請執行: pip uninstall onnxruntime && pip install onnxruntime-gpu)"
            )
            return "cpu", desc
        else:
            return "cpu", "CPU 模式 (未偵測到相容 GPU，使用 CPU 多核心平行運算)"

def _safe_print(text: str):
    """Safely print UTF-8 strings even on Windows CP950/traditional terminals."""
    try:
        print(text)
    except UnicodeEncodeError:
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
                print(text)
            else:
                encoded = text.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8")
                print(encoded)
        except Exception:
            clean_text = text.encode("ascii", errors="replace").decode("ascii")
            print(clean_text)

def create_rapidocr_engine(device: str = "auto", device_id: int = 0):
    """
    Factory function to instantiate RapidOCR with device awareness and fallback protection.
    Ensures complete, crash-free execution on RTX 5090 (Blackwell sm_120) and other GPUs.
    """
    from rapidocr_onnxruntime import RapidOCR
    import numpy as np

    resolved_device, _ = resolve_device(device)
    gpu_info = get_gpu_hardware_info()

    if resolved_device == "cuda":
        try:
            engine = RapidOCR(
                det_use_cuda=True,
                cls_use_cuda=True,
                rec_use_cuda=True
            )

            # Warm-up test using a small dummy image to ensure CUDA kernel compilation & sm_120 compatibility
            dummy_img = np.zeros((48, 128, 3), dtype=np.uint8)
            engine(dummy_img)
            return engine

        except Exception as e:
            err_msg = str(e)
            _safe_print(f"\n⚠️  [GPU 警告] CUDA 加速初始化或核心執行失敗: {err_msg}")
            if gpu_info.get("is_blackwell_rtx50") or "5090" in gpu_info.get("name", ""):
                _safe_print("💡 系統偵測到 NVIDIA GeForce RTX 5090 (Blackwell 架構 / sm_120)：")
                _safe_print("   • 原因分析：目前安裝的 onnxruntime-gpu 二進位檔未包含 sm_120 預編譯核心，或 CUDA 版本未相容。")
                _safe_print("   • 解決方案 1：安裝支援 CUDA 12.8+ / Blackwell 之最新 onnxruntime-gpu。")
                _safe_print("   • 解決方案 2：在 Windows 上改用 DirectML 加速：--device dml")
            _safe_print("🛡️  系統已自動安全回退至 CPU 執行，確保辨識任務順利完成！\n")
            return RapidOCR()

    elif resolved_device == "dml":
        try:
            engine = RapidOCR(
                det_use_dml=True,
                cls_use_dml=True,
                rec_use_dml=True
            )
            dummy_img = np.zeros((48, 128, 3), dtype=np.uint8)
            engine(dummy_img)
            return engine
        except Exception as e:
            _safe_print(f"\n⚠️  [DirectML 警告] 初始化失敗: {e}，自動回退至 CPU 執行。")
            return RapidOCR()

    else:
        return RapidOCR()

def print_hardware_banner(device: str = "auto"):
    """Display clean hardware diagnostics banner."""
    resolved, desc = resolve_device(device)
    gpu_info = get_gpu_hardware_info()
    providers = get_onnxruntime_providers()

    _safe_print("=" * 75)
    _safe_print("[GeoLogParser] 硬體加速診斷資訊：")
    _safe_print(f"   • 目標運算裝置 : {device.upper()} -> 解析為 [{resolved.upper()}]")
    _safe_print(f"   • 運行裝置配置 : {desc}")
    if gpu_info["has_nvidia_gpu"]:
        vram_str = f"{gpu_info['vram_mb'] / 1024:.1f} GB" if gpu_info['vram_mb'] > 0 else "未知"
        _safe_print(f"   • 偵測實體顯卡 : {gpu_info['name']} (顯存: {vram_str}, 驅動: {gpu_info['driver_version']})")
        if gpu_info["arch_note"]:
            _safe_print(f"   • 架構資訊     : {gpu_info['arch_note']}")
    _safe_print(f"   • ONNX Providers : {', '.join(providers) if providers else '未安裝 onnxruntime'}")
    _safe_print("=" * 75)
