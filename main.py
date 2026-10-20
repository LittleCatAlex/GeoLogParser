import sys
import os
import argparse

# Ensure UTF-8 console output for Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Dependency Checker
REQUIRED_PACKAGES = {
    "cv2": "opencv-python>=4.8.0",
    "rapidocr_onnxruntime": "rapidocr-onnxruntime>=1.4.0",
    "opencc": "opencc-python-reimplemented>=0.1.7",
    "requests": "requests>=2.31.0",
    "numpy": "numpy>=1.24.0",
    "PIL": "pillow>=10.0.0"
}

missing_packages = []
for mod, pkg_spec in REQUIRED_PACKAGES.items():
    try:
        __import__(mod)
    except ImportError:
        missing_packages.append(pkg_spec)

if missing_packages:
    print("\n" + "=" * 70)
    print("❌ 【執行中斷】偵測到缺少必要的 Python 套件！")
    print("=" * 70)
    print("缺少套件清單：")
    for p in missing_packages:
        print(f"  • {p}")
    print("\n💡 請依照以下方式之一進行安裝：")
    print("----------------------------------------------------------------------")
    print("【方法 1】使用 uv 自動執行（推薦，最快速且自動處理虛擬環境）：")
    print("   uv run main.py")
    print("\n【方法 2】使用 uv 安裝至目前虛擬環境：")
    print(f"   uv pip install {' '.join(missing_packages)}")
    print("\n【方法 3】使用標準 pip 安裝：")
    print(f"   pip install {' '.join(missing_packages)}")
    print("=" * 70 + "\n")
    sys.exit(1)

from src.pipeline import run_pipeline
from src.device import get_gpu_hardware_info, get_onnxruntime_providers

def main():
    parser = argparse.ArgumentParser(
        description="GeoLogParser - 離線地質鑽孔 OCR 與 Ollama 繁體校對管線 (支援 RTX 5090 GPU 加速與斷點續傳)"
    )
    parser.add_argument("--input", "-i", default="input", help="輸入圖片目錄路徑 (預設: input)")
    parser.add_argument("--output", "-o", default="output/borehole_final_results.json", help="輸出 JSON 檔案路徑 (預設: output/borehole_final_results.json)")
    parser.add_argument("--error", "-e", default="output/error.json", help="異常待手動驗證 JSON 檔案路徑 (預設: output/error.json)")
    parser.add_argument("--model", "-m", default="qwen3:8b", help="Ollama 模型名稱 (預設: qwen3:8b)")
    parser.add_argument("--no-llm", action="store_true", help="僅使用本地 OpenCC 與規則，不呼叫 Ollama")
    parser.add_argument("--retries", "-r", type=int, default=2, help="辨識異常時之最大重試次數 (預設: 2)")
    parser.add_argument("--workers", "-w", type=int, default=min(4, os.cpu_count() or 1), help="平行處理工作執行緒數量 (預設: 自動偵測 CPU 核心數，上限 4)")
    parser.add_argument("--device", "-d", choices=["auto", "cuda", "dml", "cpu"], default="auto", help="運算硬體裝置 (預設: auto 自動偵測 GPU/CUDA，支援 cuda, dml, cpu；具備 RTX 5090 Blackwell sm_120 相容防護)")
    parser.add_argument("--overwrite", action="store_true", help="強制重新處理所有檔案，不使用斷點續傳 (預設會自動接續歷史進度)")
    parser.add_argument("--no-resume", action="store_true", help="關閉斷點續傳功能 (等同 --overwrite)")
    parser.add_argument("--retry-errors", action="store_true", help="重新嘗試之前失敗於 error.json 中的圖檔")
    
    args = parser.parse_args()

    # If user explicitly requested CUDA, check if onnxruntime-gpu is available
    if args.device == "cuda":
        providers = get_onnxruntime_providers()
        if "CUDAExecutionProvider" not in providers:
            gpu_info = get_gpu_hardware_info()
            print("\n" + "=" * 72)
            print("⚠️  【硬體提示】已指定 --device cuda，但目前環境尚未安裝 onnxruntime-gpu！")
            print("=" * 72)
            if gpu_info["has_nvidia_gpu"]:
                print(f"偵測到實體顯示卡 : {gpu_info['name']} (顯存: {gpu_info['vram_mb']} MB)")
                if gpu_info.get("is_blackwell_rtx50"):
                    print(f"架構代號         : {gpu_info['arch_note']} (RTX 50 系列 sm_120)")
            print("\n💡 啟用 GPU (CUDA) 加速安裝步驟：")
            print("   1. 卸載 CPU 版本：pip uninstall -y onnxruntime")
            print("   2. 安裝 GPU 版本：pip install onnxruntime-gpu")
            print("   (若為 RTX 5090 等 Blackwell 架構，建議配合 CUDA 12.8+ 驅動使用)")
            print("   (在 Windows 上亦可直接使用 --device dml 啟用 DirectML 加速)")
            print("=" * 72 + "\n")

    should_resume = not (args.overwrite or args.no_resume)

    run_pipeline(
        input_dir=args.input,
        output_file=args.output,
        error_file=args.error,
        model_name=args.model,
        use_llm=not args.no_llm,
        max_retries=args.retries,
        max_workers=args.workers,
        device=args.device,
        resume=should_resume,
        overwrite=args.overwrite or args.no_resume,
        retry_errors=args.retry_errors
    )

if __name__ == "__main__":
    main()
