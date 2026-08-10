import sys

# Ensure UTF-8 console output for Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

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

import argparse
from src.pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="地質鑽孔柱狀圖離線 OCR 與 Ollama (qwen3:8b) 繁體化校對管線")
    parser.add_argument("--input", "-i", default="input", help="輸入圖片目錄路徑 (預設: input)")
    parser.add_argument("--output", "-o", default="output/borehole_final_results.json", help="輸出 JSON 檔案路徑 (預設: output/borehole_final_results.json)")
    parser.add_argument("--error", "-e", default="output/error.json", help="異常待手動驗證 JSON 檔案路徑 (預設: output/error.json)")
    parser.add_argument("--model", "-m", default="qwen3:8b", help="Ollama 模型名稱 (預設: qwen3:8b)")
    parser.add_argument("--no-llm", action="store_true", help="僅使用本地 OpenCC 與規則，不呼叫 Ollama")
    parser.add_argument("--retries", "-r", type=int, default=2, help="辨識異常時之最大重試次數 (預設: 2)")
    
    args = parser.parse_args()
    run_pipeline(
        input_dir=args.input,
        output_file=args.output,
        error_file=args.error,
        model_name=args.model,
        use_llm=not args.no_llm,
        max_retries=args.retries
    )

if __name__ == "__main__":
    main()
