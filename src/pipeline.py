import os
import sys
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from opencc import OpenCC
from .parser import BoreholeLogParser
from .cleaner import OllamaBoreholeCleaner

# Thread-local cache for parsers and cleaners
_thread_local_parser = None
_thread_local_cleaner = None

def get_thread_parser():
    global _thread_local_parser
    if _thread_local_parser is None:
        _thread_local_parser = BoreholeLogParser()
    return _thread_local_parser

def get_thread_cleaner(model_name="qwen3:8b"):
    global _thread_local_cleaner
    if _thread_local_cleaner is None:
        _thread_local_cleaner = OllamaBoreholeCleaner(model_name=model_name)
    return _thread_local_cleaner

def process_single_image(
    img_file: Path,
    max_retries: int = 2,
    use_llm: bool = True,
    model_name: str = "qwen3:8b"
) -> tuple[str, bool, dict, str, str]:
    """
    Process a single borehole log image with adaptive retries, validation, and fast-path/LLM cleaning.
    Returns: (file_name, success, data_or_error_dict, mode_message, error_message)
    """
    parser = get_thread_parser()
    cleaner = get_thread_cleaner(model_name=model_name)
    cc = OpenCC('s2twp')

    last_error_msg = ""
    current_data = None
    total_attempts = 1 + max_retries

    for attempt in range(1, total_attempts + 1):
        try:
            # 1. OCR & Rule extraction with attempt-specific preprocessing
            raw_result = parser.parse_borehole(str(img_file), attempt=attempt)
            
            # 2. Validation
            is_valid, val_msg = parser.validate_extraction(raw_result)
            if not is_valid:
                last_error_msg = f"OCR 階段驗證未通過 ({val_msg})"
                current_data = raw_result
                continue

            # 3. Clean with Fast-Path / Ollama Token-Minimized Cleaner
            cleaned_result, clean_mode = cleaner.clean_data(raw_result, use_llm=use_llm)

            # Final traditional Chinese normalization
            final_str = cc.convert(json.dumps(cleaned_result, ensure_ascii=False))
            final_str = final_str.replace("頁巖", "頁岩").replace("砂巖", "砂岩").replace("泥巖", "泥岩").replace("塊巖", "塊岩")
            final_data = json.loads(final_str)

            # 4. Post-clean validation
            is_valid_final, val_final_msg = parser.validate_extraction(final_data)
            if is_valid_final:
                mode_desc = f"嘗試 {attempt} 次 | {clean_mode}"
                return (img_file.name, True, final_data, mode_desc, "")
            else:
                last_error_msg = f"校對後驗證未通過 ({val_final_msg})"
                current_data = final_data

        except Exception as e:
            last_error_msg = f"執行異常: {str(e)}"
            continue

    # If all attempts failed
    error_payload = {
        "file_name": img_file.name,
        "file_path": str(img_file),
        "attempts_made": total_attempts,
        "error_reason": last_error_msg,
        "raw_extracted_data": current_data,
        "suggested_action": "請使用者手動驗證與補正此圖檔之層位與描述"
    }
    return (img_file.name, False, error_payload, "異常失敗", last_error_msg)

def run_pipeline(
    input_dir: str = "input",
    output_file: str = "output/borehole_final_results.json",
    error_file: str = "output/error.json",
    model_name: str = "qwen3:8b",
    use_llm: bool = True,
    max_retries: int = 2,
    max_workers: int = None
):
    input_path = Path(input_dir)
    output_path = Path(output_file)
    error_path = Path(error_file)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)

    if max_workers is None:
        max_workers = min(4, os.cpu_count() or 1)

    print("=" * 75)
    print("🚀 啟動 GeoLogParser 高效能萃取與繁體校對 Pipeline (方案一、二、五優化版)")
    print(f"📁 輸入目錄: {input_path}")
    print(f"📄 成功輸出: {output_path}")
    print(f"⚠️ 異常記錄: {error_path}")
    print(f"🤖 語言模型: {model_name if use_llm else '本地 OpenCC + 規則'}")
    print(f"⚡ 平行工作緒 (Workers): {max_workers} 執行緒")
    print(f"🔄 失敗重試上限: {max_retries} 次 (最多共 {max_retries + 1} 次嘗試)")
    print("=" * 75)

    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if not input_path.exists():
        print(f"[!] 找不到輸入目錄: {input_path}")
        return

    image_files = sorted([
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in valid_exts and not f.name.startswith("cleaned")
    ])

    if not image_files:
        print(f"[!] 在目錄 {input_path} 中未找到任何鑽孔圖片檔案。")
        return

    print(f"[*] 找到 {len(image_files)} 個鑽孔柱狀圖檔案，開始平行批次處理...\n")

    start_time = time.time()
    final_results = {}
    error_records = {}

    # Parallel Execution (Solution 2)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(
                process_single_image,
                img_file,
                max_retries,
                use_llm,
                model_name
            ): img_file for img_file in image_files
        }

        completed_count = 0
        for future in as_completed(future_to_file):
            completed_count += 1
            file_name, success, payload, mode_desc, error_msg = future.result()

            if success:
                final_results[file_name] = payload
                print(f"[{completed_count}/{len(image_files)}] ✅ {file_name} -> {mode_desc}")
            else:
                error_records[file_name] = payload
                print(f"[{completed_count}/{len(image_files)}] ❌ {file_name} -> {error_msg}")

    total_elapsed = time.time() - start_time
    avg_time = total_elapsed / len(image_files) if image_files else 0

    # Save successful results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    # Save or update error.json
    if error_records:
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(error_records, f, indent=2, ensure_ascii=False)
        print(f"\n⚠️ 發現 {len(error_records)} 筆超出預期之檔案，已寫入手動驗證檔: {error_path}")
    else:
        if error_path.exists():
            error_path.unlink()
        print("\n🎉 所有檔案均成功通過地質岩性驗證，無任何異常！")

    print("\n" + "=" * 75)
    print(f"✅ 全流程完成！")
    print(f"⏱️ 總耗時: {total_elapsed:.2f} 秒 (平均每張圖: {avg_time:.2f} 秒)")
    print(f"📊 成功筆數: {len(final_results)} / 異常待檢筆數: {len(error_records)}")
    print(f"📄 成果檔案: {output_path}")
    if error_records:
        print(f"⚠️ 異常記錄: {error_path}")
    print("=" * 75)
