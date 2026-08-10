import os
import sys
import json
import argparse
from pathlib import Path
from opencc import OpenCC
from .parser import BoreholeLogParser
from .cleaner import OllamaBoreholeCleaner

def run_pipeline(
    input_dir: str = "input",
    output_file: str = "output/borehole_final_results.json",
    error_file: str = "output/error.json",
    model_name: str = "qwen3:8b",
    use_llm: bool = True,
    max_retries: int = 2
):
    input_path = Path(input_dir)
    output_path = Path(output_file)
    error_path = Path(error_file)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("🚀 啟動地質鑽孔柱狀圖萃取與繁體校對 Pipeline")
    print(f"📁 輸入目錄: {input_path}")
    print(f"📄 成功輸出: {output_path}")
    print(f"⚠️ 異常記錄: {error_path}")
    print(f"🤖 語言模型: {model_name if use_llm else '本地 OpenCC + 規則'}")
    print(f"🔄 失敗重試上限: {max_retries} 次 (最多共 {max_retries + 1} 次嘗試)")
    print("=" * 75)

    parser = BoreholeLogParser()
    cleaner = OllamaBoreholeCleaner(model_name=model_name)
    cc = OpenCC('s2twp')

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

    print(f"[*] 找到 {len(image_files)} 個鑽孔柱狀圖檔案，開始處理...\n")

    final_results = {}
    error_records = {}

    for idx, img_file in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] 正在處理: {img_file.name}")
        success = False
        last_error_msg = ""
        current_data = None

        # Try up to (1 + max_retries) times
        total_attempts = 1 + max_retries
        for attempt in range(1, total_attempts + 1):
            if attempt > 1:
                print(f"    ⚠️ [重試 {attempt-1}/{max_retries}] 啟用第 {attempt} 階段自適應增強前處理...")

            try:
                # 1. OCR & Rule extraction with attempt-specific preprocessing
                raw_result = parser.parse_borehole(str(img_file), attempt=attempt)
                
                # 2. Pre-clean & validate
                is_valid, val_msg = parser.validate_extraction(raw_result)
                if not is_valid:
                    last_error_msg = f"OCR 階段驗證失敗: {val_msg}"
                    current_data = raw_result
                    continue

                # 3. Clean with LLM or OpenCC (Strictly no hallucination)
                if use_llm:
                    cleaned_result = cleaner.clean_with_llm(raw_result)
                else:
                    cleaned_result = cleaner.local_preclean(raw_result)

                # Final traditional Chinese normalization
                final_str = cc.convert(json.dumps(cleaned_result, ensure_ascii=False))
                final_str = final_str.replace("頁巖", "頁岩").replace("砂巖", "砂岩").replace("泥巖", "泥岩").replace("塊巖", "塊岩")
                final_data = json.loads(final_str)

                # 4. Post-clean validation
                is_valid_final, val_final_msg = parser.validate_extraction(final_data)
                if is_valid_final:
                    final_results[img_file.name] = final_data
                    success = True
                    print(f"    ✅ 成功解析 (嘗試 {attempt} 次):")
                    print(json.dumps(final_data, indent=2, ensure_ascii=False))
                    break
                else:
                    last_error_msg = f"校對後驗證失敗: {val_final_msg}"
                    current_data = final_data

            except Exception as e:
                last_error_msg = f"執行異常: {str(e)}"
                continue

        if not success:
            print(f"    ❌ 重試 {max_retries} 次後依然超出預期錯誤: {last_error_msg}")
            error_records[img_file.name] = {
                "file_name": img_file.name,
                "file_path": str(img_file),
                "attempts_made": total_attempts,
                "error_reason": last_error_msg,
                "raw_extracted_data": current_data,
                "suggested_action": "請使用者手動驗證與補正此圖檔之層位與描述"
            }

    # Save successful results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    # Save or update error.json
    if error_records:
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(error_records, f, indent=2, ensure_ascii=False)
        print(f"\n⚠️ 發現 {len(error_records)} 筆超出預期之檔案，已寫入手動驗證檔: {error_path}")
    else:
        # If all succeeded, write empty error report or clean old one
        if error_path.exists():
            error_path.unlink()
        print("\n🎉 所有檔案均成功通過地質岩性驗證，無任何異常！")

    print("\n" + "=" * 75)
    print(f"✅ 全流程完成！成功筆數: {len(final_results)} / 異常待檢筆數: {len(error_records)}")
    print(f"📄 成功成果檔案: {output_path}")
    if error_records:
        print(f"⚠️ 異常記錄檔案: {error_path}")
    print("=" * 75)
