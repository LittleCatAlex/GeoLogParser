import os
import sys
import json
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from opencc import OpenCC

from .parser import BoreholeLogParser
from .cleaner import OllamaBoreholeCleaner
from .device import print_hardware_banner, resolve_device

# Ensure UTF-8 console output for Windows terminals
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

_builtin_print = print

def safe_print(*args, **kwargs):
    """Safely print text handling Windows CP950/traditional console encoding."""
    try:
        _builtin_print(*args, **kwargs)
    except UnicodeEncodeError:
        try:
            sep = kwargs.get("sep", " ")
            end = kwargs.get("end", "\n")
            text = sep.join(str(a) for a in args) + end
            sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
            sys.stdout.buffer.flush()
        except Exception:
            clean_args = [str(a).encode("ascii", errors="replace").decode("ascii") for a in args]
            _builtin_print(*clean_args, **kwargs)

# Real Thread-local storage for parsers and cleaners
_thread_local = threading.local()

def get_thread_parser(device: str = "auto") -> BoreholeLogParser:
    if not hasattr(_thread_local, "parser") or _thread_local.parser is None:
        _thread_local.parser = BoreholeLogParser(device=device)
    return _thread_local.parser

def get_thread_cleaner(model_name: str = "qwen3:8b") -> OllamaBoreholeCleaner:
    if not hasattr(_thread_local, "cleaner") or _thread_local.cleaner is None:
        _thread_local.cleaner = OllamaBoreholeCleaner(model_name=model_name)
    return _thread_local.cleaner

def atomic_save_json(target_path: Path, data: dict):
    """
    Write JSON atomically via temporary file replacement.
    Prevents corrupt or half-written JSON files if interrupted (Ctrl+C, crash, power loss).
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(f"{target_path.suffix}.tmp")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        temp_path.replace(target_path)
    except Exception:
        # Fallback direct write if atomic replacement fails
        try:
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass

def process_single_image(
    img_file: Path,
    max_retries: int = 2,
    use_llm: bool = True,
    model_name: str = "qwen3:8b",
    device: str = "auto"
) -> tuple[str, bool, dict, str, str]:
    """
    Process a single borehole log image with adaptive retries, validation, and fast-path/LLM cleaning.
    Returns: (file_name, success, data_or_error_dict, mode_message, error_message)
    """
    parser = get_thread_parser(device=device)
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
            final_str = (
                final_str.replace("頁巖", "頁岩")
                .replace("砂巖", "砂岩")
                .replace("泥巖", "泥岩")
                .replace("塊巖", "塊岩")
            )
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
    max_workers: int = None,
    device: str = "auto",
    resume: bool = True,
    overwrite: bool = False,
    retry_errors: bool = False
):
    """
    Run borehole extraction pipeline with:
    - Automatic hardware acceleration (GPU / CUDA / RTX 5090 Blackwell / DML / CPU)
    - Full checkpointing & resume capabilities (斷點續傳)
    - Atomic thread-safe disk persistence
    """
    input_path = Path(input_dir)
    output_path = Path(output_file)
    error_path = Path(error_file)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)

    if max_workers is None:
        max_workers = min(4, os.cpu_count() or 1)

    # 1. Print hardware diagnostics
    print_hardware_banner(device=device)

    safe_print("=" * 75)
    safe_print("啟動 GeoLogParser 管線")
    safe_print(f"輸入目錄     : {input_path}")
    safe_print(f"成功輸出路徑 : {output_path}")
    safe_print(f"異常記錄路徑 : {error_path}")
    safe_print(f"運算裝置模式 : {device.upper()}")
    safe_print(f"語言模型     : {model_name if use_llm else '本地 OpenCC + 規則'}")
    safe_print(f"平行工作緒   : {max_workers} 執行緒")
    safe_print(f"重試嘗試上限 : 最多共 {max_retries + 1} 次嘗試")
    safe_print(f"斷點續傳     : {'關閉 (強制覆寫)' if overwrite else ('開啟' if resume else '關閉')}")
    if retry_errors:
        safe_print(f"異常重試     : 開啟 (將重新處理 error.json 中之圖檔)")
    safe_print("=" * 75)

    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if not input_path.exists():
        safe_print(f"[!] 找不到輸入目錄: {input_path}")
        return

    image_files = sorted([
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in valid_exts and not f.name.startswith("cleaned")
    ])

    if not image_files:
        safe_print(f"[!] 在目錄 {input_path} 中未找到任何鑽孔圖片檔案。")
        return

    # 2. Checkpoint & Resume: Load existing progress
    final_results = {}
    error_records = {}

    if not overwrite and resume:
        if output_path.exists():
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        final_results = json.loads(content)
            except Exception as e:
                safe_print(f"[!] 讀取歷史成功進度時發生問題 ({e})，將重新累積。")
                final_results = {}

        if error_path.exists():
            try:
                with open(error_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        error_records = json.loads(content)
            except Exception:
                error_records = {}

    # Filter files that need processing
    pending_files = []
    already_success_count = 0
    already_error_count = 0

    for img in image_files:
        name = img.name
        if name in final_results:
            already_success_count += 1
            continue
        if name in error_records:
            if retry_errors:
                pending_files.append(img)
            else:
                already_error_count += 1
                continue
        else:
            pending_files.append(img)

    total_images = len(image_files)

    if not pending_files:
        safe_print(f"\n🎉 斷點續傳檢查完畢：所有 {total_images} 個檔案均已處理完成！無需重複運算。")
        safe_print(f"   • 已成功筆數: {len(final_results)}")
        if error_records:
            safe_print(f"   • 異常待檢筆數: {len(error_records)} (可使用 --retry-errors 重新嘗試)")
        safe_print(f"   （如需徹底重新處理所有檔案，請使用 --overwrite 參數）\n")
        return

    if already_success_count > 0 or already_error_count > 0:
        safe_print(f"[*] ⚡ 斷點續傳已生效：")
        safe_print(f"   • 歷史已成功 : {already_success_count} 筆 (自動略過)")
        if already_error_count > 0:
            safe_print(f"   • 歷史待複核 : {already_error_count} 筆 (使用 --retry-errors 可重新嘗試)")
        safe_print(f"   • 本次待執行 : {len(pending_files)} 筆 (總共 {total_images} 筆)\n")
    else:
        safe_print(f"[*] 找到 {len(pending_files)} 個鑽孔柱狀圖檔案，開始平行批次處理...\n")

    start_time = time.time()
    save_lock = threading.Lock()
    completed_batch = 0
    total_batch = len(pending_files)

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(
                    process_single_image,
                    img_file,
                    max_retries,
                    use_llm,
                    model_name,
                    device
                ): img_file for img_file in pending_files
            }

            for future in as_completed(future_to_file):
                completed_batch += 1
                overall_progress = already_success_count + completed_batch
                file_name, success, payload, mode_desc, error_msg = future.result()

                with save_lock:
                    if success:
                        final_results[file_name] = payload
                        if file_name in error_records:
                            del error_records[file_name]
                        safe_print(f"[{overall_progress}/{total_images}] ✅ {file_name} -> {mode_desc}")
                    else:
                        error_records[file_name] = payload
                        safe_print(f"[{overall_progress}/{total_images}] ❌ {file_name} -> {error_msg}")

                    # Real-time atomic checkpoint save (zero progress loss on abort)
                    atomic_save_json(output_path, final_results)
                    if error_records:
                        atomic_save_json(error_path, error_records)
                    elif error_path.exists():
                        try:
                            error_path.unlink()
                        except Exception:
                            pass

    except KeyboardInterrupt:
        safe_print("\n\n⚠️ 收到使用者中斷信號 (Ctrl+C)！正在緊急保存目前進度...")
        with save_lock:
            atomic_save_json(output_path, final_results)
            if error_records:
                atomic_save_json(error_path, error_records)
        safe_print(f"💾 進度已安全寫入磁碟 (目前累積成功: {len(final_results)} 筆，待檢: {len(error_records)} 筆)。")
        safe_print(f"👉 下次啟動程式時，系統將自動接續中斷處繼續執行！\n")
        return

    total_elapsed = time.time() - start_time
    avg_time = total_elapsed / len(pending_files) if pending_files else 0

    # Final disk sync
    atomic_save_json(output_path, final_results)
    if error_records:
        atomic_save_json(error_path, error_records)
        safe_print(f"\n⚠️ 發現 {len(error_records)} 筆超出預期之檔案，已寫入手動驗證檔: {error_path}")
    else:
        if error_path.exists():
            try:
                error_path.unlink()
            except Exception:
                pass
        safe_print("\n🎉 所有檔案均成功通過地質岩性驗證，無任何異常！")

    safe_print("\n" + "=" * 75)
    safe_print("✅ 全流程完成！")
    safe_print(f"⏱️ 本次執行耗時: {total_elapsed:.2f} 秒 (平均每張圖: {avg_time:.2f} 秒)")
    safe_print(f"📊 累積成功筆數: {len(final_results)} / 累積異常筆數: {len(error_records)}")
    safe_print(f"📄 成果檔案: {output_path}")
    if error_records:
        safe_print(f"⚠️ 異常記錄: {error_path}")
    safe_print("=" * 75)
