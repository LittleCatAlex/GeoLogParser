# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2026-10-21
### Fixed
- **地質岩性判定過度嚴格問題修正 (大幅降低誤判率)**:
  - 擴充英文地質關鍵字 (`clay`, `sand`, `silt`, `gravel`, `shale`, `rock`, `fill`, `dense`, `stiff`, `brown`, `gray`, `yellow` 等) 與全格式 USCS 分類代碼，徹底解決英文與雙語鑽孔柱狀圖全數誤判為異常之問題。
  - 擴充繁體中文工程地質詞庫與常見 OCR 訛字字典（新增 `剪裂帶`、`斷層`、`破碎帶`、`煤層`、`煤碳`、`變質砂岩`、`硬頁岩`、`臭巖` $\to$ `頁岩`、`漫質砂岩` $\to$ `變質砂岩`、`膠結尚可` 等）。
  - 新增浮水印與頁尾雜訊自動過濾機制（去除 `地質探勘資料庫系統`、`中央地質調查所` 等浮水印，避免浮水印被錯誤辨識為地層）。
  - 強化地層解析之深度單調性防護（Monotonic Depth Enforcement），自動忽略非遞增之內部量測數值（如層面傾角、百分比）。
  - 自動合併或略過空白地層，避免因表格底部邊界線產生無描述之地層而導致整張圖檔驗證失敗。
  - 調整驗證容錯邏輯：地質鑽孔中只要存在有效岩性特徵且有效層數比例正常，即判定為通過，不再因單一註記或次要修飾詞而全盤捨棄整張柱狀圖。
- **歷史成果資料庫修復與復原**:
  - 重新評估並過濾 `output/error.json`，成功復原 459 筆原本被誤判之鑽孔成果並合併回 `output/borehole_final_results.json`（成功總數由 394 筆大幅提升至 853 筆，異常清單大幅縮減 71%）。

---

## [0.3.0] - 2026-10-20
### Added
- **GPU 硬體加速與 RTX 5090 (Blackwell 架構) 相容防護** (`src/device.py`):
  - 自動偵測系統實體 GPU、顯存容量、驅動版本與架構世代（支援 RTX 5090 / 5080 sm_120、Ada Lovelace sm_89、Ampere sm_86 等）。
  - 自動查詢並調度 ONNX Runtime Execution Providers (`CUDAExecutionProvider`, `DmlExecutionProvider`, `CPUExecutionProvider`)。
  - 專為 NVIDIA GeForce RTX 5090 設計的 CUDA 核心預編譯檢測與熱機測試（Warm-up test）；若偵測到當前 `onnxruntime-gpu` 缺少 sm_120 核心二進位檔，系統會印出詳細診斷指引並自動安全平滑回退至 CPU 或 DirectML，確保辨識工作零中斷。
  - 支援 Windows DirectML GPU 加速模式 (`--device dml`)，無須依賴 CUDA Toolkit 即可直接調用 DirectX 12 顯卡運算。
  - 提供啟動時的硬體診斷橫幅 (`print_hardware_banner`)。
- **斷點續傳與即時原子存檔機制 (Checkpoint & Resume)** (`src/pipeline.py`):
  - 啟動時自動掃描現有 `output/borehole_final_results.json`，精準過濾並略過已辨識完成之圖檔，避免重複運算。
  - 實現多執行緒即時原子磁碟寫入（Atomic Disk Persistence）：每完成一張圖檔立即透過臨時檔置換（`.tmp -> .json`）同步寫入磁碟，避免因程式當機、斷電或手動中斷造成 JSON 損壞。
  - 優雅處理 `KeyboardInterrupt`（Ctrl+C）訊號，在使用者中斷時自動取消未派發任務並將已完成之成果安全刷入磁碟。
- **全新 CLI 命令列參數** (`main.py`):
  - `--device` / `-d`：指定運算裝置 (`auto`, `cuda`, `dml`, `cpu`，預設: `auto`)。
  - `--overwrite`：強制重新處理所有檔案，略過斷點續傳快取。
  - `--no-resume`：停用斷點續傳功能。
  - `--retry-errors`：重新嘗試先前記錄於 `error.json` 之異常待檢圖檔。
- **套件依賴配置** (`pyproject.toml`):
  - 加入 `[project.optional-dependencies]` 之 `gpu` 套件群組 (`onnxruntime-gpu>=1.18.0`)。

### Changed
- 將管線內部共享的解析器變更為標準 Thread-local 隔離 (`threading.local`)，確保多執行緒平行 OCR 時執行個體安全獨立。
- 強化 Windows 終端機控制台在 CP950 編碼下的 UTF-8 安全輸出 (`safe_print`)，杜絕 UnicodeEncodeError 異常。

---

## [0.2.0] - 2026-10-11
### Added
- 多階段自適應前處理增強重試機制（最多重試 2 次）。
- Fast-Path 智慧快篩與輕量 Token 校對管線。
- 地質岩性嚴格驗證引擎與 `output/error.json` 自動分流。
- 台灣 22 縣市 368 鄉鎮區真實地名資料庫對照。
