# GeoLogParser - 地質鑽孔柱狀圖自動化結構化萃取與繁體校對系統

本系統專為台灣工程地質鑽孔柱狀圖設計，實現**全地端／完全離線（On-Premise Offline）**的影像辨識（OCR）、動態欄位解析、地質岩性層位綁定，並結合本機 **Ollama (`qwen3:8b`)** 與台灣真實地名資料庫進行純繁體語意校正與資料完整性驗證。

最新版本已全面支援 **NVIDIA GPU 硬體加速（含 RTX 5090 Blackwell 架構相容保護）** 與 **強固型斷點續傳機制（Checkpoint & Resume）**。

---

## 專案目錄結構

```text
.
├── input/                             # 📥 輸入目錄：存放待辨識的鑽孔柱狀圖 (*.png, *.jpg)
├── output/                            # 📤 輸出目錄：存放處理成果與異常報告
│   ├── borehole_final_results.json    # 檢驗通過之結構化 JSON 成果（即時原子更新）
│   └── error.json                     # (若有) 異常待手動複核之詳細記錄檔
├── data/                              # 📚 參考資料目錄
│   └── taiwan_places.json             # 台灣 22 縣市 368 鄉鎮區真實地名資料庫
├── src/                               # ⚙️ 核心原始碼模組
│   ├── __init__.py
│   ├── device.py                      # 🖥️ 硬體加速管理、RTX 5090 核心偵測與 CPU/DML 平滑回退
│   ├── parser.py                      # 離線 OCR、多階段自適應前處理與地質驗證引擎
│   ├── cleaner.py                     # Ollama qwen3:8b 零腦補繁體校對器
│   ├── geo.py                         # 台灣真實行政區與地名對照模組
│   └── pipeline.py                    # 批次排程、斷點續傳、即時原子寫入與中斷安全保存
├── main.py                            # 🚀 一鍵執行入口（內建套件防呆與硬體診斷）
├── pyproject.toml                     # 📦 專案環境依賴配置 (含可選 GPU 加速群組)
├── CHANGELOG.md                       # 📝 版本演進日誌
└── README.md                          # 📖 專案說明與操作手冊
```

---

## ⚡ 核心新功能亮點 (v0.3.0)

### 1. 🚀 全面支援 GPU 運算與 RTX 5090 (Blackwell 架構)
* **自動硬體偵測**：程式啟動時自動掃描 NVIDIA GPU、顯存大小、CUDA 驅動版本與架構代號（Blackwell `sm_120`、Ada Lovelace `sm_89`、Ampere `sm_86` 等）。
* **RTX 5090 核心執行防護**：
  * RTX 50 系列採用最新 Blackwell 架構 (`sm_120`)。若當前安裝的 `onnxruntime-gpu` 未內建 sm_120 核心二進位檔或 CUDA 版本不符，系統會在熱機階段（Warm-up）精確攔截異常。
  * 提供明確之除錯指引，並**自動安全平滑回退至 CPU 或 DirectML 執行**，徹底杜絕程式因 CUDA 核心缺失而崩潰中斷。
* **Windows DirectML GPU 加速 (`--device dml`)**：
  * 支援免安裝 CUDA Toolkit 的 DirectX 12 GPU 加速方案，適用於所有現代顯卡。
* **Ollama 大模型本機 GPU 卸載**：
  * 搭配 RTX 5090（32GB 顯存）可將 `qwen3:8b` 模型 100% 裝載於顯存中，享受極速推論。

### 2. 💾 斷點續傳與即時原子存檔 (Checkpoint & Resume)
* **智能比對略過**：重新執行時自動比對 `output/borehole_final_results.json`，已成功處理的圖片將自動跳過，不再重複浪費算力。
* **零資料遺失保障（Zero-Loss Atomic Persistence）**：
  * 多執行緒每辨識完一張圖片，立即透過臨時檔案（`.tmp`）原子置換寫入磁碟。
  * 即使遇到斷電、電腦當機或使用者隨時按下 `Ctrl+C` 強制中止，先前完成的所有成果均 100% 完整保留於硬碟中。
* **優雅中斷處理**：捕捉 `KeyboardInterrupt` 訊號，自動取消後續排隊任務，並乾淨關閉執行緒池後安全存檔。

---

## 🛠️ 安裝教學（Installation Guide）

### 1. 基本環境需求
* **作業系統**：Windows 10/11、macOS 或 Linux
* **Python 版本**：Python 3.10 或更高版本
* **套件管理工具**：建議使用 [`uv`](https://docs.astral.sh/uv/) 或標準 `pip`
* **（選用）Ollama**：若需使用大語言模型輔助繁體語意校正，需於本機安裝 [Ollama](https://ollama.com/)

---

### 2. 環境安裝步驟

#### 方式 A：CPU 基礎環境（適用於無獨立顯卡或一般測試）
```bash
# 使用 pip
pip install opencv-python rapidocr-onnxruntime opencc-python-reimplemented requests numpy pillow
```

#### 方式 B：NVIDIA GPU 加速環境（強烈推薦，支援 RTX 5090 / 4090 / 3090 等）
如欲在 NVIDIA GPU 上獲得 5~10 倍的 OCR 推論加速：

1. **卸載可能衝突的 CPU 版本，並安裝 GPU 版 ONNX Runtime**：
   ```bash
   pip uninstall -y onnxruntime
   pip install onnxruntime-gpu
   ```
2. **安裝其餘必要依賴**：
   ```bash
   pip install opencv-python rapidocr-onnxruntime opencc-python-reimplemented requests numpy pillow
   ```
   > 💡 **RTX 5090 (Blackwell) 特別注意**：
   > * 請確保 NVIDIA 顯示卡驅動版本 $\ge 570.xx$（支援 CUDA 12.8+）。
   > * 若目前安裝的 `onnxruntime-gpu` 尚未支援 Blackwell sm_120 核心，系統會印出提示並自動平滑轉為 CPU 或 DirectML 執行；您亦可在 Windows 直接加上 `--device dml` 啟用 GPU 運算。

---

### 3. 設定 Ollama 本機模型（選用）

若欲啟用 `qwen3:8b` 進行地質名詞語意校正與嚴格繁體化：

1. 下載並安裝 [Ollama](https://ollama.com/)。
2. 下載 `qwen3:8b` 模型：
   ```bash
   ollama pull qwen3:8b
   ```
3. 確認 Ollama 正在執行（預設監聽於 `http://localhost:11434`），若本機有 RTX 5090，Ollama 將自動全數放入 GPU 顯存中。

---

## 🚀 執行方式

### 1. 一鍵自動執行（自動啟用 GPU、斷點續傳、讀取 `./input`）

```bash
python main.py
```
*(若有安裝 `uv`，亦可使用 `uv run main.py`)*

### 2. 指定運算硬體裝置

```bash
# 自動偵測 GPU（預設）
python main.py --device auto

# 強制使用 NVIDIA CUDA 加速
python main.py --device cuda

# 使用 Windows DirectML GPU 加速 (適用於全系列顯卡)
python main.py --device dml

# 強制使用 CPU 運算
python main.py --device cpu
```

### 3. 斷點續傳控制

```bash
# 預設情況：自動檢查 output/borehole_final_results.json，跳過已完成之檔案
python main.py

# 強制重新辨識所有圖片（不使用斷點續傳快取）
python main.py --overwrite

# 重新嘗試先前在 output/error.json 中被列為異常的檔案
python main.py --retry-errors
```

### 4. 高速離線模式（僅使用本地規則 + OpenCC，不需啟動 Ollama）

```bash
python main.py --no-llm
```

### 5. 完整參數說明

```bash
python main.py --help
```

| 參數 | 縮寫 | 預設值 | 說明 |
| :--- | :---: | :---: | :--- |
| `--device` | `-d` | `auto` | 運算裝置：`auto` (自動偵測), `cuda`, `dml`, `cpu` |
| `--input` | `-i` | `input` | 待處理圖檔之資料夾目錄路徑 |
| `--output` | `-o` | `output/borehole_final_results.json` | 成功辨識結構化 JSON 輸出路徑 |
| `--error` | `-e` | `output/error.json` | 異常待手動複核圖檔之記錄路徑 |
| `--model` | `-m` | `qwen3:8b` | Ollama 語言模型名稱 |
| `--no-llm` | - | `False` | 停用 LLM，採用純本地規則與 OpenCC 高速處理 |
| `--retries` | `-r` | `2` | 辨識異常時之最大自適應增強重試次數 |
| `--workers` | `-w` | 自動 (CPU 核心) | 平行處理執行緒數量 |
| `--overwrite` | - | `False` | 強制覆寫並重新處理所有檔案（關閉斷點續傳） |
| `--no-resume` | - | `False` | 等同 `--overwrite` |
| `--retry-errors`| - | `False` | 重新嘗試 `error.json` 中的檔案 |

---

## 🛡️ 防呆與強固設計

### 1. 斷點續傳與非預期中斷保護
* 執行過程中若使用者手動中斷（`Ctrl+C`），終端機會立即顯示：
  ```text
  ⚠️ 收到使用者中斷信號 (Ctrl+C)！正在緊急保存目前進度...
  💾 進度已安全寫入磁碟 (目前累積成功: 42 筆，待檢: 1 筆)。
  👉 下次啟動程式時，系統將自動接續中斷處繼續執行！
  ```
* 再次執行時：
  ```text
  [*] ⚡ 斷點續傳已生效：
     • 歷史已成功 : 42 筆 (自動略過)
     • 本次待執行 : 58 筆 (總共 100 筆)
  ```

### 2. 零額外腦補政策（Zero Hallucination）
* `description` 欄位除繁簡轉換與特定地質錯字修正（如 `灭` $\to$ `夾`、`碟` $\to$ `礫`、`蛋` $\to$ `黃`、`碍` $\to$ `磚`、`唇` $\to$ `層`）外，**嚴禁添加、刪除或臆測任何原文未提及之語句**。
* 完整保留工程分類代碼（如 `(SF)`, `(CL)`, `(GM)`）、地層分層深度與原意。

### 3. 多階段自適應增強重試（最多 3 次嘗試）
* **第 1 次嘗試（標準）**：全局灰階二值化去浮水印。
* **第 2 次嘗試（重試 1）**：1.3x 雙三次插值放大 + CLAHE 局部直方圖對比度增強 + Otsu 閾值 + 擴展直欄搜尋邊界。
* **第 3 次嘗試（重試 2）**：1.5x Lanczos4 放大 + 自適應高斯動態局部閾值。

---

## 📊 成果輸出範例 (`borehole_final_results.json`)

```json
{
  "test_data (1).png": {
    "borehole_info": {
      "borehole_id": "BH-1",
      "location": "新北市新店區",
      "coordinates": {
        "datum": "TWD97",
        "N": 2761442.0,
        "E": 305120.0
      },
      "elevation_m": 42.5,
      "date": "111/04/15"
    },
    "strata": [
      {
        "depth_m": 1.2,
        "description": "回填土夾碎磚塊與柏油碎石"
      },
      {
        "depth_m": 5.8,
        "description": "棕黃色粉土質細砂夾薄層黏土"
      },
      {
        "depth_m": 15.0,
        "description": "青灰色頁岩夾砂岩互層，岩心呈柱狀，極堅實"
      }
    ]
  }
}
```

---

## 📄 授權條款

本專案採用 MIT 授權條款開放原始碼。
