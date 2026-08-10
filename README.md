# GeoLogParser - 地質鑽孔柱狀圖自動化結構化萃取與繁體校對系統

本系統專為台灣工程地質鑽孔柱狀圖設計，實現**全地端／完全離線（On-Premise Offline）**的影像辨識（OCR）、動態欄位解析、地質岩性層位綁定，並結合本機 **Ollama (`qwen3:8b`)** 與台灣真實地名資料庫進行純繁體語意校正與資料完整性驗證。

---

## 專案目錄結構

```text
.
├── input/                             # 📥 輸入目錄：存放待辨識的鑽孔柱狀圖 (*.png, *.jpg)
├── output/                            # 📤 輸出目錄：存放處理成果與異常報告
│   ├── borehole_final_results.json    # 檢驗通過之結構化 JSON 成果
│   └── error.json                     # (若有) 異常待手動複核之詳細記錄檔
├── data/                              # 📚 參考資料目錄
│   └── taiwan_places.json             # 台灣 22 縣市 368 鄉鎮區真實地名資料庫
├── src/                               # ⚙️ 核心原始碼模組
│   ├── __init__.py
│   ├── parser.py                      # 離線 OCR、多階段自適應前處理與地質驗證引擎
│   ├── cleaner.py                     # Ollama qwen3:8b 零腦補繁體校對器
│   ├── geo.py                         # 台灣真實行政區與地名對照模組
│   └── pipeline.py                    # 批次排程、驗證、自動重試與異常分流管線
├── main.py                            # 🚀 一鍵執行入口（內建套件防呆檢查）
├── pyproject.toml                     # 📦 專案環境依賴配置
└── README.md                          # 📖 專案說明與操作手冊
```

---

## 🛠️ 安裝教學（Installation Guide）

### 1. 環境需求
* **作業系統**：Windows 10/11、macOS 或 Linux
* **Python 版本**：Python 3.10 或更高版本
* **套件管理工具**：建議使用 [`uv`](https://docs.astral.sh/uv/)（極速、免手動切換環境）或標準 `pip`
* **（選用）Ollama**：若需使用大語言模型輔助繁體語意校正，需於本機安裝 [Ollama](https://ollama.com/)

---

### 2. 環境安裝步驟

#### 方式 A：使用 `uv` 建立虛擬環境（推薦，最簡單快速）

1. **安裝 `uv`（若尚未安裝）**：
   * Windows (PowerShell)：
     ```powershell
     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
     ```
   * Linux / macOS：
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```

2. **建立虛擬環境並安裝依賴套件**：
   在專案根目錄下執行：
   ```bash
   uv venv
   uv pip install -r pyproject.toml
   ```
   > 💡 提示：使用 `uv` 時，亦可直接執行 `uv run main.py`，系統將自動配置環境。

---

#### 方式 B：使用標準 Python `pip` 安裝

1. **建立並啟用虛擬環境**：
   * Windows：
     ```powershell
     python -m venv .venv
     .venv\Scripts\activate
     ```
   * Linux / macOS：
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

2. **安裝所需套件**：
   ```bash
   pip install opencv-python rapidocr-onnxruntime opencc-python-reimplemented requests numpy pillow
   ```

---

### 3. 設定 Ollama 本機模型（選用）

若欲啟用 `qwen3:8b` 進行地質名詞語意校正與嚴格繁體化：

1. 下載並安裝 [Ollama](https://ollama.com/)。
2. 於終端機下載 `qwen3:8b` 模型：
   ```bash
   ollama pull qwen3:8b
   ```
3. 確認 Ollama 服務正在執行（預設監聽於 `http://localhost:11434`）。

---

## 🚀 執行方式

### 1. 預設執行完整管線（自動讀取 `./input`，重試 2 次，輸出至 `./output`）

```bash
uv run main.py
```
*(若使用傳統 pip 虛擬環境，可直接輸入 `python main.py`)*

### 2. 高速離線模式（僅使用本地規則 + OpenCC 繁體轉換，不需啟動 Ollama）

```bash
uv run main.py --no-llm
```

### 3. 自訂目錄與參數

```bash
uv run main.py --input "./input" --output "./output/borehole_final_results.json" --error "./output/error.json" --retries 2 --model "qwen3:8b"
```

#### 參數說明：
* `--input` / `-i`：輸入圖片目錄路徑（預設：`input`）
* `--output` / `-o`：辨識成功成果之 JSON 輸出路徑（預設：`output/borehole_final_results.json`）
* `--error` / `-e`：異常待複核圖檔之記錄路徑（預設：`output/error.json`）
* `--model` / `-m`：Ollama 模型名稱（預設：`qwen3:8b`）
* `--retries` / `-r`：辨識異常時之最大自適應重試次數（預設：`2`）
* `--no-llm`：關閉 LLM，改採純本機規則與 OpenCC 高速處理

---

## 🛡️ 防呆與異常處理機制

### 1. 套件缺失自動防呆
若於未安裝套件的環境直接執行 `main.py`，系統將自動攔截並顯示友善指引：
```text
======================================================================
❌ 【執行中斷】偵測到缺少必要的 Python 套件！
======================================================================
缺少套件清單：
  • opencv-python>=4.8.0
  • rapidocr-onnxruntime>=1.4.0

💡 請依照以下方式之一進行安裝：
----------------------------------------------------------------------
【方法 1】使用 uv 自動執行（推薦，最快速且自動處理虛擬環境）：
   uv run main.py

【方法 2】使用 uv 安裝至目前虛擬環境：
   uv pip install opencv-python>=4.8.0 rapidocr-onnxruntime>=1.4.0

【方法 3】使用標準 pip 安裝：
   pip install opencv-python>=4.8.0 rapidocr-onnxruntime>=1.4.0
======================================================================
```

### 2. 零額外腦補政策（Zero Hallucination）
* `description` 欄位**除繁簡轉換與明顯 OCR 錯字修正（如 `灭` $\to$ `夾`、`碟` $\to$ `礫`、`蛋` $\to$ `黃`、`碍` $\to$ `磚`、`纵` $\to$ `級`、`唇` $\to$ `層`）外，嚴禁添加、刪除或改寫任何原文未提及之語句**。
* 完整保留工程分類代碼（如 `(SF)`, `(CL)`, `(GM)`）、原始標點與結構。

### 3. 多階段自適應增強重試（最多重試 2 次）
* **第 1 次嘗試（標準）**：全局灰階二值化去浮水印。
* **第 2 次嘗試（重試 1）**：1.3x 雙三次插值放大 + CLAHE 局部直方圖對比度增強 + Otsu 閾值 + 擴展直欄搜尋邊界。
* **第 3 次嘗試（重試 2）**：1.5x Lanczos4 放大 + 自適應高斯動態局部閾值。

### 4. 異常檔案自動分流（`output/error.json`）
若圖檔重試 2 次（共 3 次嘗試）後依然未能通過地質岩性檢驗，系統會將該圖檔資訊記錄於 `output/error.json`，供人工複核：

```json
{
  "sample_corrupted.png": {
    "file_name": "sample_corrupted.png",
    "file_path": "input/sample_corrupted.png",
    "attempts_made": 3,
    "error_reason": "第 2 層描述內容非土壤或岩石岩性: '### 雜訊 ###'",
    "raw_extracted_data": {
      "borehole_info": { "borehole_id": "BH-99", "location": "未知" },
      "strata": [
        { "depth_m": 1.0, "description": "回填土" },
        { "depth_m": 3.5, "description": "### 雜訊 ###" }
      ]
    },
    "suggested_action": "請使用者手動驗證與補正此圖檔之層位與描述"
  }
}
```

---

## 📊 輸出成果範例（`output/borehole_final_results.json`）

```json
{
  "768667379_1019032617600761_2227929436781220652_n.png": {
    "borehole_info": {
      "borehole_id": "BH-1",
      "location": "未知 / 無記載",
      "coordinates": {
        "datum": "TWD97",
        "N": 2616442.0,
        "E": 298000.0
      },
      "elevation_m": 240.0,
      "date": "未知 / 無記載"
    },
    "strata": [
      {
        "depth_m": 0.5,
        "description": "柏油，混凝土夾粗砂礫；石；卵石"
      },
      {
        "depth_m": 1.6,
        "description": "礫石夾灰色粗砂礫石"
      },
      {
        "depth_m": 4.19,
        "description": "頁岩巖塊夾灰色粉土質；黏土"
      },
      {
        "depth_m": 15.1,
        "description": "灰色頁岩夾泥岩"
      }
    ]
  }
}
```
