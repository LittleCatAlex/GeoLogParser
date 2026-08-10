import json
import re
import sys
import requests
from opencc import OpenCC
from .geo import TaiwanPlaceMatcher

class OllamaBoreholeCleaner:
    def __init__(self, model_name="qwen3:8b", ollama_host="http://localhost:11434"):
        self.model_name = model_name
        self.ollama_url = f"{ollama_host}/api/generate"
        self.cc = OpenCC('s2twp')
        self.geo_matcher = TaiwanPlaceMatcher()

    def local_preclean(self, data: dict) -> dict:
        """Rule-based and OpenCC precleaning for fast deterministic conversion."""
        raw_str = json.dumps(data, ensure_ascii=False)
        trad_str = self.cc.convert(raw_str)
        cleaned_data = json.loads(trad_str)

        # Standardize location
        loc = cleaned_data.get("borehole_info", {}).get("location", "")
        if loc:
            cleaned_data["borehole_info"]["location"] = self.geo_matcher.match_and_standardize_location(loc)

        return cleaned_data

    def clean_with_llm(self, raw_data: dict) -> dict:
        """
        Call Ollama qwen3:8b strictly for:
        1. Converting simplified characters to Traditional Chinese.
        2. Fixing broken line breaks and abnormal semicolons/punctuation (e.g. '粗砂礫；石；卵石' -> '粗砂礫石，卵石').
        3. Fixing obvious OCR typos without hallucinating new descriptions.
        4. Ensuring valid JSON.
        """
        precleaned = self.local_preclean(raw_data)
        
        prompt = f"""你是一個精準的地質工程資料校對助手。
請校對以下由 OCR 萃取之地質鑽孔柱狀圖 JSON 資料。

【校對核心任務與規範】：
1. 【清理異常斷句與分號】：
   - 修復因 OCR 換行切分產生的多餘分號與斷詞（例如：「粗砂礫；石；卵石」應修正為「粗砂礫石，卵石」或「粗砂礫石、卵石」；「粉土質；黏土」應修正為「粉土質黏土」；「粒；徑」應修正為「粒徑」）。
   - 使語意標點通順自然，但【絕不可憑空腦補或增添原文不存在的岩性或內容】。
2. 【保留原始工程代碼與括號】：完整保留 (SF), (CL), (GM) 等土壤分類代碼、數值與單位。
3. 【嚴格純繁體中文】：結果中絕不可出現任何簡體中文字、日文或異體字雜訊。
4. 【數值嚴格保真】：不可更動 coordinates (N, E), elevation_m, depth_m, borehole_id, date 之原始數值。
5. 【輸出格式】：【僅輸出標準 JSON 字串】，絕不包含任何解釋、Markdown 外框或前言後記。

【待校對資料】：
{json.dumps(precleaned, ensure_ascii=False, indent=2)}

請輸出校對完成後的 JSON："""

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.05
            }
        }

        try:
            resp = requests.post(self.ollama_url, json=payload, timeout=60)
            if resp.status_code == 200:
                result_text = resp.json().get("response", "").strip()
                result_text = re.sub(r'^```(?:json)?\s*', '', result_text)
                result_text = re.sub(r'\s*```$', '', result_text)
                
                parsed_json = json.loads(result_text)
                final_str = self.cc.convert(json.dumps(parsed_json, ensure_ascii=False))
                return json.loads(final_str)
            else:
                print(f"[WARN] Ollama API returned status {resp.status_code}, using local precleaned data.")
                return precleaned
        except Exception as e:
            print(f"[WARN] Ollama LLM clean failed ({e}), using local precleaned data.")
            return precleaned
