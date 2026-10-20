import cv2
import json
import re
import os
import sys
import numpy as np
from pathlib import Path
from rapidocr_onnxruntime import RapidOCR
from .device import create_rapidocr_engine

# Standard OCR typo corrections for Taiwanese geotechnical terms
CORRECTION_DICT = {
    "灭": "夾",
    "碟": "礫",
    "赚": "礫",
    "碍": "磚",
    "纵": "級",
    "堤": "塊",
    "坠画": "堅實",
    "坚画": "堅實",
    "坚实": "堅實",
    "坚硬": "堅硬",
    "真岩": "頁岩",
    "真若": "頁岩",
    "頁若": "頁岩",
    "黏土": "黏土",
    "粘土": "黏土",
    "雾峰": "霧峰",
    "蛋": "黃",
    "棕蛋": "棕黃",
    "趣繁密": "極緊密",
    "繁密": "緊密",
    "唇": "層",
    "孔内": "鑽孔內",
    "算":"質",
    "箕":"質",
    "爽":"夾",
}

# Geological / Geotechnical keywords for validation
GEOLOGICAL_KEYWORDS = [
    "土", "砂", "礫", "粘", "黏", "頁", "泥", "岩", "石", "塊", "卵", 
    "回填", "柏油", "混凝土", "級配", "粉土", "紅土", "表土", "岩塊",
    "堅實", "稠度", "緊密", "風化", "岩心", "粒徑", "級配"
]

USCS_PATTERN = r'\((?:CL|SF|GM|GW|GP|SW|SP|GC|SM|SC|ML|MH|CH|OH|OL|PT)\)'

# Table headers to filter out if accidentally captured
TABLE_HEADER_NOISE = {"深度", "取樣記錄", "標準貫入", "地質圖元", "顏色", "鑽進速率", "孔水位", "RQD", "RQD(%)"}

def clean_geological_text(text: str) -> str:
    """Clean typos without hallucinating or altering meaning."""
    for k, v in CORRECTION_DICT.items():
        text = text.replace(k, v)
    text = text.replace(":", "；").replace("：", "；").replace(";", "；")
    text = re.sub(r'^[,\s;，；]+', '', text)
    text = re.sub(r'[,\s;，；]+$', '', text)
    return text.strip()

class BoreholeLogParser:
    def __init__(self, device: str = "auto", ocr_engine=None):
        if ocr_engine is not None:
            self.ocr = ocr_engine
        else:
            self.ocr = create_rapidocr_engine(device=device)

    def preprocess_image(self, img_path: str, attempt: int = 1):
        """
        Load and preprocess image with different strategies depending on attempt number:
        - Attempt 1: Standard thresholding (watermark removal)
        - Attempt 2: 1.3x Upscale + CLAHE contrast enhancement + Otsu
        - Attempt 3: 1.5x Upscale + Adaptive thresholding
        """
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if attempt == 1:
            # Standard threshold
            _, binary = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY)
            return img, cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        elif attempt == 2:
            # 1.3x Upscale + CLAHE contrast enhancement
            h, w = gray.shape
            resized = cv2.resize(gray, (int(w * 1.3), int(h * 1.3)), interpolation=cv2.INTER_CUBIC)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(resized)
            _, binary = cv2.threshold(enhanced, 180, 255, cv2.THRESH_BINARY)
            return img, cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        else: # attempt == 3
            # 1.5x Upscale + Adaptive thresholding
            h, w = gray.shape
            resized = cv2.resize(gray, (int(w * 1.5), int(h * 1.5)), interpolation=cv2.INTER_LANCZOS4)
            binary = cv2.adaptiveThreshold(
                resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
            )
            return img, cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def parse_borehole(self, img_path: str, attempt: int = 1) -> dict:
        orig_img, cleaned_img = self.preprocess_image(img_path, attempt=attempt)
        h, w = cleaned_img.shape[:2]

        ocr_results, _ = self.ocr(cleaned_img)
        if not ocr_results:
            return {"borehole_info": {}, "strata": []}

        # Structure OCR boxes
        boxes = []
        for line in ocr_results:
            box, text, score = line
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            cx, cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
            boxes.append({
                "box": box,
                "text": text.strip(),
                "score": score,
                "cx": cx, "cy": cy,
                "min_x": min_x, "max_x": max_x,
                "min_y": min_y, "max_y": max_y
            })

        # Locate table header row (Y: 15% ~ 25% of image height)
        header_row_boxes = [b for b in boxes if h * 0.15 <= b["cy"] <= h * 0.26]
        
        # Find "岩石或土壤性質描述" column header
        desc_col_box = None
        for b in header_row_boxes:
            t = b["text"]
            if any(kw in t for kw in ["性質描述", "岩石或土壤", "土壤性質", "性質", "描述"]):
                desc_col_box = b
                break

        # Dynamic column X-range with expansion on retries
        margin = 15 if attempt == 1 else 25
        if desc_col_box:
            col_min_x = desc_col_box["min_x"] - margin
            table_header_y = desc_col_box["max_y"]
            
            right_cols = [b for b in header_row_boxes if b["min_x"] > desc_col_box["max_x"] + 5]
            if right_cols:
                closest_right = min(right_cols, key=lambda b: b["min_x"])
                col_max_x = closest_right["min_x"] - 2
            else:
                col_max_x = desc_col_box["max_x"] + 70
        else:
            col_min_x = w * 0.35
            col_max_x = w * 0.65
            table_header_y = h * 0.22

        # Step 1: Parse Header Information (Y <= table_header_y)
        header_boxes = [b for b in boxes if b["max_y"] <= table_header_y + 10]
        borehole_info = self._parse_header(header_boxes, boxes)

        # Step 2: Parse Strata Information (inside Description Column ROI)
        col_boxes = [
            b for b in boxes 
            if b["min_y"] >= table_header_y - 5
            and (col_min_x <= b["cx"] <= col_max_x or (b["min_x"] >= col_min_x - 10 and b["max_x"] <= col_max_x + 15))
            and b["min_y"] < h * 0.93
        ]
        
        strata = self._parse_strata(col_boxes)

        return {
            "borehole_info": borehole_info,
            "strata": strata
        }

    def _parse_header(self, header_boxes: list, all_boxes: list) -> dict:
        full_header_text = " ".join([b["text"] for b in header_boxes])

        # 1. Borehole ID
        borehole_id = "未知"
        m_id = re.search(r'(?:鑽|孔|鑽孔)?\s*編號\s*[:：]\s*([A-Za-z0-9\-_#]+)', full_header_text)
        if not m_id:
            m_id = re.search(r'(?:鑽|孔|鑽孔)?\s*编號\s*[:：]\s*([A-Za-z0-9\-_#]+)', full_header_text)
        if m_id:
            borehole_id = m_id.group(1).strip()
        else:
            for b in header_boxes:
                m = re.search(r'(?:編號|编號)\s*[:：]\s*([A-Za-z0-9\-_#]+)', b["text"])
                if m:
                    borehole_id = m.group(1).strip()
                    break

        # 2. Location
        location = "未知 / 無記載"
        for b in header_boxes:
            if "地點" in b["text"]:
                loc_val = re.sub(r'.*地點\s*[:：]\s*', '', b["text"]).strip()
                if loc_val and loc_val not in ["-", "：", "無"]:
                    location = loc_val
                break

        # 3. Elevation
        elevation_m = None
        m_ele = re.search(r'(?:鑽孔|孔)?\s*(?:標高|棵高)\s*[:：]?\s*(\d+\.?\d*)\s*M?', full_header_text)
        if m_ele:
            try: elevation_m = float(m_ele.group(1))
            except: pass

        # 4. Datum & Coordinates
        datum = "TWD97"
        if "TWD67" in full_header_text:
            datum = "TWD67"
        elif "TWD97" in full_header_text:
            datum = "TWD97"

        coord_n = None
        coord_e = None
        m_n = re.search(r'(?:坐標|坐棵|座標|坐\s*標)?\s*N\s*[:：=]\s*(\d+\.?\d*)', full_header_text)
        if m_n:
            try: coord_n = float(m_n.group(1))
            except: pass

        m_e = re.search(r'(?:坐標|坐棵|座標|坐\s*標)?\s*E\s*[:：=]\s*(\d+\.?\d*)', full_header_text)
        if m_e:
            try: coord_e = float(m_e.group(1))
            except: pass

        # 5. Date
        date_str = "未知 / 無記載"
        m_date = re.search(r'日\s*期\s*[:：]\s*([0-9\~/~-]+)', full_header_text)
        if m_date and m_date.group(1).strip() not in ["~", "-", ""]:
            date_str = m_date.group(1).strip()
        else:
            for b in all_boxes:
                m_sub = re.search(r'(\d{2,3}[/\.]\d{2}[/\.]\d{2})', b["text"])
                if m_sub:
                    date_str = m_sub.group(1).replace('.', '/')
                    break

        return {
            "borehole_id": borehole_id,
            "location": location,
            "coordinates": {
                "datum": datum,
                "N": coord_n,
                "E": coord_e
            },
            "elevation_m": elevation_m,
            "date": date_str
        }

    def _parse_strata(self, col_boxes: list) -> list:
        depth_items = []
        desc_items = []

        for b in col_boxes:
            t = b["text"]
            # Check for depth pattern
            m = re.search(r'(\d+\.\d+)\s*[M|m]', t)
            if not m:
                m = re.search(r'^(\d+\.\d+)$', t)
            
            if m:
                depth_val = float(m.group(1))
                depth_items.append({
                    "depth_m": depth_val,
                    "cy": b["cy"],
                    "min_y": b["min_y"],
                    "max_y": b["max_y"],
                    "text": t
                })
                prefix_text = re.sub(r'\d+\.\d+\s*[M|m]?', '', t).strip()
                if len(prefix_text) >= 2:
                    desc_items.append({
                        "text": prefix_text,
                        "cy": b["cy"],
                        "min_y": b["min_y"],
                        "max_y": b["max_y"]
                    })
            else:
                desc_items.append({
                    "text": t,
                    "cy": b["cy"],
                    "min_y": b["min_y"],
                    "max_y": b["max_y"]
                })

        depth_items = sorted(depth_items, key=lambda d: d["cy"])
        
        # Deduplicate
        unique_depths = []
        for d in depth_items:
            if not unique_depths:
                unique_depths.append(d)
            else:
                prev = unique_depths[-1]
                if abs(d["depth_m"] - prev["depth_m"]) < 0.01:
                    continue
                if abs(d["depth_m"] - prev["depth_m"]) < 0.2 and abs(d["cy"] - prev["cy"]) < 30:
                    unique_depths[-1] = d
                    continue
                unique_depths.append(d)

        strata = []
        prev_y = 0

        for d in unique_depths:
            curr_y = d["max_y"] + 6
            layer_texts = []
            for item in desc_items:
                if prev_y <= item["cy"] <= curr_y:
                    cleaned = clean_geological_text(item["text"])
                    if cleaned and cleaned not in layer_texts and cleaned not in TABLE_HEADER_NOISE:
                        layer_texts.append(cleaned)
            
            # Intelligently merge multi-line text fragments without forcing artificial commas
            merged = ""
            for seg in layer_texts:
                seg = seg.strip()
                if not seg:
                    continue
                if not merged:
                    merged = seg
                    continue
                
                last_c = merged[-1]
                first_c = seg[0]

                # If previous segment ends with punctuation or current starts with punctuation/bracket, merge directly
                if last_c in "；;,，、：:()（）[]【】" or first_c in "；;,，、：:()（）[]【】":
                    merged += seg
                else:
                    # Natural continuous text flow across multi-line OCR boxes
                    merged += seg
            
            # Normalize repeated punctuation
            merged = re.sub(r'[，,]+', '，', merged)
            merged = re.sub(r'[；;]+', '；', merged)
            merged = re.sub(r'；[，,]', '；', merged)
            merged = re.sub(r'[，,]；', '；', merged)
            merged = re.sub(r'^[，,；;\s]+', '', merged)
            merged = re.sub(r'[，,；;\s]+$', '', merged)

            strata.append({
                "depth_m": d["depth_m"],
                "description": merged
            })
            prev_y = curr_y

        return strata

    def validate_extraction(self, data: dict) -> tuple[bool, str]:
        """
        Validate whether the extracted data conforms to geological expectations:
        1. Must contain borehole_info.
        2. Strata must not be empty.
        3. Every stratum's description must contain soil/rock geotechnical keywords or USCS symbols.
        4. Description must not be empty or solely table header noise.
        """
        if not data or not isinstance(data, dict):
            return False, "解析資料為空或格式錯誤"

        strata = data.get("strata", [])
        if not strata or len(strata) == 0:
            return False, "未偵測到任何地層深度與描述 (strata 為空)"

        for idx, layer in enumerate(strata, 1):
            desc = layer.get("description", "").strip()
            depth = layer.get("depth_m")

            if depth is None or depth <= 0:
                return False, f"第 {idx} 層深度數值異常 ({depth})"

            if not desc:
                return False, f"第 {idx} 層 (深度 {depth}m) 之描述為空"

            # Check if desc contains any geological keywords or USCS classification code
            has_geo_kw = any(kw in desc for kw in GEOLOGICAL_KEYWORDS)
            has_uscs = bool(re.search(USCS_PATTERN, desc, re.IGNORECASE))

            if not (has_geo_kw or has_uscs):
                return False, f"第 {idx} 層描述內容非土壤或岩石岩性: '{desc}'"

        return True, "驗證通過"
