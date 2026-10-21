import cv2
import json
import re
import os
import sys
import numpy as np
from pathlib import Path
from rapidocr_onnxruntime import RapidOCR
from .device import create_rapidocr_engine

# Standard OCR typo corrections for Taiwanese and English geotechnical terms
CORRECTION_DICT = {
    "灭": "夾", "买": "夾", "買": "夾", "爽": "夾",
    "碟": "礫", "赚": "礫", "碍": "磚", "纵": "級", "堤": "塊",
    "坠画": "堅實", "坚画": "堅實", "坚实": "堅實", "坚硬": "堅硬",
    "真岩": "頁岩", "真若": "頁岩", "頁若": "頁岩", "臭巖": "頁岩", "臭岩": "頁岩",
    "硬良岩": "硬頁岩", "硬臭岩": "硬頁岩", "硬直岩": "硬頁岩", "硬其岩": "硬頁岩",
    "板老": "板岩", "被岩": "板岩",
    "黏土": "黏土", "粘土": "黏土", "粉質粘士": "粉土質黏土", "粉質粘土": "粉土質黏土",
    "雾峰": "霧峰", "蛋": "黃", "棕蛋": "棕黃",
    "趣繁密": "極緊密", "繁密": "緊密", "唇": "層", "孔内": "鑽孔內",
    "算": "質", "箕": "質",
    "脆": "脈", "石英脆": "石英脈", "石英派": "石英脈",
    "股结": "膠結", "隧结": "膠結", "膨结": "膠結", "图结": "膠結", "题结": "膠結",
    "胶结": "膠結", "膨結": "膠結", "隧結": "膠結", "胶洁": "膠結",
    "燮質": "變質", "漫質": "變質", "發質": "變質", "鍵質": "變質",
    "護質": "變質", "澄質": "變質", "婆質": "變質", "發賀": "變質", "裝資": "變質",
    "前眉": "崩積層", "崩磨": "崩積層", "崩精層": "崩積層", "崩猜": "崩積",
    "破²": "破碎", "破辟": "破碎", "破驿": "破碎",
    "岩局": "岩屑", "岩属": "岩屑", "碎局": "碎屑", "碎属": "碎屑",
    "即礫石": "卵礫石", "磚瑰": "磚塊", "水泥瑰": "水泥塊", "岩瑰": "岩塊", "安山岩瑰": "安山岩塊", "硬瑰": "硬塊",
    "剪裂带": "剪裂帶", "通状": "塊狀", "次阎状": "次圓狀",
    "淘湿度": "淘選度", "淘激度": "淘選度",
    "Backfiled": "Backfilled",
}

# Geological / Geotechnical keywords for validation (Chinese)
GEOLOGICAL_KEYWORDS_ZH = [
    # Soils & aggregates
    "土", "砂", "礫", "粘", "黏", "泥", "粉土", "沈泥", "沉泥", "回填", "柏油", "混凝土",
    "級配", "紅土", "表土", "卵石", "塊石", "碎石", "轉石", "崩積", "沖積", "殘積", "堆積",
    # Rocks
    "頁岩", "砂岩", "泥岩", "板岩", "硬頁岩", "千枚岩", "片岩", "片麻岩", "大理岩", "石灰岩",
    "玄武岩", "安山岩", "凝灰岩", "集塊岩", "角礫岩", "花崗岩", "粗粒玄武岩", "石英岩",
    "岩", "石", "塊", "頁", "岩塊", "岩心", "岩體", "岩盤", "基岩", "生岩", "互層",
    # Special geological features & states
    "煤", "煤碳", "煤炭", "煤層", "炭", "炭質", "碳質", "瀝青", "石膏", "石英", "高嶺土",
    "剪裂", "剪動", "破碎", "碎屑", "斷層", "斷層泥", "節理", "劈理", "層理", "風化",
    "堅實", "堅硬", "緊密", "極緊密", "鬆散", "軟弱", "稠度", "塑性", "膠結", "透水",
    "崩磨", "板砂", "泥砂", "砂泥", "變質砂岩", "凝灰", "角礫", "安山", "玄武", "石英脈"
]

# Geological / Geotechnical keywords for validation (English)
GEOLOGICAL_KEYWORDS_EN = [
    # Soils
    "clay", "sand", "silt", "gravel", "cobble", "boulder", "peat", "loam", "till",
    "fill", "backfilled", "backfill", "topsoil", "debris", "alluvium", "colluvium", "sediment",
    # Rocks
    "rock", "stone", "shale", "sandstone", "mudstone", "siltstone", "slate",
    "phyllite", "schist", "gneiss", "marble", "quartzite", "limestone", "coal",
    "basalt", "andesite", "tuff", "breccia", "agglomerate", "granite", "diorite",
    "argillite",
    # Modifiers & states
    "silty", "clayey", "sandy", "gravelly", "weathered", "fractured", "jointed",
    "dense", "loose", "stiff", "soft", "hard", "firm", "coarse", "medium", "fine",
    "gray", "grey", "brown", "yellow", "black", "dark", "light", "red", "green", "blue",
    "brownish", "grayish", "yellowish", "whitish",
    # Common OCR typo variants
    "sit", "sity", "yelowish", "bxown", "saond", "vellowish", "gry"
]

USCS_PATTERN = r'(?:\b|\()(?:CL|CH|ML|MH|SC|SM|SP|SW|GC|GM|GP|GW|PT|OH|OL|SF|CL-ML|GW-GM|GP-GM|GW-GC|GP-GC|SW-SM|SP-SM|SW-SC|SP-SC)(?:\b|\))'

# Watermark & Database Footers to strip out
WATERMARK_REGEX = re.compile(
    r'(?:(?:工程|地質|株|賀|深|也|乙|價|皙|暫|暂|世|曾|七|所)?\s*探勘?\s*(?:資料庫|資訊|詢|系|系統|瓷料庫|資科庫|料库|資料库|查詢系|香詢系|香询系|询系统|查询系统|查詢系統|查询系疏|休詢)*)|'
    r'(?:(?:地質|工程)?\s*資料庫\s*(?:系統)?)|'
    r'(?:地質調查所|經濟部中央地質|探勘查詢系|查询系統|查詢系統|查询系统|不四|人、、)',
    re.IGNORECASE
)

# Table headers to filter out if accidentally captured
TABLE_HEADER_NOISE = {
    "深度", "取樣記錄", "標準貫入", "地質圖元", "顏色", "鑽進速率", "孔水位", "RQD", "RQD(%)",
    "depth", "sample", "symbol", "description", "lithology", "recovery", "tcr"
}

def clean_geological_text(text: str) -> str:
    """Clean typos and watermark noise without hallucinating or altering meaning."""
    for k, v in CORRECTION_DICT.items():
        text = text.replace(k, v)
    text = WATERMARK_REGEX.sub("", text)
    text = text.replace(":", "；").replace("：", "；").replace(";", "；")
    text = re.sub(r'^[,\s;，；~、]+', '', text)
    text = re.sub(r'[,\s;，；~、]+$', '', text)
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
            _, binary = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY)
            return img, cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        elif attempt == 2:
            h, w = gray.shape
            resized = cv2.resize(gray, (int(w * 1.3), int(h * 1.3)), interpolation=cv2.INTER_CUBIC)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(resized)
            _, binary = cv2.threshold(enhanced, 180, 255, cv2.THRESH_BINARY)
            return img, cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        else: # attempt == 3
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

        # Locate table header row (Y: 8% ~ 32% of image height)
        header_row_boxes = [b for b in boxes if h * 0.08 <= b["cy"] <= h * 0.32]
        
        # Find "岩石或土壤性質描述" or English "Description" column header
        desc_col_box = None
        desc_keywords = [
            "性質描述", "岩石或土壤", "土壤性質", "性質", "描述",
            "岩性", "地層描述", "地質柱狀", "description", "lithology", "material"
        ]
        for b in header_row_boxes:
            t_lower = b["text"].lower()
            if any(kw in t_lower for kw in desc_keywords):
                desc_col_box = b
                break

        # Dynamic column X-range with expansion on retries
        margin = 20 if attempt == 1 else 35
        if desc_col_box:
            col_min_x = desc_col_box["min_x"] - margin
            table_header_y = desc_col_box["max_y"]
            
            right_cols = [b for b in header_row_boxes if b["min_x"] > desc_col_box["max_x"] + 5]
            if right_cols:
                closest_right = min(right_cols, key=lambda b: b["min_x"])
                col_max_x = closest_right["min_x"] - 2
            else:
                col_max_x = desc_col_box["max_x"] + 85
        else:
            col_min_x = w * 0.30
            col_max_x = w * 0.70
            table_header_y = h * 0.20

        # Step 1: Parse Header Information (Y <= table_header_y)
        header_boxes = [b for b in boxes if b["max_y"] <= table_header_y + 10]
        borehole_info = self._parse_header(header_boxes, boxes)

        # Step 2: Parse Strata Information (inside Description Column ROI)
        col_boxes = [
            b for b in boxes 
            if b["min_y"] >= table_header_y - 5
            and (col_min_x <= b["cx"] <= col_max_x or (b["min_x"] >= col_min_x - 15 and b["max_x"] <= col_max_x + 20))
            and b["min_y"] < h * 0.94
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
            # Exclude dip angles (e.g. 50°, 40~50°), RQD/percentages
            if re.search(r'°|度|%|％', t):
                desc_items.append({
                    "text": t,
                    "cy": b["cy"],
                    "min_y": b["min_y"],
                    "max_y": b["max_y"]
                })
                continue

            # Support integer and decimal depth values
            m = re.search(r'(\d+(?:\.\d+)?)\s*[M|m]', t)
            if not m:
                m = re.search(r'^(\d+(?:\.\d+)?)$', t)
            
            if m:
                depth_val = float(m.group(1))
                if 0.1 <= depth_val <= 600.0:
                    depth_items.append({
                        "depth_m": depth_val,
                        "cy": b["cy"],
                        "min_y": b["min_y"],
                        "max_y": b["max_y"],
                        "text": t
                    })
                    prefix_text = re.sub(r'\d+(?:\.\d+)?\s*[M|m]?', '', t).strip()
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
            else:
                desc_items.append({
                    "text": t,
                    "cy": b["cy"],
                    "min_y": b["min_y"],
                    "max_y": b["max_y"]
                })

        depth_items = sorted(depth_items, key=lambda d: d["cy"])
        
        # Deduplicate and enforce monotonic depth progression
        unique_depths = []
        for d in depth_items:
            if not unique_depths:
                unique_depths.append(d)
            else:
                prev = unique_depths[-1]
                if abs(d["depth_m"] - prev["depth_m"]) < 0.05:
                    continue
                # Reject non-monotonic depths (internal numbers / coordinates)
                if d["depth_m"] < prev["depth_m"]:
                    continue
                if abs(d["depth_m"] - prev["depth_m"]) < 0.2 and abs(d["cy"] - prev["cy"]) < 30:
                    unique_depths[-1] = d
                    continue
                unique_depths.append(d)

        strata = []
        prev_y = 0

        for d in unique_depths:
            curr_y = d["max_y"] + 8
            layer_texts = []
            for item in desc_items:
                if prev_y <= item["cy"] <= curr_y:
                    cleaned = clean_geological_text(item["text"])
                    if cleaned and cleaned not in layer_texts and cleaned not in TABLE_HEADER_NOISE:
                        layer_texts.append(cleaned)
            
            # Intelligently merge multi-line text fragments
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
                if last_c in "；;,，、：:()（）[]【】" or first_c in "；;,，、：:()（）[]【】":
                    merged += seg
                else:
                    merged += seg

            # Normalize repeated punctuation
            merged = re.sub(r'[，,]+', '，', merged)
            merged = re.sub(r'[；;]+', '；', merged)
            merged = re.sub(r'；[，,]', '；', merged)
            merged = re.sub(r'[，,]；', '；', merged)
            merged = re.sub(r'^[，,；;\s]+', '', merged)
            merged = re.sub(r'[，,；;\s]+$', '', merged)

            # If merged description is empty or solely watermark/noise:
            # Do NOT create an empty stratum!
            # If we already have a previous stratum, update its depth to current depth
            if not merged:
                if strata:
                    strata[-1]["depth_m"] = d["depth_m"]
                continue

            strata.append({
                "depth_m": d["depth_m"],
                "description": merged
            })
            prev_y = curr_y

        # Handle trailing text after the last depth if any
        if unique_depths and strata:
            last_d = unique_depths[-1]
            trailing_texts = [
                clean_geological_text(item["text"]) for item in desc_items
                if item["cy"] > last_d["max_y"] + 8
            ]
            trailing_valid = [t for t in trailing_texts if t and t not in TABLE_HEADER_NOISE]
            if trailing_valid:
                strata[-1]["description"] += "；" + "；".join(trailing_valid)

        return strata

    def validate_extraction(self, data: dict) -> tuple[bool, str]:
        """
        Validate whether the extracted data conforms to geological expectations:
        1. Must contain borehole_info.
        2. Strata must not be empty.
        3. At least 1 valid geological stratum exists.
        4. Valid geological strata must represent at least 50% of the total layers.
        """
        if not data or not isinstance(data, dict):
            return False, "解析資料為空或格式錯誤"

        strata = data.get("strata", [])
        if not strata or len(strata) == 0:
            return False, "未偵測到任何地層深度與描述 (strata 為空)"

        valid_count = 0
        prev_has_geo = False
        non_empty_strata = []

        for idx, layer in enumerate(strata, 1):
            desc = layer.get("description", "").strip()
            depth = layer.get("depth_m")

            if depth is None or depth <= 0:
                continue

            if not desc:
                continue

            non_empty_strata.append(layer)

            # Check Chinese keywords
            has_geo_zh = any(kw in desc for kw in GEOLOGICAL_KEYWORDS_ZH)
            
            # Check English keywords
            desc_lower = desc.lower()
            has_geo_en = any(kw in desc_lower for kw in GEOLOGICAL_KEYWORDS_EN)

            # Check USCS symbols
            has_uscs = bool(re.search(USCS_PATTERN, desc, re.IGNORECASE))

            # Sub-layer note/condition (describing state under a parent geological layer)
            is_sub_qualifier = prev_has_geo and (
                desc.startswith("~") or desc.startswith("-") or
                any(k in desc for k in ["破碎", "膠結", "高角度", "填充", "浸染", "理面", "劈理", "剪動", "銹染"])
            )

            if has_geo_zh or has_geo_en or has_uscs or is_sub_qualifier:
                valid_count += 1
                prev_has_geo = True
            else:
                prev_has_geo = False

        if not non_empty_strata:
            return False, "所有地層之描述均為空"

        total_layers = len(non_empty_strata)
        if valid_count == 0:
            sample_desc = non_empty_strata[0].get('description', '')[:30]
            return False, f"描述內容未包含土壤或岩石岩性特徵 (範例: '{sample_desc}')"

        if valid_count / total_layers < 0.5:
            return False, f"地質岩性特徵比例過低 ({valid_count}/{total_layers})"

        return True, "驗證通過"
