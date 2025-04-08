import os
import cv2
import json
import base64
import pytesseract
import numpy as np
import requests
from collections import defaultdict, Counter
from openai import OpenAI

# OpenAI-Key und GitHub-Token aus Umgebungsvariablen
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


def encode_image_to_base64(image):
    _, buffer = cv2.imencode(".png", image)
    return base64.b64encode(buffer).decode("utf-8")


def detect_symbols(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        area = w * h
        if 30 < w < 300 and 30 < h < 300 and area > 800 and 0.4 < aspect_ratio < 2.2:
            boxes.append((x, y, w, h))
    return boxes


def classify_with_openai(image, box):
    x, y, w, h = box
    symbol = image[y:y+h, x:x+w]
    text_area = image[y:y+h, x+w:x+w+2000]
    ocr_text = pytesseract.image_to_string(text_area).strip()
    img_base64 = encode_image_to_base64(symbol)

    prompt = f'''
    Du bekommst ein Symbolbild und angrenzenden Beschreibungstext.
    Verwende es nur, wenn der Text "Egcobox" oder "Isokorb" enthält.

    Beschreibungstext:
    """{ocr_text}"""

    Antworte mit: "verwenden" oder "ignorieren"
    '''

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                ]
            }
        ],
        max_tokens=10
    )
    decision = response.choices[0].message.content.strip().lower()
    print(f"🔍 Entscheidung: {decision} | OCR: {ocr_text}")
    return decision, ocr_text


def match_template(plan_gray, template_gray, template_name, threshold=0.92):
    h_temp, w_temp = template_gray.shape
    result = cv2.matchTemplate(plan_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)
    matches = []
    for pt in zip(*locations[::-1]):
        score = float(result[pt[1], pt[0]])
        matches.append({
            "template": template_name,
            "position": {"x": int(pt[0]), "y": int(pt[1])},
            "confidence": score,
            "bounding_box": [int(pt[0]), int(pt[1]), w_temp, h_temp]
        })
    return matches


def nms(matches, iou_threshold=0.3):
    def iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xa, ya = max(x1, x2), max(y1, y2)
        xb, yb = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        inter = max(0, xb - xa) * max(0, yb - ya)
        union = w1 * h1 + w2 * h2 - inter
        return inter / union if union else 0

    grouped = defaultdict(list)
    for m in matches:
        grouped[m['template']].append(m)

    final = []
    for group in grouped.values():
        group.sort(key=lambda x: x.get('confidence', 1.0), reverse=True)
        keep = []
        while group:
            current = group.pop(0)
            keep.append(current)
            group = [g for g in group if iou(current['bounding_box'], g['bounding_box']) < iou_threshold]
        final.extend(keep)
    return final


def upload_to_github(path, remote_name):
    with open(path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf-8")

    repo = "slistl1393/Template-Matching"
    api_url = f"https://api.github.com/repos/{repo}/contents/json_output/{remote_name}.json"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}

    r = requests.get(api_url, headers=headers)
    sha = r.json().get("sha") if r.status_code == 200 else None
    payload = {
        "message": f"Update {remote_name}",
        "content": content,
        "branch": "main"
    }
    if sha:
        payload["sha"] = sha

    requests.put(api_url, headers=headers, json=payload)


def run_full_pipeline(plan_path, verzeichnis_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plan = cv2.imread(plan_path)
    verzeichnis = cv2.imread(verzeichnis_path)

    if plan is None or verzeichnis is None:
        raise ValueError("⛔ plan.png oder verzeichnis.png konnte nicht geladen werden.")

    boxes = detect_symbols(verzeichnis)
    templates = []

    for box in boxes:
        try:
            decision, bauteil = classify_with_openai(verzeichnis, box)
        except Exception as e:
            print(f"❌ Fehler bei classify_with_openai: {e}")
            continue
        if decision == "verwenden":
            x, y, w, h = box
            symbol = verzeichnis[y:y+h, x:x+w]
            template_path = os.path.join(output_dir, f"template_{x}_{y}.png")
            cv2.imwrite(template_path, symbol)
            templates.append({"path": template_path, "name": f"template_{x}_{y}", "bauteil": bauteil})

    plan_gray = cv2.cvtColor(plan, cv2.COLOR_BGR2GRAY)
    all_matches = []
    for tpl in templates:
        tpl_gray = cv2.imread(tpl["path"], cv2.IMREAD_GRAYSCALE)
        matches = match_template(plan_gray, tpl_gray, tpl["name"])
        for m in matches:
            m["bauteil"] = tpl["bauteil"]
        all_matches.extend(matches)

    filtered = nms(all_matches)

    json_output_dir = os.path.join(output_dir, "json_output")
    os.makedirs(json_output_dir, exist_ok=True)

    by_template = defaultdict(list)
    for m in filtered:
        by_template[m["template"]].append(m)

    for template_name, data in by_template.items():
        fname = os.path.join(json_output_dir, f"{template_name}.json")
        with open(fname, "w") as f:
            json.dump({"template": template_name, "count": len(data), "matches": data}, f, indent=2)
        upload_to_github(fname, template_name)

    summary = Counter([m["template"] for m in filtered])
    streamlit_export = {"summary": dict(summary), "matches": filtered}

    export_path = os.path.join(output_dir, "streamlit_export.json")
    with open(export_path, "w") as f:
        json.dump(streamlit_export, f, indent=2)

    return streamlit_export
