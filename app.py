# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import openai
import base64
from fastapi import FastAPI, File, UploadFile
import pytesseract
from typing import List

from collections import defaultdict

# Umgebungsvariablen laden
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()


# 🔹 Base64-Kodierung für OpenAI

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# 🔹 Symbolerkennung

def detect_symbols_with_balanced_filtering(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_eq = cv2.equalizeHist(image_gray)
    _, thresh = cv2.threshold(image_eq, 150, 255, cv2.THRESH_BINARY_INV)
    median_val = np.median(image_eq)
    lower = int(max(0, 0.5 * median_val))
    upper = int(min(255, 1.5 * median_val))
    edged = cv2.Canny(thresh, lower, upper)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        x, y, w, h = cv2.boundingRect(approx)
        area = w * h
        ratio = w / float(h)
        if 30 < w < 300 and 30 < h < 300 and area > 800 and 0.4 < ratio < 2.2:
            boxes.append((x, y, w, h))
    return boxes


# 🔹 OpenAI Klassifikation

def classify_symbol_with_openai_from_image(image, box):
    x, y, w, h = box
    symbol = image[y:y+h, x:x+w]
    _, buffer = cv2.imencode(".png", symbol)
    b64_symbol = base64.b64encode(buffer).decode("utf-8")
    text_area = image[y:y+h, x+w:x+w+1000]
    ocr_text = pytesseract.image_to_string(text_area).strip()

    prompt = f'''
    Du bekommst links ein Symbolbild und rechts angrenzenden Beschreibungstext.
    Verwende das Symbol nur, wenn der Text \"Egcobox\" oder \"Isokorb\" enthält.

    Beschreibungstext:
    """{ocr_text}"""

    Antworte mit: \"verwenden\" oder \"ignorieren\"
    '''

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=10
    )

    return response.choices[0].message.content.strip().lower()


# 🔹 Template Matching

def match_template(plan, template, template_name, threshold=0.85):
    plan_gray = cv2.cvtColor(plan, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    plan_gray = cv2.equalizeHist(plan_gray)
    template_gray = cv2.equalizeHist(template_gray)
    h, w = template_gray.shape

    result = cv2.matchTemplate(plan_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)

    matches = []
    for pt in zip(*locations[::-1]):
        matches.append({
            "template": template_name,
            "position": {"x": int(pt[0]), "y": int(pt[1])},
            "confidence": float(result[pt[1], pt[0]]),
            "bounding_box": [int(pt[0]), int(pt[1]), w, h]
        })
    return matches


# 🔹 NMS

def non_max_suppression_per_template(matches, iou_threshold=0.3):
    def compute_iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xa, ya = max(x1, x2), max(y1, y2)
        xb, yb = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        inter_area = max(0, xb - xa) * max(0, yb - ya)
        union_area = w1 * h1 + w2 * h2 - inter_area
        return inter_area / union_area if union_area != 0 else 0

    grouped = defaultdict(list)
    for m in matches:
        grouped[m['template']].append(m)

    filtered = []
    for group in grouped.values():
        boxes = [m["bounding_box"] for m in group]
        scores = [m.get("confidence", 1.0) for m in group]
        indices = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)

        keep = []
        while indices:
            current = indices.pop(0)
            keep.append(current)
            indices = [i for i in indices if compute_iou(boxes[current], boxes[i]) < iou_threshold]

        filtered.extend([group[i] for i in keep])
    return filtered


# 🔹 FastAPI-Endpunkt
@app.post("/upload/")
async def upload_file(plan_image: UploadFile = File(...), verzeichnis_image: UploadFile = File(...)):
    try:
        print("➡️ Anfrage empfangen")
        plan_bytes = await plan_image.read()
        verzeichnis_bytes = await verzeichnis_image.read()
        print("✅ Bilder empfangen")

        # Hier folgen deine weiteren Schritte (z. B. Symbolerkennung etc.)
        # und an jedem Schritt so etwas einbauen:
        print("▶️ Starte Symbolerkennung...")

        # ...

        return {"status": "success", "matches": []}  # Beispielantwort
    except Exception as e:
        print(f"❌ Fehler im Backend: {e}")
        return {"status": "error", "message": str(e)}


    plan = cv2.imdecode(np.frombuffer(plan_bytes, np.uint8), cv2.IMREAD_COLOR)
    verzeichnis = cv2.imdecode(np.frombuffer(verzeichnis_bytes, np.uint8), cv2.IMREAD_COLOR)

    boxes = detect_symbols_with_balanced_filtering(verzeichnis)
    templates = []
    for idx, box in enumerate(boxes):
        decision = classify_symbol_with_openai_from_image(verzeichnis, box)
        if decision == "verwenden":
            x, y, w, h = box
            temp = verzeichnis[y:y+h, x:x+w]
            templates.append((f"template_{idx}", temp))

    all_matches = []
    for name, template in templates:
        matches = match_template(plan, template, name)
        all_matches.extend(matches)

    final_matches = non_max_suppression_per_template(all_matches)
    return {"matches": final_matches, "count": len(final_matches)}

