import os
import cv2
import numpy as np
import openai
import base64
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
import pytesseract
import json
import requests
from collections import defaultdict, Counter

# FastAPI App initialisieren
app = FastAPI()

# Zugriff auf die Umgebungsvariablen
openai_api_key = os.getenv("OPENAI_API_KEY")
github_token = os.getenv("GITHUB_TOKEN")

# Überprüfen, ob die Umgebungsvariablen gesetzt sind
if openai_api_key is None or github_token is None:
    raise ValueError("API-Keys sind nicht korrekt gesetzt!")

# Funktion zum Bildverarbeiten und Erkennen von Symbolen
def detect_symbols_with_balanced_filtering(image):
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]
    return bounding_boxes

def classify_symbol_with_openai_from_image(image, box):
    x, y, w, h = box
    symbol = image[y:y+h, x:x+w]
    img_base64 = encode_image_to_base64(symbol)

    # OCR-Erkennung des Textes in der Nähe des Symbols
    text_area = image[y:y+h, x+w:x+w+2000]
    ocr_text = pytesseract.image_to_string(text_area).strip()

    # OpenAI API Anfrage
    prompt = f'''
    Du bekommst links ein Symbolbild und rechts angrenzenden Beschreibungstext.
    Verwende das symbol nur, wenn der Text "Egcobox" oder "Isokorb" enthält.

    Beschreibungstext:
    """{ocr_text}"""

    Antworte mit: "verwenden" oder "ignorieren"
    '''

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=10
    )


    decision = response['choices'][0]['message']['content'].strip().lower()
    return {"entscheidung": decision, "ocr_text": ocr_text}

def encode_image_to_base64(image):
    _, buffer = cv2.imencode(".png", image)
    return base64.b64encode(buffer).decode("utf-8")

def match_template_on_large_plan(plan_image, templates):
    all_matches = []

    for template in templates:
        result = cv2.matchTemplate(plan_image, template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= 0.92)  # Threshold 0.92

        for pt in zip(*locations[::-1]):
            all_matches.append({
                "position": {"x": int(pt[0]), "y": int(pt[1])},
                "confidence": float(result[pt[1], pt[0]]),
                "bounding_box": [int(pt[0]), int(pt[1]), template.shape[1], template.shape[0]]
            })

    return all_matches

# POST-Endpunkt für den Empfang der Bilder
@app.post("/upload/")
async def upload_file(plan_image: UploadFile = File(...), verzeichnis_image: UploadFile = File(...)):
    print("Received request to upload files.")
    # Empfange die Bilddaten als Bytes und dekodiere sie
    plan_bytes = await plan_image.read()
    verzeichnis_bytes = await verzeichnis_image.read()

    # Lese die Bytes in ein OpenCV-Bild
    plan_image_cv = cv2.imdecode(np.frombuffer(plan_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    verzeichnis_image_cv = cv2.imdecode(np.frombuffer(verzeichnis_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)

    # Verarbeite die Bilder
    bounding_boxes = detect_symbols_with_balanced_filtering(verzeichnis_image_cv)
    templates = []

    for box in bounding_boxes:
        result = classify_symbol_with_openai_from_image(verzeichnis_image_cv, box)
        if result["entscheidung"] == "verwenden":
            x, y, w, h = box
            template = verzeichnis_image_cv[y:y+h, x:x+w]
            templates.append(template)

    # Starte das Matching des Plans auf die Templates
    matches = match_template_on_large_plan(plan_image_cv, templates)

    # Ergebnisse zurückgeben
    return {"status": "success", "matches": matches}


    # Ergebnisse zurückgeben
    return {"status": "success", "matches": matches}
