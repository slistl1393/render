from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
import uuid

from grundfunktion import run_full_pipeline

app = FastAPI()

import subprocess
result = subprocess.run(['tesseract', '--version'], stdout=subprocess.PIPE)
print(result.stdout.decode())

@app.post("/process")
async def process_files(
    plan: UploadFile = File(...),
    verzeichnis: UploadFile = File(...)
):
    session_id = str(uuid.uuid4())
    session_dir = f"/tmp/session_{session_id}"
    os.makedirs(session_dir, exist_ok=True)

    plan_path = os.path.join(session_dir, "plan.png")
    verzeichnis_path = os.path.join(session_dir, "verzeichnis.png")

    with open(plan_path, "wb") as f:
        shutil.copyfileobj(plan.file, f)

    with open(verzeichnis_path, "wb") as f:
        shutil.copyfileobj(verzeichnis.file, f)

    try:
        json_result = run_full_pipeline(plan_path, verzeichnis_path, session_dir)
        return JSONResponse(content=json_result)
    except Exception as e:
        return {"error": str(e)}
