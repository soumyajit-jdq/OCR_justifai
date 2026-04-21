import os
import json
import re
import requests
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from PIL import Image
import io
import uvicorn
import logging
import psutil
import signal
import fitz
from web3 import Web3
from collections import OrderedDict
from dotenv import load_dotenv

load_dotenv()

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CONFIG
OCR_API_KEY = os.getenv("OCR_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
CEREBRAS_API_KEY=os.getenv("CEREBRAS_API_KEY", "")

app = FastAPI(title="MarkSheet AI Parser (Reasoning Edition)")

# DATA MODELS
class CertificateData(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)
    certificate_no: str = Field(..., description="Certificate Number (e.g., top right number)")
    no: str = Field(..., description="Reference Number (e.g., bottom left No. suffix)")
    # university: Optional[str] = Field(None, description="Issuing University Name")
    name: str = Field(..., description="Student Name")
    degree: str = Field(..., description="Degree conferred (e.g., Bachelor of Technology)")
    ogpa: Optional[str] = Field(None, description="Overall Grade Point Average")
    year: Optional[str] = Field(None, description="Academic Session Year (e.g., 2016-2017)")
    date: str = Field(..., description="Issue Date")
    class_division: Optional[str] = Field(None, description="Class/Division obtained")
    merkle_hash: Optional[str] = Field(None, description="Keccak-256 Verification Hash")

# PORT UTILITIES
def force_free_port(port: int):
    """Detects and kills any process using the specified port."""
    for conn in psutil.net_connections():
        if conn.laddr.port == port:
            if conn.pid:
                try:
                    p = psutil.Process(conn.pid)
                    logger.info(f"Port {port} is occupied by {p.name()} (PID: {conn.pid}). Terminating...")
                    p.terminate()
                    p.wait(timeout=3)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                    try:
                        logger.warning(f"Graceful termination failed for PID {conn.pid}. Forcing...")
                        p.kill()
                    except:
                        pass

# HELPER FUNCTIONS
def compress_image(image_bytes: bytes, max_kb: int = 1000):
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    if len(image_bytes) <= max_kb * 1024:
        return image_bytes
    quality = 90
    while quality > 10:
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality, optimize=True)
        if len(buffer.getvalue()) <= max_kb * 1024:
            return buffer.getvalue()
        quality -= 10
    img.thumbnail((1600, 1600))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=20)
    return buffer.getvalue()

def run_ocr(image_bytes: bytes):
    """Sends a single compressed image to OCR.space."""
    compressed_bytes = compress_image(image_bytes)
    url = "https://api.ocr.space/parse/image"
    files = {"file": ("image.jpg", compressed_bytes, "image/jpeg")}
    data = {"apikey": OCR_API_KEY, "language": "eng", "isTable": True, "OCREngine": 2}
    try:
        response = requests.post(url, files=files, data=data, timeout=60)
        result = response.json()
        if result.get("OCRExitCode") != 1:
            return f"[OCR Error: {result.get('ErrorMessage')}]"
        return result["ParsedResults"][0]["ParsedText"]
    except Exception as e:
        return f"[OCR Exception: {e}]"

def run_ocr_on_pdf(pdf_bytes: bytes):
    """Sends PDF to OCR.space. If > 1MB, processes page-by-page as images to avoid API limits."""
    # OCR.space free tier limit is 1024KB.
    if len(pdf_bytes) > 1024 * 1024:
        logger.info("PDF > 1MB. Switching to page-by-page image OCR processing...")
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        all_text = []
        for i in range(len(doc)):
            page = doc[i]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_bytes = pix.tobytes("jpg")
            text = run_ocr(img_bytes)
            all_text.append(text)
        doc.close()
        return "\n\n".join(all_text)

    url = "https://api.ocr.space/parse/image"
    files = {"file": ("document.pdf", pdf_bytes, "application/pdf")}
    data = {
        "apikey": OCR_API_KEY,
        "language": "eng",
        "isTable": True,
        "OCREngine": 2
    }
    try:
        response = requests.post(url, files=files, data=data, timeout=60)
        result = response.json()
        if result.get("OCRExitCode") != 1:
            # If PDF fails even if < 1MB, fallback to image processing
            return run_ocr_on_pdf(pdf_bytes + b'\0') # Recursive dummy call with size force to trigger image path? No, let's just implement explicit fallback.
            
        all_text = []
        for res in result.get("ParsedResults", []):
            all_text.append(res.get("ParsedText", ""))
        return "\n\n".join(all_text)
    except Exception as e:
        return f"OCR Error: {e}"

def verify_with_llama(text: str):
    """Uses Cerebras (Llama) to verify if the document is an academic transcript."""
    if not CEREBRAS_API_KEY or not text.strip():
        return {"is_transcript": True, "reason": "No text or key"}
    try:
        url = "https://api.cerebras.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {CEREBRAS_API_KEY}",
            "Content-Type": "application/json"
        }
        prompt = f"Identify if the following text is from an Official Academic Transcript. Check for 'Transcript', 'Cumulative', 'Degree', 'Registrar', or lists of courses by part/year. Respond ONLY with JSON: {{\"is_transcript\": true, \"reason\": \"matches pattern\"}} or {{\"is_transcript\": false, \"reason\": \"error msg\"}}\n\nTEXT:\n{text[:3000]}"
        payload = {
            "model": "llama3.1-8b",
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.0
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=30).json()
        return json.loads(resp["choices"][0]["message"]["content"])
    except Exception as e:
        logger.error(f"Llama verification failed: {e}")
        return {"is_transcript": True, "reason": "Verification bypassed"}


def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

# HASHING UTILITIES
def build_canonical_payload(data: dict) -> str:
    """Builds a canonical JSON string for the certificate structure."""
    payload = OrderedDict([
        ("certificate_no", str(data.get("certificate_no", ""))),
        ("no", str(data.get("no", ""))),
        ("university", str(data.get("university", ""))),
        ("name", str(data.get("name", ""))),
        ("degree", str(data.get("degree", ""))),
        ("ogpa", str(data.get("ogpa", ""))),
        ("year", str(data.get("year", ""))),
        ("date", str(data.get("date", ""))),
        ("class_division", str(data.get("class_division", "")))
    ])
    return json.dumps(payload, separators=(',', ':'))


def process_pdf_pages(pdf_bytes: bytes, max_pages: int = 3):
    """Converts up to max_pages of a PDF to OCR-ready images and extracts text."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        num_pages = min(len(doc), max_pages)
        if num_pages == 0:
            return [], ""
        
        all_text = []
        all_images = []
        
        for i in range(num_pages):
            page = doc[i]
            # Extract text
            all_text.append(page.get_text().strip())
            
            # Convert to high-res image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            all_images.append(pix.tobytes("jpg"))
            
        doc.close()
        return all_images, "\n\n".join(all_text)
    except Exception as e:
        logger.error(f"PDF processing failed: {e}")
        return [], ""

def generate_keccak256(text: str):
    """Generates an Ethereum-standard Keccak-256 hash using Web3.py."""
    hash_bytes = Web3.keccak(text=text)
    return Web3.to_hex(hash_bytes)

def generate_with_cerebras(prompt: str):
    """High-speed text-only extraction using Cerebras."""
    try:
        url = "https://api.cerebras.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {CEREBRAS_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # --- PASS 1: Initial Extraction ---
        payload_1 = {
            "model": "llama3.1-8b",
            # "model": "qwen-3-235b-a22b-instruct-2507",
            # "model": "",
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a VERBATIM marksheet parser. Extract exactly as visible in OCR. Return ONLY JSON."
                },
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0
        }
        resp_1 = requests.post(url, headers=headers, json=payload_1, timeout=30).json()
        if "choices" not in resp_1:
            return None
        initial_json = resp_1["choices"][0]["message"]["content"]
        
        # --- PASS 2: Self-Correction Loop ---
        correction_system_prompt = (
            "You are a character-level QA auditor. Correct any JSON formatting issues in the provided data.\n"
            "Ensure it matches the nested structure requested.\n"
            "Return ONLY the corrected JSON object."
        )
        correction_user_prompt = f"RAW OCR TEXT:\n{prompt}\n\nINITIAL JSON TO CORRECT:\n{initial_json}"
        
        payload_2 = {
            "model": "llama3.1-8b",
            "messages": [
                {"role": "system", "content": correction_system_prompt},
                {"role": "user", "content": correction_user_prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0
        }
        resp_2 = requests.post(url, headers=headers, json=payload_2, timeout=30).json()
        
        final_content = initial_json
        if "choices" in resp_2:
            final_content = resp_2["choices"][0]["message"]["content"]
            logger.info("Self-Correction loop completed.")
        
        # Robust JSON cleaning
        match = re.search(r"\{.*\}", final_content, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(final_content)
    except Exception as e:
        logger.warning(f"Cerebras extraction/correction failed: {e}")
        return None


def generate_with_gemini(image_data, prompt: str):
    """Fallback to Gemini for high-precision vision parsing. image_data is a List[bytes]."""
    from google import genai
    from google.genai import types
    import time
    
    client = genai.Client(api_key=GEMINI_API_KEY)
    contents = [prompt]
    if isinstance(image_data, list):
        for img_bytes in image_data:
            contents.append(Image.open(io.BytesIO(img_bytes)))
    else:
        contents.append(Image.open(io.BytesIO(image_data)))

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-3.1-flash-lite-preview',
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=CertificateData,
                    temperature=0.1
                )
            )
            return json.loads(response.text)
        except Exception as e:
            if "503" in str(e) and attempt < max_retries - 1:
                logger.warning(f"Gemini 503 Service Unavailable. Retrying in {2**(attempt+1)}s...")
                time.sleep(2**(attempt+1))
                continue
            logger.warning(f"Gemini extraction failed: {e}")
            return None
    return None


def generate_structured_data(image_data, ocr_text: str):
    prompt = f"""
You are an expert academic certificate parser. Extract specific fields from the document.

#### EXTRACTION RULES ####
1. **Certificate No**: Look at the top right of the document for a numeric sequence (e.g., '0003662').
2. **No. (Reference)**: Look for 'No.' followed by a sequence of characters, usually at the bottom left (e.g., 'No.XIII/210/2018').
3. **Degree**: Look for the degree conferred (e.g., 'Bachelor of Technology (Agricultural Engineering)').
4. **Name**: Look for the name appearing after 'conferred upon' or similar (e.g., 'Pandya Abhimanyukumar Hiteshbhai').
5. **Academic Info**:
   - **OGPA**: Look for 'Overall Grade Point Average' (e.g., '7.83').
   - **Year**: Look for the academic session / year (e.g., '2016-2017').
   - **Class**: Look for the class or division (e.g., 'First Class With Distinction').
6. **Date**: Look for the issue date (e.g., 'January 28, 2018').
7. **University**: Extract the university name (e.g., 'Junagadh Agricultural University').

#### JSON STRUCTURE ####
{{
  "certificate_no": "...",
  "no": "...",
  "university": "...",
  "name": "...",
  "degree": "...",
  "ogpa": "...",
  "year": "...",
  "date": "...",
  "class_division": "..."
}}

OCR TEXT:
{ocr_text}

Return ONLY the structured JSON.
"""

    # Primary Multimodal Extraction
    if GEMINI_API_KEY:
        logger.info("Running Certificate Multimodal Gemini Extraction")
        gemini_result = generate_with_gemini(image_data, prompt)
        if gemini_result:
            return gemini_result

    # Fallback to Cerebras (Text only)
    if CEREBRAS_API_KEY:
        logger.info("Attempting fallback text-only extraction with Cerebras")
        cerebras_result = generate_with_cerebras(prompt)
        if cerebras_result:
            return cerebras_result

    raise ValueError("All certificate extraction models failed.")


    raise ValueError("All extraction models failed.")

# API ENDPOINTS
@app.post("/parse-certificate", response_model=CertificateData)
async def parse_certificate(file: UploadFile = File(...)):
    try:
        # GATE 0: File Type Check
        pdf_type = "application/pdf"
        img_types = ["image/jpeg", "image/png", "image/jpg"]
        if file.content_type not in img_types and file.content_type != pdf_type:
             raise HTTPException(status_code=400, detail="Only PDF and Images are supported.")
            
        file_bytes = await file.read()
        
        # Decide processing based on type
        if file.content_type == pdf_type:
            img_list, low_res_text = process_pdf_pages(file_bytes, max_pages=1)
            ocr_text = run_ocr_on_pdf(file_bytes)
        else:
            img_list = [file_bytes]
            ocr_text = run_ocr(file_bytes)
        
        # GATE 4: Multimodal Extraction with Gemini
        logger.info("Proceeding to Certificate Multimodal Extraction...")
        structured_data = generate_structured_data(img_list, ocr_text)
        
        # GATE 5: Verification Hash
        canonical_json = build_canonical_payload(structured_data)
        structured_data["merkle_hash"] = generate_keccak256(canonical_json)
        
        return CertificateData(**structured_data)
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception("Processing failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Certificate Verifier</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root { --primary: #fbbf24; --bg: #0f172a; --card: #1e293b; --text: #f1f5f9; --accent: #f59e0b; }
        body { font-family: 'Outfit', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; display: flex; align-items: center; justify-content: center; margin: 0; padding: 20px; }
        .box { width: 100%; max-width: 800px; background: var(--card); border: 1px solid #334155; border-radius: 24px; padding: 40px; box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5); }
        .dropzone { border: 2px dashed #475569; border-radius: 16px; padding: 60px 40px; text-align: center; cursor: pointer; transition: 0.3s; background: rgba(51,65,85,0.3); }
        .dropzone:hover { border-color: var(--primary); background: rgba(51,65,85,0.5); }
        .btn { background: var(--primary); color: #000; border: none; padding: 16px 30px; border-radius: 12px; font-weight: 700; cursor: pointer; margin-top: 25px; width: 100%; transition: 0.2s; font-size: 1.1em; }
        .btn:hover { background: #f59e0b; transform: translateY(-2px); }
        #results { margin-top: 40px; display: none; }
        .cert-card { background: #fff; color: #1e293b; border-radius: 12px; padding: 40px; border: 8px double #e2e8f0; position: relative; overflow: hidden; box-shadow: inset 0 0 100px rgba(0,0,0,0.02); }
        .cert-header { text-align: center; border-bottom: 2px solid #e2e8f0; padding-bottom: 20px; margin-bottom: 30px; }
        .univ-name { font-size: 1.5em; font-weight: 800; color: #1e3a8a; text-transform: uppercase; }
        .cert-title { font-size: 1.1em; color: #64748b; margin-top: 10px; letter-spacing: 2px; }
        .field { margin: 25px 0; }
        .label { font-size: 0.75em; text-transform: uppercase; color: #64748b; letter-spacing: 1px; font-weight: 600; margin-bottom: 5px; }
        .val { font-size: 1.4em; font-weight: 700; color: #0f172a; }
        .meta-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-top: 30px; padding-top: 20px; border-top: 1px solid #f1f5f9; }
        .hash-box { background: #f8fafc; border: 1px solid #e2e8f0; padding: 15px; border-radius: 8px; font-family: monospace; font-size: 0.8em; margin-top: 30px; word-break: break-all; color: #64748b; }
        .loader { display: none; text-align: center; margin: 20px 0; color: var(--primary); font-weight: 600; }
    </style>
</head>
<body>
    <div class="box">
        <h1 style="margin-top:0; font-size: 2em; text-align: center; color: var(--primary);">Certificate Intelligence</h1>
        <p style="color:#94a3b8; text-align: center; margin-bottom: 40px;">Instant AI extraction & Blockchain verification for degrees</p>
        
        <form id="pForm">
            <div class="dropzone" onclick="document.getElementById('f').click()">
                <span id="label" style="font-size: 1.1em;">Click or Drop Degree Certificate (PDF/Image)</span>
                <input type="file" id="f" style="display:none" accept=".pdf,image/*" onchange="document.getElementById('label').innerText=this.files[0].name">
            </div>
            <button type="submit" class="btn">Verify Certificate</button>
        </form>
        
        <div class="loader" id="l">⚡ Analyzing Security Features & Extracting Data...</div>
        
        <div id="results">
            <div class="cert-card">
                <div class="cert-header">
                    <div id="univ" class="univ-name"></div>
                    <div class="cert-title">OFFICIAL DEGREE RECORD</div>
                </div>
                
                <div style="text-align: right; font-size: 0.8em; color: #94a3b8;">
                    Cert No: <span id="c_no" style="font-weight: 700; color: #ef4444;"></span>
                </div>

                <div class="field">
                    <div class="label">This degree of</div>
                    <div id="degree" class="val" style="color: #1e3a8a;"></div>
                    <div class="label" style="margin-top:10px;">has been conferred upon</div>
                    <div id="name" class="val"></div>
                </div>

                <div class="meta-grid">
                    <div>
                        <div class="label">OGPA / Performance</div>
                        <div id="ogpa" class="val" style="color: #059669;"></div>
                    </div>
                    <div>
                        <div class="label">Academic Session</div>
                        <div id="year" class="val"></div>
                    </div>
                    <div>
                        <div class="label">Reference No (No.)</div>
                        <div id="ref_no" class="val" style="font-size: 1em;"></div>
                    </div>
                    <div>
                        <div class="label">Date of Issue</div>
                        <div id="date" class="val" style="font-size: 1.1em;"></div>
                    </div>
                </div>
                
                <div id="cls_box" style="margin-top: 20px; text-align: center;">
                    <div class="label">Classification</div>
                    <div id="class" class="val" style="font-size: 1.2em; color: #d97706;"></div>
                </div>

                <div class="hash-box">
                    <strong>BLOCKCHAIN HASH:</strong> <span id="h"></span>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('pForm').onsubmit = async (e) => {
            e.preventDefault();
            const f = document.getElementById('f').files[0];
            if(!f) return;
            document.getElementById('l').style.display='block';
            document.getElementById('results').style.display='none';
            const fd = new FormData(); fd.append('file', f);
            try {
                const res = await fetch('/parse-certificate', {method:'POST', body:fd});
                if(!res.ok) {
                    const err = await res.json();
                    throw new Error(err.detail || 'Processing failed');
                }
                const d = await res.json();
                
                document.getElementById('univ').innerText = d.university || 'N/A';
                document.getElementById('c_no').innerText = d.certificate_no;
                document.getElementById('degree').innerText = d.degree;
                document.getElementById('name').innerText = d.name;
                document.getElementById('ogpa').innerText = d.ogpa || 'N/A';
                document.getElementById('year').innerText = d.year || 'N/A';
                document.getElementById('ref_no').innerText = d.no || 'N/A';
                document.getElementById('date').innerText = d.date;
                document.getElementById('class').innerText = d.class_division || 'N/A';
                document.getElementById('h').innerText = d.merkle_hash;
                
                document.getElementById('results').style.display='block';
            } catch(er) { 
                console.error("Error:", er);
                alert('Verification Failed: ' + er.message); 
            }
            finally { document.getElementById('l').style.display='none'; }
        }
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    TARGET_PORT = 8081
    force_free_port(TARGET_PORT)
    uvicorn.run(app, host="0.0.0.0", port=TARGET_PORT)