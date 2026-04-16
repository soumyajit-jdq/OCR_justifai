import os
import sys
import json
import argparse
import time
import base64
import re
from typing import List, Optional, Dict, Union, Any
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor, as_completed

# Override print to output to stderr by default
import builtins
_original_print = builtins.print
def _stderr_print(*args, **kwargs):
    if 'file' not in kwargs:
        kwargs['file'] = sys.stderr
    _original_print(*args, **kwargs)
builtins.print = _stderr_print

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Determine available libraries
try:
    from google import genai
    from google.genai import types
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

# Define the Target JSON Schema using Pydantic
class KeyValueField(BaseModel):
    key: str = Field(description="The exact label text from the document, preserving all characters including colons, spaces, and supplementary words (e.g., 'Name :', 'Enrollment No. :')")
    value: str = Field(description="The extracted value for this field")

class TableColumn(BaseModel):
    header: str = Field(description="The main header text for this column")
    sub_headers: Optional[List[str]] = Field(None, description="List of sub-headers if this is a grouped column (e.g. ['Internal', 'External', 'Total'] under 'Passing Marks')")

class DataTable(BaseModel):
    table_name: Optional[str] = Field(None, description="A descriptive name for the table if explicitly labeled in the document")
    columns: Optional[List[TableColumn]] = Field(None, description="The hierarchical column structure of the table, if present")
    rows: List[List[str]] = Field(description="The data rows. Each row is a simple list of cell values corresponding exactly to the flat sequence of all leaf headers/sub-headers.")

class GenericDocumentData(BaseModel):
    issuing_authority: str = Field(description="The organization that issued the document (e.g., University Name, Board Name)")
    document_title: str = Field(description="The main title of the document (e.g., 'Statement of Marks', 'Bachelor of Science Semester-V', 'certificate')")
    main_details: List[KeyValueField] = Field(description="General key-value information extracted from the document")
    tables: List[DataTable] = Field(description="Any structured or tabular data found in the document (like lists of subjects, marks, or any other tables)")


# Parsing Functions using AI Models
def parse_with_gemini(image_input: Union[str, Image.Image], api_key: str, max_retries: int = 4) -> str:
    """Uses Google's Gemini API with native Vision capabilities and Structured Outputs"""
    if not HAS_GEMINI:
        raise ImportError("google-genai package is not installed. Run `pip install google-genai`")
    if not HAS_PIL:
        raise ImportError("pillow package is not installed. Run `pip install pillow`")
    
    client = genai.Client(api_key=api_key)
    
    # --- ABSOLUTE DETERMINISM LOCK ---
    # models_to_try = [
    #     'gemini-flash-latest',
    #     'gemini-3.1-flash-lite-preview',
    #     'gemini-3-pro-preview',
    #     'gemini-2.5-pro',          # Priority 1: Current Stable Flagship
    #     'gemini-2.5-flash',        # Priority 2: Stable Performance
    #     'gemini-3.1-pro-preview',  # Priority 3: Advanced Preview Fallback
    # ]
    # We lock to ONE architecture. Switching models breaks Merkle hashes.
    TARGET_MODEL = 'gemini-3.1-flash-lite-preview'
    MAX_RETRIES = 5

    last_exception = None

    prompt = (
        "Analyze the provided images and extract all structured details into JSON with 100% literal precision.\n\n"
        "BIT-PERFECT CONSISTENCY REQUIRED:\n"
        "1. This data is used for Merkle tree hashing. Any variation is a failure.\n"
        "2. CONTEXT AWARENESS: You are provided with TWO images:\n"
        "   - Image 1 is the FULL page.\n"
        "   - Image 2 is a HIGH-RESOLUTION ZOOM of the Header area.\n"
        "3. SOURCE OF TRUTH: Use Image 2 (the zoom) to extract and verify all fields in 'main_details' (Date, Reg No, Name, etc.). Use Image 1 for overall layout and tables.\n"
        "4. MANDATORY FIELDS: You MUST find and extract 'Date', 'Registration No', 'Academic Year', and 'GPA'. They are critical for hashing.\n"
        "5. NO CLEANUP: Extraction must be character-by-character literal.\n"
        "6. TRILINGUAL SUPPORT: English, Gujarati, and Hindi. Extract exactly as visible.\n\n"
        "DOCUMENT STRUCTURE:\n"
        "- `document_title`: Main title.\n"
        "- `main_details`: List of key-value pairs.\n"
        "- `tables`: Capture all rows. Ensure alignment."
    )

    # If input is a list, it's already [Full, Zoom]
    if isinstance(image_input, list):
        imgs = image_input
    else:
        # For single image input, generate the zoom crop on the fly
        if isinstance(image_input, str):
            img = Image.open(image_input)
        else:
            img = image_input
        
        # Zoom crop (Top 25%)
        w, h = img.size
        zoom_img = img.crop((0, 0, w, int(h * 0.25)))
        imgs = [img, zoom_img]

    for attempt in range(MAX_RETRIES):
        try:
            print(f"  [{TARGET_MODEL}] Extraction Attempt {attempt + 1}/{MAX_RETRIES}...")
            # Prepare content parts: prompt + all images
            contents = [prompt] + imgs
            
            response = client.models.generate_content(
                model=TARGET_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GenericDocumentData,
                    temperature=0.0,
                    seed=42, 
                ),
            )
            return response.text
        except Exception as e:
            last_exception = e
            error_str = str(e)
            if any(token in error_str for token in ["503", "429", "UNAVAILABLE", "demand"]):
                wait_time = (attempt + 1) * 2
                print(f"  [{TARGET_MODEL}] API Busy/Rate-limited. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                print(f"  [{TARGET_MODEL}] Failed: {error_str[:150]}")
                break
                    
    raise Exception(f"Deterministic extraction failed. Last error: {str(last_exception)}")


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def parse_with_openai(image_input: Union[str, Image.Image], api_key: str) -> str:
    """Uses OpenAI's API with native Vision capabilities and Structured Outputs"""
    if not HAS_OPENAI:
        raise ImportError("openai package is not installed. Run `pip install openai`")
    
    client = OpenAI(api_key=api_key)
    
    if isinstance(image_input, str):
        base64_image = encode_image(image_input)
    else:
        import io
        buffered = io.BytesIO()
        image_input.save(buffered, format="JPEG")
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    prompt = (
        "Analyze the image and extract all structured details into JSON with 100% literal precision.\n\n"
        "BIT-PERFECT CONSISTENCY REQUIRED:\n"
        "1. This data is used for Merkle tree hashing. Any variation (even a missing colon, space, or special mark) is a total failure.\n"
        "2. NO CLEANUP: Extraction must be character-by-character literal. If a label has a colon (e.g., 'DATE :'), the JSON key MUST include that colon exactly as it appears. Do not normalize labels.\n"
        "3. ZERO OMISSIONS: Capture EVERY piece of text. Do not skip GPA sections, NOTE definitions at the bottom, or Remarks. If it is visible, it must be in the JSON.\n"
        "4. CROSS-MODAL INVARIANCE: Whether this is a perfect PDF render or a mobile photo, you must extract the EXACT SAME text tokens. Ignore visual noise/shadows and focus only on characters.\n\n"
        "TRILINGUAL SUPPORT: English, Gujarati, and Hindi. Extract text EXACTLY as it appears. DO NOT translate.\n\n"
        "DOCUMENT STRUCTURE:\n"
        "- `document_title`: Main title of the document.\n"
        "- `main_details`: List of key-value pairs. Keys MUST include punctation (colons, etc.) exactly.\n"
        "- `tables`: Capture all rows. Ensure vertical alignment is perfect. Use empty strings for empty cells.\n\n"
        "SANITY CHECK: Before completing, re-read the page and confirm every visible line of text is accounted for in your JSON."
    )
    
    completion = client.beta.chat.completions.parse(
        model="gpt-5.4-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that parses trilingual documents (English, Gujarati, Hindi) into a structured JSON format. You prioritize extreme precision and completeness for blockchain verification."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }}
            ]}
        ],
        response_format=GenericDocumentData,
        temperature=0.0,
        seed=42
    )
    return completion.choices[0].message.parsed.model_dump_json(indent=2)


def normalize_key(key: str) -> str:
    """Zero-variance normalization: Unicode-aware stripping, lowercase, maps synonyms."""
    import re
    # Remove Western punctuation and spaces, but PRESERVE Unicode word characters (Gujarati/Hindi/English)
    # \W matches anything that is NOT a Unicode word character. Flags=re.UNICODE ensures script safety.
    k = re.sub(r'[\W\s_]+', '', key, flags=re.UNICODE).lower().strip()
    
    # HARD SYNONYM MAP (Pre-stripped keys)
    synonyms = {
        "motto": "institutionalslogan",
        "logotext": "institutionalslogan",
        # Pinned Academic Keys
        "academicyear": "academicyear",
        "date": "date",
        "dated": "date",
        "regno": "registrationno",
        "regino": "registrationno",
        "enrollmentno": "registrationno",
        "seatno": "registrationno",
        "registrationno": "registrationno",
        # Language/Domain Keys
        "grade": "gpa",
        "gradepointaverage": "gpa",
        "serialno": "sn",
        "srno": "sn",
        "coursename": "titleofcourses",
        "total": "sumtotal",
    }
    return synonyms.get(k, k)

def recursive_sort(obj: Any) -> Any:
    """Recursively sorts AND prunes Nulls/EmptyStrings for ultra-dense Merkle consistency."""
    if isinstance(obj, dict):
        # 1. Prune Nulls and Empty Strings, then sort
        cleaned_dict = {
            k: recursive_sort(v) 
            for k, v in sorted(obj.items()) 
            if v is not None and v != ""
        }
        # 2. If the dict is now empty, return None (to be pruned by parent)
        return cleaned_dict if cleaned_dict else None
        
    if isinstance(obj, list):
        # 3. Prune Empty Strings/Nulls/EmptyContainers and recurse
        cleaned_list = []
        for x in obj:
            item = recursive_sort(x)
            if item is not None and item != "" and item != {}:
                cleaned_list.append(item)
        
        try:
            return sorted(cleaned_list, key=lambda x: str(x))
        except:
            return cleaned_list
            
    return obj

def normalize_value(val: str) -> str:
    """Aggressive value normalization for dates, numbers, and casing."""
    import re
    v = str(val).strip().lower()
    
    # 1. Punctuation Scrubbing: Remove jitter at the start/end of values (colons, dots, dashes)
    v = re.sub(r'^[:\.\-\s]+', '', v)
    v = re.sub(r'[:\.\-\s]+$', '', v)
    
    # 2. Date Padding
    date_match = re.match(r'^(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})$', v)
    if date_match:
        d, m, y = date_match.groups()
        if len(y) == 2: y = "20" + y
        return f"{int(d):02}/{int(m):02}/{y}"
    
    # 3. Numeric Normalization
    if re.match(r'^-?\d+\.?\d*$', v):
        try:
            f_val = float(v)
            return str(f_val).rstrip('0').rstrip('.') if '.' in str(f_val) else str(int(f_val))
        except: pass

    # 4. Keyword Canon & Domain Dictionary (OCR Spelling Fixes)
    v = re.sub(r'(\d+)\s*([\+\-])\s*(\d+)', r'\1\2\3', v)
    
    # Domain-specific spelling sanitization
    v = v.replace("manageriai", "managerial")
    v = v.replace("accountng", "accounting")
    
    if any(token in v for token in ["total", "gpa", "result"]):
        v = re.sub(r'[:\-\s]+', '', v)
        
    # 5. Whitespace Collapse & Entropy Polish
    v = re.sub(r'\s+', ' ', v).strip()
    if len(v) < 5 and not any(ord(c) > 127 for c in v):
        v = re.sub(r'[^a-zA-Z0-9]', '', v)
        
    return v

def process_file(file_path: str, llm: str, api_key: str, output_json: bool = False):
    """Processes file with Deep Feature Engineering for bit-perfect consistency"""
    if not os.path.exists(file_path):
        if not output_json:
            print(f"Error: File '{file_path}' not found.")
        return

    ext = os.path.splitext(file_path)[1].lower()
    images_to_process = []
    
    if ext == ".pdf":
        if not HAS_FITZ:
            raise ImportError("pymupdf package is not installed. Run `pip install pymupdf` to process PDFs.")
        doc = fitz.open(file_path)
        for i in range(len(doc)):
            page = doc.load_page(i)
            # Full Page
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
            img_full = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Zoom Header (Top 25%)
            zoom_rect = fitz.Rect(0, 0, page.rect.width, page.rect.height * 0.25)
            pix_zoom = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0), clip=zoom_rect) # Higher DPI for zoom
            img_zoom = Image.frombytes("RGB", [pix_zoom.width, pix_zoom.height], pix_zoom.samples)
            
            images_to_process.append(([img_full, img_zoom], f"page-{i+1}"))
    else:
        images_to_process.append((file_path, None))

    # SURGICAL SPEED UP: Parallel Threading (for multiple pages)
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_page = {}
        for img_input, suffix in images_to_process:
            if llm == "gemini":
                future = executor.submit(parse_with_gemini, img_input, api_key)
            else:
                future = executor.submit(parse_with_openai, img_input, api_key)
            future_to_page[future] = suffix

        for future in as_completed(future_to_page):
            suffix = future_to_page[future]
            try:
                json_result_str = future.result()
                results.append((json_result_str, suffix))
            except Exception as e:
                print(f"  Error on {suffix or 'file'}: {e}")

    for json_result_str, suffix in results:
        try:
            structured_data = GenericDocumentData.model_validate_json(json_result_str)
            
            # --- FEATURE ENGINEERING LAYER ---
            canonical_details = []
            
            # 1. Normalize and strip existing main_details with Artifact Pruning
            if structured_data.main_details:
                for item in structured_data.main_details:
                    k, val = normalize_key(item.key), normalize_value(item.value)
                    # Artifact Pruning
                    if not k or k == "institutionalslogan": continue
                    if not val and any(ord(c) > 127 for c in k): continue 
                    if len(k) <= 1 and not val: continue
                    canonical_details.append({"key": k, "value": val})

            # 2. Key-Value Extraction from Tables (Table-to-KV Smoothing)
            # Find rows that look like key-value pairs even if they are in a table
            remaining_tables = []
            if structured_data.tables:
                for table in structured_data.tables:
                    # GLOBAL SPARSITY FILTER: Remove columns that are empty across ALL rows
                    raw_rows = [[normalize_value(str(c)) for c in row] for row in table.rows]
                    if not raw_rows: continue
                    
                    # 1. Normalize Grid: Find max width and pad all rows
                    num_cols = max(len(r) for r in raw_rows)
                    normalized_rows = [r + [""] * (num_cols - len(r)) for r in raw_rows]
                    
                    # 2. Map Column Entropy
                    col_has_data = [False] * num_cols
                    for r in normalized_rows:
                        for idx, cell in enumerate(r):
                            if cell.strip(): col_has_data[idx] = True
                    
                    # 3. Apply Pruning with Density Filter
                    clean_table_rows = []
                    for r in normalized_rows:
                        # Prune columns that are empty globally
                        pruned_row = [r[idx] for idx, has_data in enumerate(col_has_data) if has_data]
                        # Count non-empty items for density check
                        dense_items = [c.strip() for c in pruned_row if c.strip()]
                        
                        # DENSITY FILTER: Valid data rows must have >= 2 items.
                        # This strips inconsistent labels like ["ii"] that have no associated data.
                        if len(dense_items) >= 2:
                            clean_table_rows.append(pruned_row)
                    
                    if clean_table_rows:
                        table_dict = table.model_dump()
                        table_dict["rows"] = clean_table_rows
                        if table_dict.get("table_name"): table_dict["table_name"] = normalize_key(table_dict["table_name"])
                        # Apply filter to headers (with safety bounds)
                        if table_dict.get("columns"):
                            headers = table_dict["columns"]
                            # Pad headers if shorter than col_has_data (using dicts to match model_dump)
                            if len(headers) < num_cols:
                                headers += [{"header": "", "sub_headers": None}] * (num_cols - len(headers))
                            
                            filtered_headers = []
                            for idx, has_data in enumerate(col_has_data):
                                if has_data and idx < len(headers):
                                    col = headers[idx]
                                    col["header"] = normalize_key(col["header"])
                                    filtered_headers.append(col)
                            
                            table_dict["columns"] = filtered_headers
                        remaining_tables.append(table_dict)

            # 3. Canonical Serialization
            output_data = {
                "issuing_authority": str(structured_data.issuing_authority).upper().strip(),
                "document_title": str(structured_data.document_title).upper().strip(),
                "main_details": canonical_details,
                "tables": remaining_tables
            }
            
            # Recursive Sorting for bit-perfect Merkle consistency
            final_output = recursive_sort(output_data)
            
            if output_json:
                sys.stdout.write(json.dumps(final_output, ensure_ascii=False, sort_keys=True) + '\n')
                sys.stdout.flush()
                return
            
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trilingual_json_files")
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_file = os.path.join(output_dir, f"{base_name}_rectified.json")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_output, f, indent=2, ensure_ascii=False)
                
            print(f"[{file_path}] Saved trilingual JSON to: {output_file}")
            
        except Exception as e:
            if not output_json:
                print(f"\n[{file_path}] Error during Trilingual Parsing: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Parse Trilingual Marksheet (English, Gujarati, Hindi) to JSON")
    parser.add_argument("input_path", help="Path to the Image, PDF, or folder")
    parser.add_argument("--llm", choices=["gemini", "openai"], default="gemini", help="Which LLM to use (default: gemini)")
    parser.add_argument("--json", action="store_true", help="Output JSON directly to stdout")
    args = parser.parse_args()

    # Get API Key
    if args.llm == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Please set the GEMINI_API_KEY environment variable.")
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please set the OPENAI_API_KEY environment variable.")

    # Identify files to process
    files_to_process = []
    if os.path.isdir(args.input_path):
        for f in os.listdir(args.input_path):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')):
                files_to_process.append(os.path.join(args.input_path, f))
        if not args.json:
            print(f"Found {len(files_to_process)} trilingual files in: {args.input_path}")
    else:
        files_to_process.append(args.input_path)

    for file_path in files_to_process:
        process_file(file_path, args.llm, api_key, args.json)

if __name__ == "__main__":
    main()
