import os
import json
import argparse
import time
import base64
from typing import List, Optional, Dict, Union
from pydantic import BaseModel, Field

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
    
    # Models
    models_to_try = [
        'gemini-3.1-flash-lite-preview',
        'gemini-flash-latest',
    ]

    last_exception = None

    if isinstance(image_input, str):
        img = Image.open(image_input)
    else:
        img = image_input

    prompt = (
        "Analyze the image and extract all structured details into JSON.\n\n"
        "TRILINGUAL SUPPORT: This document may contain text in English, Gujarati, and Hindi. "
        "You MUST extract all text EXACTLY as it appears in its original language. DO NOT translate, normalize, or convert script.\n\n"
        "1. DOCUMENT TITLE: Extract the main title of the document (e.g., 'Statement of Marks Bachelor of Science Semester-V') into the `document_title` field. "
        "DO NOT use this document title as the `table_name` for individual tables.\n"
        "2. MAIN DETAILS: For 'main_details', you MUST preserve the EXACT labels from the document as keys. "
        "This includes all punctuation (like colons ':'), spaces, and helper words in any of the three languages. "
        "DO NOT normalize, shorten, or clean up the labels; they should be exactly as they appear in the image.\n\n"
        "TABLE EXTRACTION INSTRUCTIONS:\n"
        "1. For tables with hierarchical headers (merged cells), use the `sub_headers` field in `TableColumn`. "
        "   For example, if 'Passing Marks' covers 3 columns ('Internal', 'External', 'Total'), the `header` should be 'Passing Marks' and `sub_headers` should be ['Internal', 'External', 'Total'].\n"
        "2. If a table does NOT have a clear title or specific column headers in the image, strictly omit the `table_name` or `columns` fields (set them to null).\n"
        "3. PRECISION ALIGNMENT: Each row must be a flat list of cell values corresponding EXACTLY to the visual leaf-headers sequence. "
        "   IMPORTANT: Pay meticulous attention to empty cells. If a row (e.g., the 'Total' row) has an empty space before the text label, you MUST provide an empty string for that column. NEVER shift values or skip empty columns. Every cell must align perfectly with its visual header.\n"
        "4. Capture every single row and column precisely. This data is used for Merkle tree hashing, so absolute structural accuracy is required.\n\n"
        "Return exactly matching the JSON schema."
    )

    for model_name in models_to_try:
        print(f"\nAttempting trilingual extraction with model: {model_name}...")
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[prompt, img],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=GenericDocumentData,
                        temperature=0.1,  # Low temperature for highly deterministic parsing
                    ),
                )
                return response.text
            except Exception as e:
                last_exception = e
                error_str = str(e)
                if "503" in error_str or "UNAVAILABLE" in error_str or "demand" in error_str.lower():
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt * 2 
                        print(f"[{model_name}] API is busy (503). Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        break
                elif "429" in error_str or "exhausted" in error_str.lower():
                    print(f"[{model_name}] Quota exhausted (429). Falling back...")
                    break
                else:
                    print(f"[{model_name}] Failed: {error_str[:150]}")
                    break
                    
    raise Exception(f"All Gemini models failed. Last error: {str(last_exception)}")


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
        "Analyze the image and extract all structured details into JSON.\n\n"
        "TRILINGUAL SUPPORT: This document may contain text in English, Gujarati, and Hindi. "
        "You MUST extract all text EXACTLY as it appears in its original language. DO NOT translate, normalize, or convert script.\n\n"
        "1. DOCUMENT TITLE: Extract the main title of the document (e.g., 'Statement of Marks Bachelor of Science Semester-V') into the `document_title` field. "
        "DO NOT use this document title as the `table_name` for individual tables.\n"
        "2. MAIN DETAILS: For 'main_details', you MUST preserve the EXACT labels from the document as keys. "
        "This includes all punctuation (like colons ':'), spaces, and helper words in any of the three languages. "
        "DO NOT normalize, shorten, or clean up the labels; they should be exactly as they appear in the image.\n\n"
        "TABLE EXTRACTION INSTRUCTIONS:\n"
        "1. For tables with hierarchical headers (merged cells), use the `sub_headers` field in `TableColumn`. "
        "   For example, if 'Passing Marks' covers 3 columns ('Internal', 'External', 'Total'), the `header` should be 'Passing Marks' and `sub_headers` should be ['Internal', 'External', 'Total'].\n"
        "2. If a table does NOT have a clear title or specific column headers in the image, omit the `table_name` or `columns` fields (set them to null).\n"
        "3. PRECISION ALIGNMENT: Each row must be a flat list of cell values corresponding EXACTLY to the visual leaf-headers sequence. "
        "   IMPORTANT: Pay meticulous attention to empty cells. If a row (e.g., the 'Total' row) has an empty space before the text label, you MUST provide an empty string for that column. NEVER shift values or skip empty columns. Every cell must align perfectly with its visual header.\n"
        "4. Capture every single row and column precisely. This data is used for Merkle tree hashing, so absolute structural accuracy is required.\n\n"
        "Return exactly matching the JSON schema."
    )
    
    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that parses trilingual documents (English, Gujarati, Hindi) into a structured JSON format."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }}
            ]}
        ],
        response_format=GenericDocumentData,
        temperature=0.1
    )
    return completion.choices[0].message.parsed.model_dump_json(indent=2)


def process_file(file_path: str, llm: str, api_key: str):
    """Processes a single image or PDF file"""
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    ext = os.path.splitext(file_path)[1].lower()
    images_to_process = []
    
    if ext == ".pdf":
        if not HAS_FITZ:
            raise ImportError("pymupdf package is not installed. Run `pip install pymupdf` to process PDFs.")
        print(f"\nProcessing trilingual PDF: {file_path}")
        doc = fitz.open(file_path)
        for i in range(len(doc)):
            page = doc.load_page(i)
            # Higher resolution matrix for scripts like Gujarati/Hindi
            pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5)) 
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images_to_process.append((img, f"page-{i+1}"))
    else:
        images_to_process.append((file_path, None))

    for img_input, suffix in images_to_process:
        try:
            print(f"\n[{file_path}] Trilingual extraction started" + (f" ({suffix})" if suffix else "") + "...")
            if llm == "gemini":
                json_result_str = parse_with_gemini(img_input, api_key)
            elif llm == "openai":
                json_result_str = parse_with_openai(img_input, api_key)
            
            structured_data = GenericDocumentData.model_validate_json(json_result_str)
            structured_data = structured_data.model_dump()
            
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trilingual_json_files")
            os.makedirs(output_dir, exist_ok=True)

            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_file_name = f"{base_name}_{suffix}_trilingual.json" if suffix else f"{base_name}_trilingual.json"
            output_file = os.path.join(output_dir, output_file_name)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(structured_data, f, indent=2, ensure_ascii=False) # Ensure non-ASCII characters (Guj/Hin) are preserved
                
            print(f"[{file_path}] Saved trilingual JSON to: {output_file}")
            
        except Exception as e:
            print(f"\n[{file_path}] Error during Trilingual Parsing: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Parse Trilingual Marksheet (English, Gujarati, Hindi) to JSON")
    parser.add_argument("input_path", help="Path to the Image, PDF, or folder")
    parser.add_argument("--llm", choices=["gemini", "openai"], default="gemini", help="Which LLM to use (default: gemini)")
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
        print(f"Found {len(files_to_process)} trilingual files in: {args.input_path}")
    else:
        files_to_process.append(args.input_path)

    for file_path in files_to_process:
        process_file(file_path, args.llm, api_key)

if __name__ == "__main__":
    main()
