import os
import json
import argparse
import time
import base64
from typing import List, Optional, Dict
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

# =====================================================================
# 1. Define the Target JSON Schema using Pydantic
# =====================================================================

class KeyValueField(BaseModel):
    key: str = Field(description="The name of the field (e.g., 'Student Name', 'Date of Issue', 'CGPA')")
    value: str = Field(description="The extracted value for this field. Can be in English or Gujarati.")

class TableRow(BaseModel):
    cells: List[KeyValueField] = Field(description="The cells of the row. Use the column name as the 'key' and the cell contents as the 'value'. Values can be English or Gujarati.")

class DataTable(BaseModel):
    table_name: str = Field(description="A descriptive name for the table (e.g., 'Semester Marks', 'Grades')")
    columns: List[str] = Field(description="The column headers of the table")
    rows: List[TableRow] = Field(description="The rows of data.")

class GenericDocumentData(BaseModel):
    document_type: str = Field(description="The detected type of the document (e.g., 'Marksheet', 'Degree', 'Certificate')")
    issuing_authority: str = Field(description="The organization that issued the document (e.g., University Name, Board Name)")
    main_details: List[KeyValueField] = Field(description="General key-value information extracted from the document. Supports English and Gujarati values.")
    tables: List[DataTable] = Field(description="Any structured or tabular data found in the document. Supports English and Gujarati content.")


# =====================================================================
# 2. Parsing Functions using AI Models
# =====================================================================

def parse_with_gemini(image_path: str, api_key: str, max_retries: int = 4) -> str:
    """Uses Google's Gemini API with native Vision capabilities and Structured Outputs (Multilingual)"""
    if not HAS_GEMINI:
        raise ImportError("google-genai package is not installed. Run `pip install google-genai`")
    if not HAS_PIL:
        raise ImportError("pillow package is not installed. Run `pip install pillow`")
    
    print("\nSending image directly to Gemini API for Multilingual processing (English + Gujarati)...")
    client = genai.Client(api_key=api_key)
    # Using the most capable and available models
    # models_to_try = ['gemini-2.0-flash', 'gemini-2.5-flash', 'gemini-flash-latest']
    models_to_try = [
        'gemini-2.5-flash', 
        'gemini-2.5-pro', 
        'gemini-2.0-flash', 
        'gemini-flash-latest',
        'gemini-3.1-flash-image-preview',
        'gemini-3.1-flash-lite-preview',
        'gemini-2.0-flash-lite',
    ]

    last_exception = None

    img = Image.open(image_path)
    prompt = (
        "Analyze the provided image and extract all structured details into JSON. "
        "The document may contain text in both English and Gujarati. "
        "Extract every single field exactly as it appears, preserving the original language (Gujarati or English). "
        "Include Enrollment No, Seat No, and all intricate details in the table (Marks, CR, GR, GP, EGP, Remarks). "
        "Return the data exactly matching the provided JSON schema."
    )

    for model_name in models_to_try:
        print(f"\nAttempting extraction with model: {model_name}...")
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[prompt, img],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=GenericDocumentData,
                        temperature=0.1,
                    ),
                )
                return response.text
            except Exception as e:
                last_exception = e
                error_str = str(e)
                
                if "503" in error_str or "UNAVAILABLE" in error_str or "demand" in error_str.lower():
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt * 2
                        print(f"[{model_name}] API is busy (503). Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        break
                elif "429" in error_str or "exhausted" in error_str.lower():
                    print(f"[{model_name}] Quota exhausted (429). Falling back...")
                    break
                else:
                    print(f"[{model_name}] Error: {error_str[:150]}. Falling back...")
                    break
                    
    raise Exception(f"All Gemini models failed. Last error: {str(last_exception)}")


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def parse_with_openai(image_path: str, api_key: str) -> str:
    """Uses OpenAI's API with native Vision capabilities and Structured Outputs (Multilingual)"""
    if not HAS_OPENAI:
        raise ImportError("openai package is not installed. Run `pip install openai`")
    
    print("\nSending image directly to OpenAI API for Multilingual processing (English + Gujarati)...")
    client = OpenAI(api_key=api_key)
    base64_image = encode_image(image_path)
    prompt = (
        "Analyze the image and extract all structured details into JSON. "
        "The document may contain text in both English and Gujarati. "
        "Extract every single field exactly as it appears, preserving the original language (Gujarati or English)."
    )
    
    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that parses document images into a structured JSON format. You support multiple languages including English and Gujarati."},
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


# =====================================================================
# 3. Main Workflow
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Multilingual (Eng+Guj) Parser using AI Native Vision")
    parser.add_argument("image_path", help="Path to the Marksheet Image")
    parser.add_argument("--llm", choices=["gemini", "openai"], default="gemini", help="Which LLM to use (default: gemini)")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: File '{args.image_path}' not found.")
        return

    print(f"Analyzing Multilingual Image: {args.image_path}")

    # 2. Parse using AI Multi-modality Native Vision
    try:
        if args.llm == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Please set the GEMINI_API_KEY environment variable.")
            
            json_result_str = parse_with_gemini(args.image_path, api_key)
            
        elif args.llm == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Please set the OPENAI_API_KEY environment variable.")
                
            json_result_str = parse_with_openai(args.image_path, api_key)
            
        structured_data = json.loads(json_result_str)
        
        print("\n" + "="*60)
        print("MULTILINGUAL STRUCTURED JSON RESULT (Eng + Guj)")
        print("="*60)
        print(json.dumps(structured_data, indent=2, ensure_ascii=False))
        
        # Ensure json_files directory exists
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "json_files")
        os.makedirs(output_dir, exist_ok=True)

        # Save to output file
        base_name = os.path.splitext(os.path.basename(args.image_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}_multilingual_parsed.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, indent=2, ensure_ascii=False)
            
        print(f"\nSuccessfully saved multilingual parsed data to {output_file}")
        
    except Exception as e:
        print(f"\nError during Multilingual Parsing: {str(e)}")

if __name__ == "__main__":
    main()
