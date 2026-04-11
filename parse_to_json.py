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

# Define the Target JSON Schema using Pydantic
class KeyValueField(BaseModel):
    key: str = Field(description="The name of the field (e.g., 'Student Name', 'Date of Issue', 'CGPA')")
    value: str = Field(description="The extracted value for this field")

class TableRow(BaseModel):
    cells: List[KeyValueField] = Field(description="The cells of the row. Use the column name as the 'key' and the cell contents as the 'value'.")

class DataTable(BaseModel):
    table_name: str = Field(description="A descriptive name for the table (e.g., 'Semester Marks', 'Grades')")
    columns: List[str] = Field(description="The column headers of the table")
    rows: List[TableRow] = Field(description="The rows of data.")

class GenericDocumentData(BaseModel):
    document_type: str = Field(description="The detected type of the document (e.g., 'Marksheet', 'Degree', 'Certificate')")
    issuing_authority: str = Field(description="The organization that issued the document (e.g., University Name, Board Name)")
    main_details: List[KeyValueField] = Field(description="General key-value information extracted from the document")
    tables: List[DataTable] = Field(description="Any structured or tabular data found in the document (like lists of subjects, marks, or any other tables)")


# Parsing Functions using AI Models
def parse_with_gemini(image_path: str, api_key: str, max_retries: int = 4) -> str:
    """Uses Google's Gemini API with native Vision capabilities and Structured Outputs"""
    if not HAS_GEMINI:
        raise ImportError("google-genai package is not installed. Run `pip install google-genai`")
    if not HAS_PIL:
        raise ImportError("pillow package is not installed. Run `pip install pillow`")
    
    print("\nSending image directly to Gemini API for processing...")
    client = genai.Client(api_key=api_key)
    
    # Models
    models_to_try = [
        'gemini-2.5-flash', 
        'gemini-2.5-pro', 
        'gemini-2.0-flash', 
        'gemini-flash-latest',
        'gemini-3.1-flash-lite-preview',
        # 'gemini-1.5-flash',
        # 'gemini-2.0-flash-lite',
    ]

    last_exception = None

    img = Image.open(image_path)
    prompt = "Analyze the image and extract all structured details into JSON. Extract every single field, including Enrollment No, Seat No, and the intricate details in the table (Marks, CR, GR, GP, EGP, Remarks). Return exactly matching the JSON schema."

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
                        temperature=0.1,  # Low temperature for highly deterministic parsing
                    ),
                )
                return response.text
            except Exception as e:
                last_exception = e
                error_str = str(e)
                
                # If we get a temporary busy error, we can wait and retry it
                if "503" in error_str or "UNAVAILABLE" in error_str or "demand" in error_str.lower():
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt * 2 # 2s, 4s, 8s
                        print(f"[{model_name}] API is busy (503). Retrying in {wait_time} seconds (Attempt {attempt+1}/{max_retries})...")
                        time.sleep(wait_time)
                    else:
                        print(f"[{model_name}] Max retries reached for this model.")
                        break # Try the next model
                
                # If we get a 429 rate limit quota error, retrying won't help right now, so immediately fallback to a different model
                elif "429" in error_str or "exhausted" in error_str.lower():
                    print(f"[{model_name}] Quota exhausted (429). Immediately falling back to different model...")
                    break # Try the next model
                    
                # If we get any other unexpected bug (like 404 Not Found), don't crash! Just fallback.
                else:
                    print(f"[{model_name}] Failed with error. Reason: {error_str[:150]}. Trying next model...")
                    break # Try the next model
                    
    # If all models failed, raise the last exception
    raise Exception(f"All Gemini models failed. Last error: {str(last_exception)}")


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def parse_with_openai(image_path: str, api_key: str) -> str:
    """Uses OpenAI's API with native Vision capabilities and Structured Outputs"""
    if not HAS_OPENAI:
        raise ImportError("openai package is not installed. Run `pip install openai`")
    
    print("\nSending image directly to OpenAI API for processing...")
    client = OpenAI(api_key=api_key)
    base64_image = encode_image(image_path)
    prompt = "Analyze the image and extract all structured details into JSON. Extract every single field, including Enrollment No, Seat No, and the intricate details in the table (Marks, CR, GR, GP, EGP, Remarks)."
    
    completion = client.beta.chat.completions.parse(
        model="gpt-4o", # 'gpt-4o' supports image inputs natively
        messages=[
            {"role": "system", "content": "You are a helpful assistant that parses document images into a structured JSON format containing generic key-value fields and tabular data."},
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
    
    # Returns the JSON serialized payload perfectly matching GenericDocumentData
    return completion.choices[0].message.parsed.model_dump_json(indent=2)


# Main Workflow
def main():
    parser = argparse.ArgumentParser(description="Parse Marksheet image to JSON using AI Native Vision")
    parser.add_argument("image_path", help="Path to the Marksheet Image")
    parser.add_argument("--llm", choices=["gemini", "openai"], default="gemini", help="Which LLM to use (default: gemini)")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: File '{args.image_path}' not found.")
        return

    # Skip local OCR entirely since LLM Vision is vastly superior and perfectly preserves layout/tables.
    print(f"Input Image verified: {args.image_path}")

    # Parse using AI Multi-modality Native Vision
    try:
        if args.llm == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Please set the GEMINI_API_KEY environment variable. Example: set GEMINI_API_KEY=your_key")
            
            json_result_str = parse_with_gemini(args.image_path, api_key)
            
        elif args.llm == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Please set the OPENAI_API_KEY environment variable. Example: set OPENAI_API_KEY=your_key")
                
            json_result_str = parse_with_openai(args.image_path, api_key)
            
        # Parse text into a dict for pretty printing
        structured_data = json.loads(json_result_str)
        
        print("\n" + "="*60)
        print("FINAL STRUCTURED JSON RESULT")
        print("="*60)
        print(json.dumps(structured_data, indent=2))
        
        # Ensure json_files directory exists
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "json_files")
        os.makedirs(output_dir, exist_ok=True)

        # Save to output file
        base_name = os.path.splitext(os.path.basename(args.image_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}_parsed.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, indent=2)
            
        print(f"\nSuccessfully saved parsed data to {output_file}")
        
    except Exception as e:
        print(f"\nError during AI Parsing: {str(e)}")

if __name__ == "__main__":
    main()
