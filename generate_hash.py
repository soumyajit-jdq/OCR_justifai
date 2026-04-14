import json
import argparse
import sys

try:
    from Crypto.Hash import keccak
except ImportError:
    print("Error: pycryptodome is not installed. Run `pip install pycryptodome`")
    sys.exit(1)

def normalize_json_to_text(data):
    """
    Converts the structured OCR JSON into a deterministic Pipe-Separated Value (PSV) string.
    Order: Issuing Authority  Document Title  Main Details (Sorted) Table Rows (In Order)
    """
    parts = []
    
    # 1. Authority and Title
    parts.append(data.get("issuing_authority", "").strip())
    parts.append(data.get("document_title", "").strip())
    
    # 2. Main Details
    # We sort by 'key' to ensure the hash is identical even if the AI outputs keys in a different order.
    details = data.get("main_details", [])
    sorted_details = sorted(details, key=lambda x: x.get("key", ""))
    for item in sorted_details:
        key = item.get("key", "").strip()
        val = item.get("value", "").strip()
        # Combine key and value without separator for compactness, or with one for clarity.
        # We will use ':' to match original label visual.
        parts.append(f"{key}{val}")
    
    # 3. Tables
    # We process tables and rows in the exact order they appear in the JSON.
    for table in data.get("tables", []):
        for row in table.get("rows", []):
            for cell in row:
                # Add each cell value
                parts.append((cell or "").strip())
                
    # Join everything with a pipe separator
    return " ".join(parts)

def generate_keccak256(text):
    """Generates a Keccak-256 hash (used in Ethereum/Web3 standards)"""
    k = keccak.new(digest_bits=256)
    k.update(text.encode('utf-8'))
    return k.hexdigest()

def main():
    parser = argparse.ArgumentParser(description="Generate Keccak-256 Merkle Leaf from OCR JSON")
    parser.add_argument("json_path", help="Path to the parsed JSON file")
    args = parser.parse_args()
    
    try:
        with open(args.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        normalized_text = normalize_json_to_text(data)
        leaf_hash = generate_keccak256(normalized_text)
        
        print("\n" + "="*60)
        print("NORMALIZED TEXT (Target for Keccak-256)")
        print("="*60)
        print(normalized_text)
        
        print("\n" + "="*60)
        print("KECCAK-256 HASH RESULT")
        print("="*60)
        print(leaf_hash)
        print("="*60)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
