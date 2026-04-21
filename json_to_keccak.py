import json
import sys
import os
from web3 import Web3
from collections import OrderedDict


def generate_keccak256(text: str):
    """Generates an Ethereum-standard Keccak-256 hash using Web3.py."""
    hash_bytes = Web3.keccak(text=text)
    return Web3.to_hex(hash_bytes)


def build_canonical_payload(data: dict) -> str:
    """
    Builds a canonical JSON string with STRICT key ordering:
      registration_no -> name -> gpa -> subjects
    Each subject maintains: code -> title -> credits -> grade
    
    Uses separators=(',', ':') to produce compact JSON 
    identical to JavaScript's JSON.stringify().
    """
    # Build subjects list preserving key order
    subjects = []
    for s in data.get("subjects", []):
        ordered_subject = OrderedDict([
            ("code", str(s.get("code", ""))),
            ("title", str(s.get("title", ""))),
            ("credit_points", str(s.get("credit_points", ""))),
            # ("grade", str(s.get("grade", "")))
        ])
        subjects.append(ordered_subject)

    # Build top-level payload with strict key order
    payload = OrderedDict([
        ("registration_no", str(data.get("registration_no", ""))),
        ("name", str(data.get("name", ""))),
        ("gpa", str(data.get("gpa", ""))),
        ("subjects", subjects)
    ])

    # Compact JSON: no spaces, matches JavaScript's JSON.stringify()
    return json.dumps(payload, separators=(',', ':'))


def run_hashing(json_file_path):
    if not os.path.exists(json_file_path):
        print(f"Error: File not found at {json_file_path}")
        return

    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle both list of students or a single student object
    if isinstance(data, list):
        for student in data:
            canonical_json = build_canonical_payload(student)
            row_hash = generate_keccak256(canonical_json)
            print(f"Reg No: {student.get('registration_no')} | Hash: {row_hash}")
            print(f"  JSON: {canonical_json[:120]}...")
    else:
        canonical_json = build_canonical_payload(data)
        row_hash = generate_keccak256(canonical_json)
        print(row_hash)
        print(f"JSON: {canonical_json[:120]}...")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python json_to_keccak.py <path_to_json_file>")
    else:
        run_hashing(sys.argv[1])
