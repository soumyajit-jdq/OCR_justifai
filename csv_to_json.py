import csv
import json
import os
import argparse

def convert_horizontal_csv_to_individual_json(csv_path, output_dir):
    """
    Converts horizontal CSV to individual JSON files (1 by 1).
    Matches the exact format provided by the user.
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    processed_count = 0
    with open(csv_path, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            reg_no = row.get("Registration_No", row.get("registration_no", ""))
            name = row.get("Student_Name", row.get("name", ""))
            gpa = row.get("GPA", row.get("gpa", ""))
            
            if not reg_no: continue
            
            # Match the exact format: NO hash, NO extra keys
            student_obj = {
                "registration_no": reg_no,
                "name": name,
                "gpa": gpa,
                "subjects": []
            }
            
            # Extract subjects
            for i in range(1, 20):
                code_key = f"Course_{i}_Code"
                name_key = f"Course_{i}_Name"
                credit_points_key = f"Course_{i}_Credit_Points"
                # grade_key = f"Course_{i}_Grade_Points"
                
                if code_key in row and row[code_key].strip():
                    student_obj["subjects"].append({
                        "code": row[code_key].strip(),
                        "title": row.get(name_key, "").strip(),
                        "credit_points": row.get(credit_points_key, "").strip(),
                        # "grade": row.get(grade_key, "").strip()
                    })
            
            # Save 1 by 1
            file_name = f"{reg_no}.json"
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, mode='w', encoding='utf-8') as out_f:
                json.dump(student_obj, out_f, indent=2)
            
            processed_count += 1
            
    print(f"Successfully generated {processed_count} JSON files in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate individual JSON files from CSV.")
    parser.add_argument("input", help="Path to input CSV file")
    parser.add_argument("output_dir", nargs="?", help="Directory to save JSON files", default=r"D:\blockchain\practice\JadeQuest\ocr\OCR_justifai\csv_to_json")
    
    args = parser.parse_args()
    
    convert_horizontal_csv_to_individual_json(args.input, args.output_dir)
