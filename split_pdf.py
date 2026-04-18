import fitz
import os
import sys
import glob
import re

def sanitize_filename(filename):
    """Deep cleaning for Windows path safety: removes trailing spaces and illegal chars."""
    # Strip leading/trailing whitespace which breaks Windows folder access
    clean = filename.strip()
    # Replace common illegal characters
    clean = re.sub(r'[<>:"/\\|?*]', '_', clean)
    # Ensure it doesn't end with a dot or space
    clean = clean.rstrip(". ")
    return clean if clean else "unnamed_document"

def clean_page(page):
    """Searches for watermarks/footers and covers them with white redaction blocks."""
    # List of common scanner footer/watermark phrases
    REDACT_PHRASES = [
        "scanned by", "camscanner", "watermark", 
        "tapscanner", "scanner", "adobe scan",
        "shot on", "powered by"
    ]
    
    found_any = False
    for phrase in REDACT_PHRASES:
        # Search for text (case-insensitive)
        hit_rects = page.search_for(phrase)
        for rect in hit_rects:
            # Add a white box over the area
            page.add_redact_annot(rect, fill=(1, 1, 1))
            found_any = True
            
    if found_any:
        # Apply redactions to bake the white blocks into the PDF
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
    
    # Remove all existing annotations (sticky notes, links, etc.) 
    # as these often contain watermark objects
    for annot in page.annots():
        page.delete_annot(annot)

def split_pdf(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    # Open the source PDF
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        print(f"Error opening {file_path}: {e}")
        return

    # Sanitize the base name for Windows safety
    raw_base_name = os.path.splitext(os.path.basename(file_path))[0]
    base_name = sanitize_filename(raw_base_name)
    
    # Target output directory
    output_dir = r"C:\Users\sanso\Downloads\Zip\split file"
    
    # Create a subfolder for each PDF
    pdf_output_dir = os.path.join(output_dir, base_name)
    os.makedirs(pdf_output_dir, exist_ok=True)

    print(f"\n>>> Processing: '{file_path}' ({len(doc)} pages)")
    
    for i in range(len(doc)):
        # Create a new empty PDF for the single page
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=i, to_page=i)
        
        # Clean the page (Redaction/Metadata removal)
        page = new_doc[0]
        clean_page(page)
        
        # Save with zero-padded page number
        output_path = os.path.join(pdf_output_dir, f"{base_name}_page_{i+1:04d}.pdf")
        
        try:
            new_doc.save(output_path, garbage=3, deflate=True)
            new_doc.close()
        except Exception as e:
            print(f"    [!] Error saving page {i+1}: {e}")
            new_doc.close()

    print(f"    [OK] Split and Cleaned {len(doc)} files in: {pdf_output_dir}")
    doc.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python split_pdf.py <folder_path>")
        print("  python split_pdf.py <file1.pdf> <file2.pdf> ...")
    else:
        target = sys.argv[1]
        
        if os.path.isdir(target):
            print(f"Scanning folder: {target}")
            pdf_files = glob.glob(os.path.join(target, "*.pdf")) + glob.glob(os.path.join(target, "*.PDF"))
            pdf_files.sort()
            
            print(f"Found {len(pdf_files)} PDF files.\n")
            for pdf_file in pdf_files:
                split_pdf(pdf_file)
        else:
            for file_arg in sys.argv[1:]:
                clean_path = file_arg.strip().strip('"').strip("'")
                split_pdf(clean_path)
        
        print("\nAll tasks completed!")
