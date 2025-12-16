import json
import os

# --- Configuration ---
INPUT_FILE = "csf-export.json"
OUTPUT_DIR = r"D:\sec_prog\Data\CSF-2.0"  # Your output path

def run_conversion():
    # 1. Setup
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] Could not find '{INPUT_FILE}'.")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 2. Load Data
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to read JSON: {e}")
        return

    # 3. Locate the "Elements" List
    # Based on your file: response -> elements -> elements (list)
    try:
        if "response" in raw_data:
            items = raw_data["response"]["elements"]["elements"]
        else:
            # Fallback if structure is slightly different
            items = raw_data["elements"]["elements"]
    except KeyError:
        print("[ERROR] JSON structure mismatch. Could not find 'elements' list.")
        return

    print(f"[INFO] Found {len(items)} items. Processing...")

    # 4. Build Lookup Dictionary (ID -> Title)
    # We need this to turn "GV.OC" into "Organizational Context"
    lookup = {}
    for item in items:
        eid = item.get("element_identifier")
        title = item.get("title")
        if eid:
            # If title is empty (common in some exports), use the text or ID
            if not title: 
                title = item.get("text", eid)
            lookup[eid] = title

    count = 0

    # 5. Generate Markdown Files
    for item in items:
        # Extract fields using YOUR specific keys
        eid = item.get("element_identifier", "")
        etype = item.get("element_type", "").lower()
        description = item.get("text", "")
        
        # We only want "Subcategories" (the actionable controls)
        # Logic: It's a subcategory if it has a hyphen (GV.OC-01) OR type is 'subcategory'
        is_subcategory = (etype == "subcategory") or ("-" in eid and "." in eid)

        if is_subcategory:
            # --- Parent Parsing Logic ---
            # Turn "GV.OC-01" into parents "GV" and "GV.OC"
            func_name = "Unknown Function"
            cat_name = "Unknown Category"
            
            try:
                # Standardize ID: GV.OC-01 -> GV.OC.01
                clean_id = eid.replace("-", ".") 
                parts = clean_id.split(".")
                
                if len(parts) >= 2:
                    func_id = parts[0]                # "GV"
                    cat_id = f"{parts[0]}.{parts[1]}" # "GV.OC"
                    
                    # Get human names from our lookup table
                    func_name = lookup.get(func_id, func_id)
                    cat_name = lookup.get(cat_id, cat_id)
            except:
                pass

            # --- Write Markdown ---
            md_content = f"""# {eid}
**Type:** NIST CSF Subcategory
**Function:** {func_name}
**Category:** {cat_name}

## Description
{description}

## Context
This control belongs to the **{func_name}** function, under the **{cat_name}** category.
"""
            # Safe Filename
            safe_filename = eid.replace("/", "_") + ".md"
            with open(os.path.join(OUTPUT_DIR, safe_filename), "w", encoding="utf-8") as f:
                f.write(md_content)
            
            count += 1

    print(f"[SUCCESS] Successfully created {count} markdown files in '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    run_conversion()