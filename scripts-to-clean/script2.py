import json
import os

# --- Configuration ---
INPUT_FILE = "urlhaus_full.json"
OUTPUT_DIR = r"D:\sec_prog\Data\urls" 
MAX_RECORDS = 5000  # Stop after saving this many active threats

def run_conversion():
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] Could not find '{INPUT_FILE}'.")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Loading {INPUT_FILE} (Memory intensive step)...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to read JSON: {e}")
        return

    # 1. SORT BY NEWEST (IDs are keys)
    # We convert keys to integers to sort correctly, then reverse to get newest first
    print("[INFO] Sorting records to process newest threats first...")
    try:
        all_ids = sorted([k for k in data.keys() if k.isdigit()], key=int, reverse=True)
    except Exception:
        # Fallback if keys aren't simple IDs
        all_ids = list(data.keys())

    print(f"[INFO] Total records in file: {len(all_ids)}")
    print(f"[INFO] Filtering for 'online' threats only (Limit: {MAX_RECORDS})...")

    saved_count = 0
    skipped_offline = 0

    for threat_id in all_ids:
        # Stop if we hit our limit
        if saved_count >= MAX_RECORDS:
            print(f"\n[STOP] Reached limit of {MAX_RECORDS} active records.")
            break

        # Get the record info
        record_list = data[str(threat_id)]
        if not record_list or not isinstance(record_list, list):
            continue
            
        item = record_list[0]
        
        # --- THE CRITICAL FILTER ---
        # Only process URLs that are currently ONLINE
        if item.get("url_status") != "online":
            skipped_offline += 1
            continue

        # Extract fields
        url = item.get("url", "Unknown URL")
        threat_type = item.get("threat", "unknown")
        tags = item.get("tags", [])
        reporter = item.get("reporter", "anonymous")
        date_added = item.get("dateadded", "")

        # Build Markdown content
        md_content = f"""# Threat Intel: {threat_type.upper()} ({threat_id})
**Source:** URLHaus
**Status:** {item.get('url_status')}
**Date:** {date_added}
**Reporter:** {reporter}

## Indicators
- **Malicious URL:** `{url}`
- **Threat Type:** {threat_type}
- **Tags:** {', '.join(tags) if tags else 'None'}

## Analysis
Confirmed active malicious URL hosting **{threat_type}**.
Systems connecting to this URL should be quarantined immediately.
"""
        
        # Save file
        filename = f"urlhaus_{threat_id}.md"
        with open(os.path.join(OUTPUT_DIR, filename), "w", encoding="utf-8") as f:
            f.write(md_content)
        
        saved_count += 1
        
        # Progress indicator
        if saved_count % 100 == 0:
            print(f"Saved {saved_count} active threats... (Skipped {skipped_offline} offline)", end='\r')

    print(f"\n\n[SUCCESS] Pipeline Complete.")
    print(f"- Processed: {saved_count + skipped_offline}")
    print(f"- Saved (Online): {saved_count}")
    print(f"- Skipped (Offline): {skipped_offline}")
    print(f"- Output Folder: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_conversion()