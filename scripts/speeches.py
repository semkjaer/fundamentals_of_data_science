import pandas as pd
from pathlib import Path

# Parse UN speeches dataset
data_dir = Path("data/Speeches/UNGDC_1946-2025/TXT")

speeches = []

# Recursively search for all .txt files inside all session subfolders
for file_path in data_dir.rglob("*.txt"):
    
    # 1. Skip hidden macOS metadata files (starting with ._)
    if file_path.name.startswith("._"):
        continue
        
    try:
        # 2. Extract full text from the file
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        
        # 3. Parse filename (e.g., "ARG_01_1946.txt" -> "ARG_01_1946")
        filename_clean = file_path.stem
        parts = filename_clean.split("_")
        
        # Ensure filename follows the standard ISO_Session_Year structure
        if len(parts) >= 3:
            country_code = parts[0]
            session = parts[1]
            year = parts[2]
            
            speeches.append({
                "country_code": country_code,
                "year": int(year),
                "text": text,
                "filename": file_path.name
            })
            
    except Exception as e:
        print(f"Could not read file {file_path}: {e}")

# Convert list of dictionaries into a Pandas DataFrame
df = pd.DataFrame(speeches)

# Sort chronologically by year and country for easy navigation
df = df.sort_values(by=["year", "country_code"]).reset_index(drop=True)

print(f"Successfully loaded {len(df)} speeches into the DataFrame!\n")
# Save to parquet inside your data folder
df.to_parquet("data/un_speeches.parquet", index=False)