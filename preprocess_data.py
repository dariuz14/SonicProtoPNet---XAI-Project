import json
import csv

# Convert json files to csv files
json_path = '/data01/DATASET/nsynth/nsynth-valid/examples.json' # Path to the old json file
csv_path = './data/nsynth-generated/examples.csv' # Path to the new csv file

"""
with open(json_path, "r") as f:
    data = json.load(f)

with open(csv_path, "w", newline="") as f:
    writer = None
    for filename, metadata in data.items():
        # Added extension because they are not in json 
        row = {"filename": filename + ".wav", **metadata}

        if writer is None:
            other_cols = [k for k in row.keys() if k not in ("filename", "instrument_family")]
            fieldnames = ["filename", "instrument_family"] + other_cols
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

        writer.writerow(row)
"""

# Modify csv to remove synth_lead class and adjust class indices
rows = []
with open(csv_path, 'r') as f:
    reader = csv.reader(f)
    
    header = next(reader, None)
    if header:
        rows.append(header)
    
    for row in reader:
        if len(row) < 2:
            continue
            
        filename = row[0]
        instrument_family = row[1]
        
        if filename.startswith('synth_lead'):
            continue
        
        if instrument_family == '10':
            row[1] = '9'
        
        rows.append(row)

with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print("OK")

