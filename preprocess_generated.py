import csv
import os 

# ---- NSYNTH-GENERATED DATASET ----
# Dict for mapping instrument family to class index
instrument_to_class = {
    "bass": 0,
    "brass": 1,
    "flute": 2,
    "guitar": 3,
    "keyboard": 4,
    "mallet": 5,
    "organ": 6,
    "reed": 7,
    "string": 8,
    "synth": 9,
    "vocal": 10
}

nsynth_generated_audio_dir = '/data01/DATASET/nsynth/nsynth-generated/' # Path to the audio files
nsynth_generated_csv_path = './data/nsynth-generated/examples.csv' # Path to the new csv file

files = [f for f in os.listdir(nsynth_generated_audio_dir) if os.path.isfile(os.path.join(nsynth_generated_audio_dir, f))]

with open(nsynth_generated_csv_path, "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["filename", "instrument_family"])

    for file in files:
        filename = file
        instrument_family = filename.split("_")[0]
        target_class = instrument_to_class.get(instrument_family)
        writer.writerow([filename, target_class])