# Temporary file to add spk_id to the .json data files
import json
from tqdm import tqdm

# Load JSON data from a file
with open('/home/elisam/projects/def-ravanelm/elisam/speechbrain/recipes/EmotionalTTS/results/tacotron2/1234/save/valid.json', 'r') as json_file:
    json_data = json.load(json_file)

# Process the data with tqdm
for key, value in tqdm(json_data.items(), desc="Processing JSON Data"):
    # Add the "spk_id" key-value pair
    value["spk_id"] = "1"

# Save the modified JSON data back to a file
with open('/home/elisam/projects/def-ravanelm/elisam/speechbrain/recipes/EmotionalTTS/results/tacotron2/1234/save/valid2.json', 'w') as json_file:
    json.dump(json_data, json_file, indent=4)