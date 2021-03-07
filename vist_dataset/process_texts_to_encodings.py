'''
NOTE: This only works for test and val because they are small. This does NOT work on train.
Use the scripts process_texts_to_tokens.py followed by process_tokens_to_encodings.py 
for the same effect.

Pre-processing Step 1:
Input: JSON file
Output: Pytorch (.pt) file
Generates a pytorch file containing a list of the text caption CLIP encodings.
This is generated as a preprocessing step because it is more efficient for 
the CLIP model to encode in large batches rather than 1 string at a time.


'''
import json
import os.path
from os import path
import clip
import torch

JSON_FILE = '/home/ubuntu/vist/sis/train.story-in-sequence.json'
ENCODINGS_FILE = 'train_text_encodings_0.pt' # a list of the text encodings in the same order as they appear in the file

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: %s" % device)

# Load the json annotation file
print("Loading json file...")
with open(JSON_FILE) as f:
  data = json.load(f)

annotations = data['annotations']
raw_texts = []
for annot in annotations:
  # Extract relevant fields
  annot_dict = annot[0]
  raw_texts.append(annot_dict['text'])


# Load the CLIP text encoder model
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: %s" % device)
print("Loading CLIP model...")
model, preprocess = clip.load('ViT-B/32', device)

print("Clearing json object...")
data.clear()

print("Clearing cuda cache...")
torch.cuda.empty_cache()

# Prepare the inputs
print("Tokenizing raw texts...")
text_inputs = torch.cat([clip.tokenize(text) for text in raw_texts]).to(device)

# Calculate features
print("Encoding tokenized texts...")
with torch.no_grad():
  text_features = model.encode_text(text_inputs)

print(text_features.size())

print("Saving to torch file %s" % ENCODINGS_FILE)
torch.save( text_features, open( ENCODINGS_FILE, "wb" ) )
  
