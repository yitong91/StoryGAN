'''
Pre-processing Step 1.1:
Input: JSON file
Output: Pytorch (.pt) file
Generates a pytorch file containing a list of the text tokens.

'''
import json
import os.path
from os import path
import clip
import torch

JSON_FILE = '/home/ubuntu/vist/sis/train.story-in-sequence.json'
TOKENS_FILE = 'train_text_tokens_10_10.pt' 

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
raw_texts = raw_texts[180693:]
text_inputs = torch.cat([clip.tokenize(text) for text in raw_texts]).to(device)
print("Tokens size:", text_inputs.size())

print("Saving to torch file %s" % TOKENS_FILE)
torch.save( text_inputs, open( TOKENS_FILE, "wb" ) )
  
