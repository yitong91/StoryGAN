'''
Pre-processing Step 1.2:
Input: Pytorch (.pt) file
Output: Pytorch (.pt) file
Generates a pytorch file containing a list of the text caption CLIP encodings.

'''
import os.path
from os import path
import clip
import torch

TOKENS_FILE = 'train_text_tokens_10_10.pt' 
ENCODINGS_FILE = 'train_text_encodings_10_10.pt' # a list of the text encodings in the same order as they appear in the file


# Load the CLIP text encoder model
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print("Device: %s" % device)
print("Loading CLIP model...")
model, preprocess = clip.load('ViT-B/32', device)

print("Clearing cuda cache...")
torch.cuda.empty_cache()

# Prepare the inputs
print("Loading tokenized text...")
text_inputs = torch.load(TOKENS_FILE, map_location=torch.device('cpu'))

# Calculate features
print("Encoding tokenized texts...")
with torch.no_grad():
  text_features = model.encode_text(text_inputs)

print(text_features.size())

print("Saving to torch file %s" % ENCODINGS_FILE)
torch.save( text_features, open( ENCODINGS_FILE, "wb" ) )
  
