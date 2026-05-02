from nltk.translate.bleu_score import sentence_bleu
from torch.nn.functional import cosine_similarity
from torchvision.io import read_image
import getpass
import torch
import os
import re

word_exp = re.compile(r'[a-z]+')

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

def embed(images, model, processor, device, processor_args={}):
    model.eval()
    with torch.no_grad():
        input_features = processor(images, return_tensors="pt", **processor_args).to(device)
        output_features = model(**input_features)
        return output_features.last_hidden_state

def get_collator(image_model, image_processor, eos_token, device, path, first_only=True):
    def collate(batch):
        captions = []
        images = []
        for row in batch:
            if first_only:
                captions.append(row['caption'][0] + eos_token)
            else:
                captions.append([
                    caption + eos_token for caption in row['caption']
                ])
            images.append(read_image(path + row['image']))
        return embed(images, image_model, image_processor, device), captions
    return collate

def get_score(references, generation):
    generation = word_exp.findall(generation.lower())
    references = [
        word_exp.findall(reference.lower()) for reference in references
    ]
    return sentence_bleu(references, generation)