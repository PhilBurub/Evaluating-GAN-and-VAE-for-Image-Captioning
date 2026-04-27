from torchvision.io import read_image
import getpass
import torch
import os

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

def embed(images, model, processor, device, processor_args={}):
    model.eval()
    with torch.no_grad():
        input_features = processor(images, return_tensors="pt", **processor_args).to(device)
        output_features = model(**input_features)
        return output_features.last_hidden_state

def get_collator(image_model, image_processor, eos_token, device, path):
    def collate(batch):
        captions = []
        images = []
        for row in batch:
            captions.append(row['caption'][0] + eos_token)
            images.append(read_image(path + row['image']))
        return embed(images, image_model, image_processor, device), captions
    return collate
