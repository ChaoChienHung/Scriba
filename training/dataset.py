import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class HandwritingDataset(Dataset):
    def __init__(self, dataframe, processor, max_target_length=128, cache_dir="../cache/dataset/"):
        self.data = dataframe
        self.processor = processor
        self.max_target_length = max_target_length

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        if cache_dir and os.path.exists(cache_dir):
            self.examples = torch.load(cache_dir)

        else:
            self.examples = []
            for _, row in dataframe.iterrows():
                image = Image.open(row['image_path']).convert("RGB")
                label = row['label']

                # Preprocess Image
                # ----------------
                pixel_values = processor(images=image, return_tensors="pt", cache_dir=cache_dir).pixel_values.squeeze()

                # Tokenize Label
                # --------------
                labels = processor.tokenizer(
                    label,
                    padding="max_length",
                    truncation=True,
                    max_length=max_target_length,
                    return_tensors="pt",
                    cache_dir=cache_dir
                ).input_ids.squeeze()

                self.examples.append({"pixel_values": pixel_values, "labels": labels})
            
            if cache_dir:
                torch.save(self.examples, cache_dir)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
