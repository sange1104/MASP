import os
import random
import json
from PIL import Image
from torch.utils.data import Dataset
import yaml
from collections import defaultdict
from tqdm import tqdm
from prompt_utils import create_messages, create_prefix_messages
from qwen_vl_utils import process_vision_info

def load_image(image_path):
    """
    Load an image from the given file path and convert it to RGB.
    """
    image = Image.open(image_path).convert("RGB")
    return image

def resize_with_aspect_ratio(img, max_len=1024):
    """
    Resize the image while maintaining aspect ratio so that the longest side is max_len.
    """
    w, h = img.size
    if max(w, h) <= max_len:
        return img
    if w >= h:
        new_w = max_len
        new_h = int(h * max_len / w)
    else:
        new_h = max_len
        new_w = int(w * max_len / h)
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

def load_prompts_from_txt(filepath):
    """
    Load prompt templates from a txt file.
    Each line in the file is considered a prompt template.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts

class EmotionDataset(Dataset):
    def __init__(self, image_root, json_dir=None, dataname=None, prompt_txt_path="prompts.txt", mode="train"):
        """
        Custom dataset for emotion classification.
        Loads images, annotations, and prompts for each sample.
        """
        self.samples = [] 
        if dataname.lower() == 'emotion6': 
            ctg_list = ['sadness', 'fear', 'surprise', 'anger', 'joy', 'disgust']
        elif dataname.lower() in ['abstract', 'artphoto']:
            ctg_list = ['sad', 'amusement', 'fear', 'disgust', 'excitement', 'awe', 'content', 'anger']
        elif dataname.lower() in ['flickr', 'instagram']:
            ctg_list = ['positive', 'negative']
        else:
            ctg_list = ['sadness', 'amusement', 'fear', 'disgust', 'excitement', 'awe', 'contentment', 'anger']
        
        self.prompts = load_prompts_from_txt(prompt_txt_path) 
        self.prompts = [prompt.format(random.sample(ctg_list, len(ctg_list))) for prompt in self.prompts]
        
        for emo in os.listdir(image_root):
            emo_dir = os.path.join(image_root, emo)
            if mode == "train":
                ctg_list = random.sample(ctg_list, len(ctg_list))
                
            for img_file in os.listdir(emo_dir):     
                if mode == "train":
                    prompt = random.choice(self.prompts).format(', '.join(ctg_list))
                else:
                    prompt = self.prompts[0].format(', '.join(ctg_list))
                    
                if dataname.lower() == 'emoset':
                    image_id = img_file.split('.')[0]
                    image_path = os.path.join(emo_dir, img_file)
                    json_path = os.path.join(json_dir, emo, image_id + '.json')
                    with open(json_path, "r") as f:
                        ann = json.load(f) 
                        self.samples.append({
                            "image": image_path, 
                            "text": prompt,
                            "task": 'emotion',
                            "label": ann.get('emotion', None) 
                        }) 
                else:
                    image_path = os.path.join(emo_dir, img_file)
                    self.samples.append({
                            "image": image_path, 
                            "text": prompt,
                            "task": 'emotion',
                            "label": emo
                        }) 
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx] 
        return sample  

def collate_fn(examples, processor):
    """
    Collate function for DataLoader.
    Processes images and text, applies processor, and prepares batch for training.
    """
    for example in examples:
        for msg in example["messages"]:
            if msg["role"] == "user":
                for item in msg["content"]:
                    if item["type"] == "image" and isinstance(item["image"], str):
                        pil_image = load_image(item["image"])
                        pil_image = resize_with_aspect_ratio(pil_image)
                        item["image"] = pil_image
    texts = [processor.apply_chat_template(e["messages"], tokenize=False) for e in examples]
    image_inputs = [process_vision_info(e["messages"])[0] for e in examples]
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["messages"] = [ex["messages"] for ex in examples]
    batch["labels"] = labels
    batch["task"] = [t['task'] for t in examples]
    return batch


def prepare_datasets(args, prefix_token_strs, mode="train"):
    """Prepare datasets and apply prompt templates."""
    image_root = f'{args.your_datapath}/{mode}'
    if args.dataname.lower() == 'emoset': 
        jsonl_dir = f'{args.your_datapath}/annotation'
    else:  
        jsonl_dir = None

    dataset_emo = EmotionDataset(image_root, json_dir=jsonl_dir, dataname=args.dataname, mode=mode)
    dataset_no_prefix_emo = [create_messages(row, mode) for row in dataset_emo]
    
    if mode == "test":
        return dataset_no_prefix_emo
    
    dataset_with_prefix_emo = create_prefix_messages(dataset_no_prefix_emo, prefix_token_strs)
    return dataset_with_prefix_emo 


def load_config(config_path, mode="train"):
    """Load hyperparameters for the given mode (train/test) from YAML config."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config[mode]