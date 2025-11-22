from torch.utils.data import Dataset
import os, random, json, copy
from qwen_vl_utils import process_vision_info
from PIL import Image
 
class AspectDataset(Dataset):
    '''
    Dataset for multi-view (scene, object, action, etc.) descriptions of images.

    Each sample corresponds to:
        - an image path
        - a specific "view" task (e.g., scene, object, brightness)
        - a view-specific prompt
        - an emotion label (if available)
        - a view-specific textual label (e.g., "very high", "happy", etc.)
    '''
    def __init__(self, 
                 image_root, 
                 view_names,
                 prompt_json_path="./config/prompt/prompts.json",
                 mode='train'):
        # List of all training samples.
        # Each element is a dictionary with keys:
        #   "image", "task", "text" (prompt), "emotion", "label"
        self.samples = []

        # Fixed prompts for each view/task type.
        # These will be used as the user text prompt when querying a VLM.
        self.prompt_dict = load_prompts_from_json(prompt_json_path)
        
        # List of view names to use (e.g., scene, object, action, brightness, etc.). 
        self.view_names = view_names

        # Build the full list of samples from the directory structure
        jsonl_dir = os.path.join(os.path.dirname(image_root), 'annotation')
        self._build_samples(jsonl_dir, image_root, mode)

    def _build_samples(self, jsonl_dir, image_root, mode):
        '''
        Traverse the image root directory and build samples by pairing each image
        with its annotation JSON and decomposing the annotation into multiple
        view-specific samples.
        '''
        # image_root is expected to be structured as:
        #   image_root/
        #       emotion_1/
        #           img_1.jpg
        #           img_2.jpg
        #       emotion_2/
        #           ...
        for emo in os.listdir(image_root):
            emo_dir = os.path.join(image_root, emo)
            if not os.path.isdir(emo_dir):
                # Skip non-directory entries to be safe
                continue

            for img_file in os.listdir(emo_dir):
                # Derive image_id (used to locate its JSON annotation)
                image_id = img_file.split('.')[0]
                image_path = os.path.join(emo_dir, img_file)
                json_path = os.path.join(jsonl_dir, emo, image_id + '.json')

                # Load annotation for this image
                with open(json_path, "r") as f:
                    ann = json.load(f)

                # Convert annotations for each view into individual samples
                self._add_view_samples_for_image(image_path, ann, mode)
                
    def _select_prompt_for_mode(self, view, mode):
        '''
        Choose a text prompt for a given mode ("train" or not) per each view.

        For training:
            - Randomly choose a prompt template from self.prompts and format it
              with the current category list.
        For testing/validation:
            - Use the first prompt template for deterministic behavior.

        Args:
            view (str): "scene", "colorfulness", ...
            mode (str): "train" or "test". 

        Returns:
            str: Fully formatted prompt string.
        '''  
        if mode == "train":
            chosen_prompt = random.choice(self.prompt_dict[view])
        else:
            # Use a fixed template for reproducible evaluation
            chosen_prompt = self.prompts[0]

        # Second formatting step inserts the human-readable category string
        return chosen_prompt
    
    def _add_view_samples_for_image(self, image_path, ann, mode):
        '''
        Given an image path and its annotation dictionary, expand it into
        multiple (image, view) samples and append them to `self.samples`.

        Args:
            image_path (str): Path to the image file.
            ann (dict): Annotation containing per-view entries and emotion.
            ann (string): train or not
        '''
        for view in self.view_names:
            v = ann.get(view, None)
            if not v:
                # Skip views that do not exist in this annotation
                continue

            # If the annotation is a list, join it into a comma-separated string
            if isinstance(v, list):
                v = ', '.join(v)

            # Map continuous values (e.g., brightness ∈ [0,1]) to categorical strings
            if view in ['brightness', 'colorfulness'] and v is not None:
                v = self.describe_continuous_value(v)

            # Ensure labels use spaces instead of underscores (e.g., "very_high" -> "very high")
            v = v.replace('_', ' ')

            # Append one sample for this (image, view) combination
            self.samples.append({
                "image": image_path,
                "task": view,
                "text": self._select_prompt_for_mode(view, mode),        # view-specific prompt
                "emotion": ann.get('emotion', None),   # overall emotion, if any
                "label": v.lower()                     # normalized view label
            })

    def describe_continuous_value(self, value):
        '''
        Map a float value in [0, 1] to a 5-level categorical description.

        Args:
            value (float): The input value (0~1).

        Returns:
            str: Natural language description (very low / low / moderate / high / very high).
        '''
        assert 0.0 <= value <= 1.0, "Value must be between 0 and 1."

        # Define bins for mapping a continuous value to 5 discrete categories
        bins = [
            (0.0, 0.2, "very low"),
            (0.2, 0.4, "low"),
            (0.4, 0.6, "moderate"),
            (0.6, 0.8, "high"),
            (0.8, 1.0, "very high")
        ]

        # Find which bin the input belongs to
        for lower, upper, label in bins:
            if lower <= value < upper or (upper == 1.0 and value == 1.0):
                return label

        # Fallback (should not happen if the assert holds)
        return "unknown"

    def __len__(self):
        '''
        Return the total number of samples in the dataset.
        '''
        return len(self.samples)

    def __getitem__(self, idx):
        '''
        Retrieve a single sample by index.

        Args:
            idx (int): Sample index.

        Returns:
            dict: Sample dictionary containing image path, task, prompt text,
                  emotion, and view label.
        '''
        sample = self.samples[idx]
        return sample 
  

def create_prefix_messages(dataset, prefix):
    '''
    Add a text prefix to the user message content for each sample in a dataset.

    This function:
        - Deep-copies each sample to avoid modifying the original dataset.
        - Prepends a prefix-only text chunk to the user's content list.
        - Also adds the prefix in front of each text message item.

    Expected input format:
        sample["messages"][1]["content"] = [
            {"type": "image", "image": ...},
            {"type": "text", "text": ...},
            ...
        ]

    Args:
        dataset (Sequence[dict]): Dataset where each element contains a "messages" field.
        prefix (str or list[str]): Prefix string or list of strings; if list, they are joined.

    Returns:
        list[dict]: New list of samples with modified messages.
    '''
    # If prefix is a list of strings, concatenate into one string
    prefix = "".join(prefix) if isinstance(prefix, list) else prefix

    new_dataset = []
    for i in range(len(dataset)):
        # Deep-copy to ensure we don't mutate the original dataset
        data = copy.deepcopy(dataset[i])

        # We assume the user message is at index 1, following a standard chat format
        content = data['messages'][1]['content']
        new_content = []

        # Add a separate text chunk that contains only the prefix
        new_content.append({"type": "text", "text": prefix})

        # For each original content item, add prefix to text items and keep images as-is
        for item in content:
            if item["type"] == "image":
                new_content.append(item)
            elif item["type"] == "text":
                new_content.append({"type": "text", "text": prefix + item["text"]})

        # Update the user message content and append to the new dataset
        data['messages'][1]['content'] = new_content
        new_dataset.append(data)

    return new_dataset

def load_prompts_from_json(filepath):
    """
    Load view-specific prompt templates from a JSON file.

    The JSON file must map each view name to a list of prompt strings.
    Example:
        {
            "brightness": ["...", "..."],
            "facial_expression": ["...", "..."]
        }

    Returns:
        dict[str, list[str]]: Mapping: view_name → list of prompt strings.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts = {}
    for view, items in data.items():
        # Keep only non-empty strings
        valid_items = [
            s.strip() for s in items
            if isinstance(s, str) and s.strip()
        ]
        prompts[view] = valid_items

    return prompts 

def create_chat_messages(row, include_label=True):
    '''
    Build unified chat-style messages for both training and inference.

    This function constructs a standard 3-part (system, user, assistant) message format,
    but includes the assistant reply only when `include_label=True`.

    Expected row keys:
        - row["image"]: (str) path to an image OR an image object
        - row["text"]: (str) user-side text prompt
        - row["label"]: (str) ground-truth label (required only when include_label=True)

    Args:
        row (dict):
            A dictionary representing an input sample.
        include_label (bool):
            Whether to include assistant messages (ground truth labels).
            - True → Training mode (supervised)
            - False → Inference/Test mode

    Returns:
        dict: The same row with a "messages" list added.
    '''

    # Common system + user messages
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": row['image']},
                {"type": "text", "text": row['text']}
            ]
        }
    ]

    # Append assistant message only in training mode
    if include_label:
        messages.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": row['label']}
            ]
        })

    row["messages"] = messages
    return row 

def get_emotion_categories(dataname):
    '''
    Return the list of emotion category names for a given dataset name.

    Args:
        dataname (str): Name of the dataset (e.g., "emotion6", "abstract", "flickr").

    Returns:
        list[str]: List of emotion category strings.
    '''
    # Normalize name for case-insensitive matching
    name = dataname.lower() if dataname is not None else ""

    if name == 'emotion6':
        return ['sadness', 'fear', 'surprise', 'anger', 'joy', 'disgust']
    elif name in ['abstract', 'artphoto']:
        return ['sad', 'amusement', 'fear', 'disgust', 'excitement', 'awe', 'content', 'anger']
    elif name in ['flickr', 'instagram']:
        return ['positive', 'negative']
    else:
        # Default category list for other datasets
        return ['sadness', 'amusement', 'fear', 'disgust', 'excitement', 'awe', 'contentment', 'anger']
    
def build_stage1_train_dataset(config):
    emo_root = config["aspect_data_path"]
    train_root = f"{emo_root}/train" 
    view_names = config["VIEW_NAMES"]
    prompt_path = config["prompt_json_path"]

    dataset = AspectDataset(train_root, view_names, prompt_json_path=prompt_path)
    dataset_no_prefix = [create_chat_messages(row, include_label=True) for row in dataset]
    return dataset_no_prefix

def build_stage2_datasets(config, mode='train'):
    if mode == 'train':
        image_root = config["train_data_path"]
        dataname = config["dataname"]
        prompt_path = config["prompt_json_path"]

        train_dataset = EmotionDataset(image_root, dataname=dataname, prompt_json_path=prompt_path, mode="train")
        test_dataset = EmotionDataset(image_root, dataname=dataname, prompt_json_path=prompt_path, mode="test")

        train_no_prefix = [create_chat_messages(row, include_label=True) for row in train_dataset]
        test_no_prefix = [create_chat_messages(row, include_label=False) for row in test_dataset]

        return train_no_prefix, test_no_prefix
    else:
        image_root = config["test_data_path"]
        dataname = config["dataname"]
        prompt_path = config["prompt_json_path"]

        test_dataset = EmotionDataset(image_root, dataname=dataname, prompt_json_path=prompt_path, mode="test")

        test_no_prefix = [create_chat_messages(row, include_label=False) for row in test_dataset]

        return test_no_prefix
        

def load_image(image_path):
    '''
    Load an image from disk and convert it to an RGB PIL.Image.

    Args:
        image_path (str): Path to the image file.

    Returns:
        PIL.Image.Image: Loaded RGB image.
    '''
    # Open the image file and force conversion to 3-channel RGB
    image = Image.open(image_path).convert("RGB")
    return image


def resize_with_aspect_ratio(img, max_len=1024):
    '''
    Resize an image while preserving aspect ratio, such that the longer side
    does not exceed `max_len`.

    Args:
        img (PIL.Image.Image): Input image.
        max_len (int): Maximum allowed size for the longer edge.

    Returns:
        PIL.Image.Image: Resized (or original) image.
    '''
    # Get original width and height
    w, h = img.size

    # If the image is already small enough, return as-is
    if max(w, h) <= max_len:
        return img

    # Compute new dimensions based on which side is longer
    if w >= h:
        new_w = max_len
        new_h = int(h * max_len / w)
    else:
        new_h = max_len
        new_w = int(w * max_len / h)

    # Use high-quality LANCZOS resampling when resizing
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)
 

class SubsetDataset(Dataset):
    '''
    A dataset wrapper that exposes only a subset of examples (specified by indices)
    from a base dataset.

    Useful when you want to create a smaller, stratified subset while reusing
    the original dataset's indexing and storage.
    '''
    def __init__(self, base_dataset, indices):
        # Reference to the original, full dataset
        self.base_dataset = base_dataset
        # List of indices to keep from the base dataset
        self.indices = indices

    def __len__(self):
        '''
        Return the number of samples in the subset.
        '''
        return len(self.indices)

    def __getitem__(self, idx):
        '''
        Retrieve a sample from the base dataset using the subset index.

        Args:
            idx (int): Index into the subset.

        Returns:
            Any: Corresponding sample from the base dataset.
        '''
        return self.base_dataset[self.indices[idx]]
  

def get_emotion_categories(dataname):
    '''
    Return the list of emotion category names for a given dataset name.

    Args:
        dataname (str): Name of the dataset (e.g., "emotion6", "abstract", "flickr").

    Returns:
        list[str]: List of emotion category strings.
    '''
    # Normalize name for case-insensitive matching
    name = dataname.lower() if dataname is not None else ""

    if name == 'emotion6':
        return ['sadness', 'fear', 'surprise', 'anger', 'joy', 'disgust']
    elif name in ['abstract', 'artphoto']:
        return ['sad', 'amusement', 'fear', 'disgust', 'excitement', 'awe', 'content', 'anger']
    elif name in ['flickr', 'instagram']:
        return ['positive', 'negative']
    else:
        # Default category list for other datasets
        return ['sadness', 'amusement', 'fear', 'disgust', 'excitement', 'awe', 'contentment', 'anger']


def format_prompts_with_categories(prompts, ctg_list):
    '''
    Apply an initial formatting pass to prompt templates using the category list.

    NOTE: This follows the original behavior where each prompt is formatted once
    with a shuffled category list, and can be formatted again later.

    Args:
        prompts (list[str]): Raw prompt templates loaded from file.
        ctg_list (list[str]): List of emotion categories.

    Returns:
        list[str]: List of partially formatted prompt templates.
    '''
    # For each template, apply a first-level formatting with a random permutation of categories.
    # The template may still contain placeholders for subsequent formatting.
    return [prompt.format(random.sample(ctg_list, len(ctg_list))) for prompt in prompts]


class EmotionDataset(Dataset):
    '''
    Dataset for emotion classification with prompt-based vision-language models.

    This dataset:
        - Traverses an image directory structure split by emotion labels.
        - Optionally loads per-image JSON annotations (e.g., for "Emoset").
        - Associates each image with a text prompt and an emotion label.
    '''
    def __init__(
        self,
        image_root, 
        dataname,
        prompt_json_path="./config/prompt/prompts.json",
        mode="train"
    ):
        '''
        Initialize the EmotionDataset.

        Args:
            image_root (str): Root directory of images, expected layout:
                              image_root/{mode}/{emotion_name}/*.jpg
            json_dir (str, optional): Root directory of JSON annotations
                                      (used for datasets like "Emoset").
            dataname (str, optional): Logical name of the dataset
                                      (e.g., "emotion6", "emoset", etc.).
            prompt_txt_path (str): Path to a text file containing prompt templates.
            mode (str): "train" or "test", used to control prompt sampling.
        '''
        # List of sample dictionaries:
        #   { "image": path, "text": prompt, "task": "emotion", "label": emotion_label }
        self.samples = []

        # Determine emotion categories based on dataset name
        ctg_list = get_emotion_categories(dataname)

        if dataname is not None and dataname.lower() == 'emoset':        
            json_dir = os.path.join(image_root, 'annotation')
        else:
            json_dir = None

        # Load base prompt templates and apply an initial formatting
        raw_prompts = load_prompts_from_json(prompt_json_path)
        self.prompts = format_prompts_with_categories(raw_prompts['emotion'], ctg_list)

        # Build all samples from the image directory and optional json annotations
        self._build_samples(image_root, json_dir, dataname, ctg_list, mode)

    def _build_samples(self, image_root, json_dir, dataname, ctg_list, mode):
        '''
        Populate `self.samples` by scanning the directory structure and pairing
        each image with a text prompt + emotion label.

        Args:
            image_root (str): Root directory of images.
            json_dir (str or None): Root directory of annotations (for Emoset-like datasets).
            dataname (str or None): Dataset name used for handling special cases.
            ctg_list (list[str]): List of emotion categories.
            mode (str): "train" or "test".
        '''
        # Images are expected under: image_root/mode/emotion/*
        mode_root = os.path.join(image_root, mode)

        for emo in os.listdir(mode_root):
            emo_dir = os.path.join(mode_root, emo)
            if not os.path.isdir(emo_dir):
                # Ignore files at this level
                continue

            # For training, we may reshuffle category order per emotion directory
            local_ctg_list = ctg_list
            if mode == "train":
                local_ctg_list = random.sample(ctg_list, len(ctg_list))

            for img_file in os.listdir(emo_dir):
                image_path = os.path.join(emo_dir, img_file)

                # Sample or fix a prompt depending on mode
                prompt = self._select_prompt_for_mode(mode, local_ctg_list)

                if dataname is not None and dataname.lower() == 'emoset':
                    image_id = img_file.split('.')[0]
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
                    self.samples.append({
                        "image": image_path,
                        "text": prompt,
                        "task": 'emotion',
                        "label": emo
                    })

    def _select_prompt_for_mode(self, mode, ctg_list):
        '''
        Choose a text prompt for a given mode ("train" or "test").

        For training:
            - Randomly choose a prompt template from self.prompts and format it
              with the current category list.
        For testing/validation:
            - Use the first prompt template for deterministic behavior.

        Args:
            mode (str): "train" or "test".
            ctg_list (list[str]): Category list to be inserted into the prompt.

        Returns:
            str: Fully formatted prompt string.
        '''
        category_str = ', '.join(ctg_list)

        if mode == "train":
            chosen_template = random.choice(self.prompts)
        else:
            # Use a fixed template for reproducible evaluation
            chosen_template = self.prompts[0]

        # Second formatting step inserts the human-readable category string
        return chosen_template.format(category_str)

    def __len__(self):
        '''
        Return the number of emotion samples in the dataset.
        '''
        return len(self.samples)

    def __getitem__(self, idx):
        '''
        Retrieve one emotion sample by index.

        Args:
            idx (int): Sample index.

        Returns:
            dict: Dictionary containing "image", "text", "task", and "label".
        '''
        sample = self.samples[idx] 
        return sample

def create_prefix_messages(dataset, prefix):
    '''
    Add a text prefix to the user message content for each sample in a dataset.

    This function:
        - Deep-copies each sample to avoid modifying the original dataset.
        - Prepends a prefix-only text chunk to the user's content list.
        - Also adds the prefix in front of each text message item.

    Expected input format:
        sample["messages"][1]["content"] = [
            {"type": "image", "image": ...},
            {"type": "text", "text": ...},
            ...
        ]

    Args:
        dataset (Sequence[dict]): Dataset where each element contains a "messages" field.
        prefix (str or list[str]): Prefix string or list of strings; if list, they are joined.

    Returns:
        list[dict]: New list of samples with modified messages.
    '''
    # If prefix is a list of strings, concatenate into one string
    prefix = "".join(prefix) if isinstance(prefix, list) else prefix

    new_dataset = []
    for i in range(len(dataset)):
        # Deep-copy to ensure we don't mutate the original dataset
        data = copy.deepcopy(dataset[i])

        # We assume the user message is at index 1, following a standard chat format
        content = data['messages'][1]['content']
        new_content = []

        # Add a separate text chunk that contains only the prefix
        new_content.append({"type": "text", "text": prefix})

        # For each original content item, add prefix to text items and keep images as-is
        for item in content:
            if item["type"] == "image":
                new_content.append(item)
            elif item["type"] == "text":
                new_content.append({"type": "text", "text": prefix + item["text"]})

        # Update the user message content and append to the new dataset
        data['messages'][1]['content'] = new_content
        new_dataset.append(data)

    return new_dataset

def create_chat_messages(row, include_label=True):
    '''
    Build unified chat-style messages for both training and inference.

    This function constructs a standard 3-part (system, user, assistant) message format,
    but includes the assistant reply only when `include_label=True`.

    Expected row keys:
        - row["image"]: (str) path to an image OR an image object
        - row["text"]: (str) user-side text prompt
        - row["label"]: (str) ground-truth label (required only when include_label=True)

    Args:
        row (dict):
            A dictionary representing an input sample.
        include_label (bool):
            Whether to include assistant messages (ground truth labels).
            - True → Training mode (supervised)
            - False → Inference/Test mode

    Returns:
        dict: The same row with a "messages" list added.
    '''

    # Common system + user messages
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": row['image']},
                {"type": "text", "text": row['text']}
            ]
        }
    ]

    # Append assistant message only in training mode
    if include_label:
        messages.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": row['label']}
            ]
        })

    row["messages"] = messages
    return row 

def _prepare_images_in_messages(examples):
    '''
    Load and resize images found inside "messages" for each example.

    This function walks through:
        example["messages"] → msg["content"] → item["type"] == "image"
    If an image path string is found, it loads the image (PIL), resizes it
    with aspect ratio preserved, and stores the PIL object back into the item.

    Args:
        examples (list):
            List of training samples, each containing a "messages" field.

    Returns:
        None (operates in-place).
    '''
    for example in examples:
        for msg in example["messages"]:
            if msg["role"] == "user":
                for item in msg["content"]:
                    if item["type"] == "image" and isinstance(item["image"], str):
                        pil_image = load_image(item["image"])
                        pil_image = resize_with_aspect_ratio(pil_image)
                        item["image"] = pil_image


def collate_fn(examples, processor):
    '''
    Collate function for Qwen2-VL training with soft prompt tuning (MASP).

    This function performs three major operations:

    (1) Image preparation:
        - Locates images embedded inside chat messages.
        - Loads + resizes them (PIL → resized PIL).
        - This preserves the multimodal chat format required for Qwen2-VL.

    (2) Chat template → text conversion:
        - Qwen2-VL expects chat-style prompt strings.
        - processor.apply_chat_template(..., tokenize=False) returns the raw text.

    (3) Tokenization & batch assembly:
        - processor(text, images) converts text + image into VLM-ready tensors.
        - input_ids are duplicated as labels, then:
            • padding tokens are masked to -100
            • vision tokens (<image_start>, <image_end>, etc.) masked to -100
        - Additional fields ("messages", "task") stored back into batch for the model.

    Args:
        examples (list):
            Each element contains:
                - "messages": chat-style conversation
                - "task": view/emotion task label
        processor (AutoProcessor):
            Qwen2-VL processor for tokenization & multimodal conversion.

    Returns:
        batch (dict):
            {
                "input_ids": Tensor,
                "pixel_values": Tensor,
                "attention_mask": Tensor,
                "image_grid_thw": Tensor,
                "messages": [...],
                "labels": Tensor,
                "task": list[str],
            }
    '''

    # -------------------------------------------------------
    # 1) Load & resize images embedded inside chat messages
    # -------------------------------------------------------
    _prepare_images_in_messages(examples)

    # -------------------------------------------------------
    # 2) Convert messages → chat template text
    # -------------------------------------------------------
    texts = [
        processor.apply_chat_template(e["messages"], tokenize=False)
        for e in examples
    ]

    # Extract image metadata for processor (pixel_values, grid_thw, etc.)
    image_inputs = [
        process_vision_info(e["messages"])[0]
        for e in examples
    ]

    # -------------------------------------------------------
    # 3) Tokenization (text + images)
    # processor returns:
    #     - input_ids
    #     - attention_mask
    #     - pixel_values (vision tokens)
    #     - image_grid_thw (vision grid info)
    # -------------------------------------------------------
    batch = processor(
        text=texts,
        images=image_inputs,
        return_tensors="pt",
        padding=True
    )

    # -------------------------------------------------------
    # 4) Prepare labels
    # -------------------------------------------------------

    # Clone input_ids → labels
    labels = batch["input_ids"].clone()

    # (a) Mask padding tokens
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # (b) Mask Qwen2-VL's special vision token IDs
    #     These must not contribute to LM loss.
    VISION_TOKEN_IDS = [151644, 151645, 151652, 151653, 151655]
    for tok_id in VISION_TOKEN_IDS:
        labels[labels == tok_id] = -100

    batch["labels"] = labels

    # -------------------------------------------------------
    # 5) Keep additional metadata (messages, task)
    # -------------------------------------------------------
    batch["messages"] = [ex["messages"] for ex in examples]
    batch["task"] = [ex["task"] for ex in examples]

    return batch