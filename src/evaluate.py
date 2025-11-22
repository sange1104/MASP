import argparse
import torch
from tqdm import tqdm

from config import load_config
from data import (
    build_stage2_datasets,
    create_prefix_messages,
    load_image,
    resize_with_aspect_ratio,
)
from model import (
    build_stage2_model,
    setup_prefix_tokens_and_mask,
)
from qwen_vl_utils import process_vision_info  # 경로는 실제 코드에 맞게 조정


def parse_args():
    """
    Parse command-line arguments for Stage 2 evaluation.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../config/eval.yaml") 
    return parser.parse_args()


def get_model_device(model):
    """
    Get the device on which the model parameters are allocated.
    """
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device

def load_trained_prefix_from_ckpt(spmodel, processor, ckpt_path):
    """
    Load trained prefix-token embeddings from checkpoint and register them
    in the tokenizer and VLM embedding matrix.

    The checkpoint is expected to be a dict:
        {token_str: embedding_tensor}
    """
    device = get_model_device(spmodel)

    # Load dict: { "<|reserved_special_token_0|>": tensor(D), ... }
    loaded_embeddings = torch.load(ckpt_path, map_location=device)

    # Register token strings as additional special tokens
    prefix_token_strs = list(loaded_embeddings.keys())
    processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": prefix_token_strs}
    )
    spmodel.vlm.resize_token_embeddings(len(processor.tokenizer))

    # Copy embedding weights into the embedding matrix
    with torch.no_grad():
        for token_str, embedding in loaded_embeddings.items():
            token_id = processor.tokenizer.convert_tokens_to_ids(token_str)

            # Safety check: token must be properly registered
            if token_id == processor.tokenizer.unk_token_id:
                raise ValueError(f"{token_str} was not properly registered in tokenizer.")

            spmodel.vlm.get_input_embeddings().weight[token_id] = (
                embedding.to(device)
            )

    # Build concatenated prefix string to prepend to text
    prefix = "".join(prefix_token_strs)
    return prefix


def preprocess_example_images_inplace(example):
    """
    Convert image paths inside a chat-format example into resized PIL images.
    """
    for msg in example["messages"]:
        if msg["role"] == "user":
            for item in msg["content"]:
                if item["type"] == "image" and isinstance(item["image"], str):
                    img = load_image(item["image"])
                    img = resize_with_aspect_ratio(img)
                    item["image"] = img


def build_generation_inputs(example_with_prefix, processor, device):
    """
    Build model input batch (text + image) for a single example.

    Returns:
        dict: Processor output dict moved to the correct device.
    """
    messages = example_with_prefix["messages"]

    # Build chat template with generation prompt
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Qwen2-VL vision preprocessing
    qwen_img = process_vision_info(messages)[0][0]

    # Tokenize text + image
    input_batch = processor(
        text=prompt,
        images=qwen_img,
        return_tensors="pt",
        padding=True,
    )
    input_batch = {k: v.to(device) for k, v in input_batch.items()}

    return input_batch


@torch.no_grad()
def predict_single_example(spmodel, processor, example_with_prefix, raw_example, max_new_tokens, device):
    """
    Run a single forward generation step and return (pred, gold) strings.
    """
    # 1) Prepare images (path → PIL) in-place
    preprocess_example_images_inplace(example_with_prefix)

    # 2) Build model inputs
    input_batch = build_generation_inputs(
        example_with_prefix=example_with_prefix,
        processor=processor,
        device=device,
    )

    # 3) Generate with Stage 2 MASP model
    generated_ids = spmodel.generate(
        **input_batch,
        max_new_tokens=max_new_tokens,
        task=[raw_example["task"]],  
    )

    # 4) Decode prediction
    response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    pred = response[0].lower().split("\n")[0]
    gold = raw_example["label"]

    return pred, gold

@torch.no_grad()
def evaluate_stage2(config):
    """
    Evaluate Stage 2 MASP model and print accuracy for the given dataset.
    """
    dataname = config["dataname"]

    # 1) Load datasets (train, test) and keep only the test split
    test_no_prefix = build_stage2_datasets(config, mode='test')

    # 2) Build Stage 2 model and processor (with Stage 1 weights loaded)
    spmodel, processor = build_stage2_model(config)
    device = get_model_device(spmodel)
    spmodel.to(device)
    spmodel.eval()

    # 3) Load trained soft-prompt (prefix) embeddings
    prefix_ckpt_path = config["prefix_ckpt_path"]
    prefix = load_trained_prefix_from_ckpt(
        spmodel,
        processor,
        prefix_ckpt_path
    )

    # 4) Add prefix tokens to test messages
    test_with_prefix = create_prefix_messages(test_no_prefix, prefix)

    # 5) Evaluation loop
    num_correct, num_total = 0, 0
    max_new_tokens = config.get("eval_max_new_tokens", 3)

    for example_with_prefix, raw_example in tqdm(
        zip(test_with_prefix, test_no_prefix),
        total=len(test_with_prefix),
        desc=f"Evaluating {dataname} (Stage 2)"
    ):
        pred, gold = predict_single_example(
            spmodel=spmodel,
            processor=processor,
            example_with_prefix=example_with_prefix,
            raw_example=raw_example,
            max_new_tokens=max_new_tokens,
            device=device,
        )

        num_total += 1
        if pred == gold:
            num_correct += 1
        print(pred, gold, num_correct/num_total)

    accuracy = num_correct / num_total if num_total > 0 else 0.0
    print(f"[{dataname}] Accuracy: {accuracy:.4f}")



def main():
    """
    Entry point for Stage 2 evaluation.
    """
    args = parse_args()
    train_config = load_config(
        args.config, 
        stage="stage2"
    )
    evaluate_stage2(train_config)


if __name__ == "__main__":
    main()