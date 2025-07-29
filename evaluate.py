import os
import argparse
import torch
import random
import numpy as np
from data_utils import load_config, prepare_datasets, load_image, resize_with_aspect_ratio
from softprompt_model import load_pretrained_model, initialize_and_load_softprompt_model, load_soft_prompt_embeddings
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(spmodel, processor, dataset, process_vision_info, max_new_tokens=10):
    """Run evaluation and return accuracy, predictions, and gold labels."""
    pred_ls, golden_ls = [], []
    num_correct, num_total = 0, 0

    for example in tqdm(dataset, desc="Evaluating"):
        for msg in example["messages"]:
            if msg["role"] == "user":
                for item in msg["content"]:
                    if item["type"] == "image" and isinstance(item["image"], str):
                        item["image"] = resize_with_aspect_ratio(load_image(item["image"]))

        messages = example["messages"] 
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        qwen_img = process_vision_info(messages)[0][0]
        input_batch = processor(text=prompt, images=qwen_img, return_tensors="pt", padding=True).to(spmodel.vlm.device)

        with torch.no_grad():
            generated_ids = spmodel.generate(
                **input_batch,
                max_new_tokens=max_new_tokens,
                repetition_penalty=2.0,
                task=[example['task']]
            )

        response = processor.batch_decode(generated_ids, skip_special_tokens=True)
        pred = response[0].lower().split('\n')[0].strip(' ')
        gold = example["label"]

        pred_ls.append(pred)
        golden_ls.append(gold)

        if pred == gold:
            num_correct += 1
        num_total += 1
        print('[Pred] ', pred, '[GT] ', gold)

    accuracy = num_correct / num_total if num_total > 0 else 0
    return accuracy 

def main():
    # ====== Argument Parsing ======
    parser = argparse.ArgumentParser()
    parser.add_argument("--your_datapath", type=str, help="path to your dataset")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device number to use")
    parser.add_argument("--dataname", type=str, default="EmoSet", help="dataset name (e.g., emotion6, abstract, artphoto, FI)")
    parser.add_argument("--load_sp_ckpt", type=str, default=None, help="checkpoint to load for soft prompts")
    parser.add_argument("--load_model_ckpt", type=str, default=None, help="checkpoint to load for aspect model")
    parser.add_argument("--shuffle", type=lambda x: x.lower() == 'true', default=True, help="whether to shuffle the dataset")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    # ====== Environment Setup ======
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["HF_HOME"] = "/sdc/huggingface"

    # ====== Load Hyperparameters ======
    test_config = load_config(args.config, mode="test")

    # ====== Set Seed ======
    set_seed(42)

    # ====== Model & Tokenizer Preparation ======
    model, processor = load_pretrained_model()
    model = model.to('cuda')
    for param in model.parameters():
        param.requires_grad = False

    # ====== Load Soft Prompt Embeddings ======
    prefix_token_strs = load_soft_prompt_embeddings(args.load_sp_ckpt, processor.tokenizer, model)

    # ====== Data Preparation ======
    eval_dataset = prepare_datasets(args, prefix_token_strs, mode="test")

    # ====== SoftPromptEmotionModel Initialization and Load Parameters ======
    spmodel = initialize_and_load_softprompt_model(model, processor, args.load_model_ckpt)

    # ====== Run Evaluation ======
    acc = evaluate(
        spmodel, processor, eval_dataset, process_vision_info,
        max_new_tokens=test_config.get("max_new_tokens", 10)
    )
    print(f"Accuracy: {acc:.4f}") 

if __name__ == "__main__":
    main()