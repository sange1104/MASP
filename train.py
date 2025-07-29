import os
import argparse
import torch
from transformers import Qwen2VLProcessor
from data_utils import prepare_datasets, collate_fn, load_config
from softprompt_model import prepare_model_and_tokenizer, register_embedding_gradient_mask, initialize_and_load_softprompt_model, save_soft_prompt_embeddings
from trl import SFTConfig, SFTTrainer



def main():

    # ====== Argument Parsing ======
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0", help="GPU device number to use")  
    parser.add_argument("--your_datapath", type=str, help="path to your dataset")
    parser.add_argument("--dataname", type=str, default="EmoSet", help="dataset name (e.g., emotion6, abstract, artphoto, FI)")
    parser.add_argument("--load_ckpt", type=str, default=None, help="checkpoint to load")
    parser.add_argument("--save_ckpt", type=str, default=None, help="checkpoint to save after training")
    args = parser.parse_args()

    # ====== Environment Setup ======
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["HF_HOME"] = "/sdc/huggingface" 

    # ====== Hyperparameters ======
    train_config = load_config("config.yaml", mode="train")

    # ====== Model & Tokenizer Preparation ======
    model, processor, prefix_token_ids, prefix_token_strs = prepare_model_and_tokenizer()
    NUM_SPECIAL_TOKENS_IN_PREFIX = train_config["num_special_tokens_in_prefix"]

    # ====== Embedding Gradient Masking ======
    hook_handle = register_embedding_gradient_mask(model, prefix_token_ids, NUM_SPECIAL_TOKENS_IN_PREFIX)

    # ====== Data Preparation ======
    train_dataset_with_prefix_emo = prepare_datasets(args, prefix_token_strs[:NUM_SPECIAL_TOKENS_IN_PREFIX])
    
    # ====== SoftPromptEmotionModel Initialization and Load Parameters ======
    spmodel = initialize_and_load_softprompt_model(model, processor, args.load_ckpt)

    # ====== Trainer Configuration & Training ====== 
    training_config = SFTConfig(
        per_device_train_batch_size=train_config["batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        warmup_ratio=train_config["warmup_ratio"],
        num_train_epochs=train_config["epochs"],
        learning_rate=train_config["learning_rate"],
        fp16=train_config["fp16"],
        bf16=train_config["bf16"],
        logging_steps=train_config["logging_steps"],
        optim=train_config["optim"],
        weight_decay=train_config["weight_decay"],
        lr_scheduler_type=train_config["lr_scheduler_type"],
        seed=train_config["seed"],
        save_strategy=train_config["save_strategy"],
        gradient_checkpointing=train_config["gradient_checkpointing"],
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=train_config["remove_unused_columns"]
    )

    trainer = SFTTrainer(
        model=spmodel,
        train_dataset=train_dataset_with_prefix_emo,
        tokenizer=processor.tokenizer,
        args=training_config,
        data_collator=lambda examples: collate_fn(examples, processor),
    )

    trainer.train()
    hook_handle.remove()

    # ====== Save Soft Prompt Embeddings ======
    save_soft_prompt_embeddings(model, processor, prefix_token_ids, NUM_SPECIAL_TOKENS_IN_PREFIX, args.save_ckpt)

if __name__ == "__main__":
    main()  