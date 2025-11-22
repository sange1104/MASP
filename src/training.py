from trl import SFTConfig, SFTTrainer
from .data import collate_fn
import torch, os

def build_sft_config(train_config):
    """
    Construct an SFTConfig object from the training configuration dictionary.
    """
    return SFTConfig(
        per_device_train_batch_size=train_config["per_device_train_batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        warmup_ratio=train_config["warmup_ratio"],
        num_train_epochs=train_config["num_train_epochs"],
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
        dataset_text_field=train_config["dataset_text_field"],
        dataset_kwargs=train_config["dataset_kwargs"],
        remove_unused_columns=train_config["remove_unused_columns"],
        output_dir=train_config["output_dir"],
    )


def build_trainer(spmodel, processor, train_dataset, train_config):
    """
    Build an SFTTrainer with the MASP model, processed dataset, and SFT config.
    """
    # Convert config → SFTConfig
    sft_args = build_sft_config(train_config)

    # Create trainer for supervised fine-tuning
    return SFTTrainer(
        model=spmodel,
        train_dataset=train_dataset[:5],
        tokenizer=processor.tokenizer,
        args=sft_args,
        data_collator=lambda examples: collate_fn(examples, processor),
    )

def save_soft_prompt_embeddings(model, processor, num_special_tokens_in_prefix, save_ckpt):
    """
    Save soft-prompt token embeddings into a checkpoint file.

    The function reconstructs reserved prefix token strings:
        <|reserved_special_token_0|>, ..., <|reserved_special_token_{N-1}|>
    and saves their embedding vectors as:
        {token_str: embedding_tensor(cpu)}.
    """
    if save_ckpt is None:
        return  # nothing to do

    # Reconstruct reserved prefix token strings
    prefix_token_strs = [
        f"<|reserved_special_token_{i}|>" for i in range(num_special_tokens_in_prefix)
    ]

    # Convert token strings to ids using the tokenizer
    prefix_token_ids = processor.tokenizer.convert_tokens_to_ids(prefix_token_strs)

    # Full embedding matrix
    embedding_weights = model.get_input_embeddings().weight.data

    saved_embeddings = {}

    # Extract embedding vector per prefix token id
    for token_str, token_id in zip(prefix_token_strs, prefix_token_ids):
        # Safety check: ensure the token is known by tokenizer
        if token_id == processor.tokenizer.unk_token_id:
            raise ValueError(f"{token_str} is not registered in the tokenizer.")
        saved_embeddings[token_str] = embedding_weights[token_id].cpu()

    # Write checkpoint to file
    torch.save(saved_embeddings, save_ckpt)

def save_query_fuser_ckpt(spmodel, save_path):
    """
    Save q_view, cross_attn, and fuser (if exists) into a checkpoint file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    ckpt = {
        "q_view": {k: v.detach().cpu() for k, v in spmodel.q_view.items()},
        "cross_attn": {k: v.state_dict() for k, v in spmodel.cross_attn.items()},
        "fuser": spmodel.fuser.state_dict() if hasattr(spmodel, "fuser") else None
    }

    torch.save(ckpt, save_path)
    print(f"[✔] Saved checkpoint to {save_path}")