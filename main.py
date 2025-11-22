import argparse
from src.config import load_config
from src.data import build_stage1_train_dataset, build_stage2_datasets, create_prefix_messages
from src.model import build_stage1_model, build_stage2_model, setup_prefix_tokens_and_mask
from src.training import build_trainer, save_soft_prompt_embeddings, save_query_fuser_ckpt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/train.yaml")
    parser.add_argument("--stage", type=str, choices=["stage1", "stage2"], default="stage1") 
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--aspect_data_path", type=str, default=None)
    parser.add_argument("--train_data_path", type=str, default=None)
    return parser.parse_args()

def main(): 
    args = parse_args()

    # Load merged config: common + train/<stage>
    train_config = load_config(
        args.config, 
        stage=args.stage
    )

    # Optional override for output directory
    if args.output_dir is not None:
        train_config["output_dir"] = args.override_output_dir

    # -------------------------------
    # Stage 1: learn view-specific q_view + cross-attention modules
    # -------------------------------
    if args.stage == "stage1":
        # Build AspectDataset (scene/object/action/brightness/colorfulness/etc.)
        train_dataset = build_stage1_train_dataset(train_config)

        # Load frozen VLM + initialize MASP Stage 1 model
        spmodel, processor = build_stage1_model(train_config)

        # Construct SFT trainer and start training
        trainer = build_trainer(spmodel, processor, train_dataset, train_config)
        trainer.train()
         
        save_query_fuser_ckpt(spmodel, train_config['output_path'])

    # -------------------------------
    # Stage 2: learn soft prompts
    # -------------------------------
    elif args.stage == "stage2":
        # Load emotion classification datasets (train/test)
        train_no_prefix, test_no_prefix = build_stage2_datasets(train_config)

        # Load VLM + Stage 1 pretrained q_view / cross-attn
        # (weights loaded from ckpt in build_stage2_model)
        spmodel, processor = build_stage2_model(train_config)

        # Add prefix soft tokens + enable gradient masking
        prefix, hook_handle = setup_prefix_tokens_and_mask(
            spmodel, processor, train_config
        )

        # Add prefix tokens to chat-format data
        train_with_prefix = create_prefix_messages(train_no_prefix, prefix) 

        # Build trainer & train only prefix embeddings
        trainer = build_trainer(spmodel, processor, train_with_prefix, train_config)
        trainer.train()
        
        # Save trained soft-prompt embeddings
        num_prefix = train_config["num_special_tokens_in_prefix"]
        save_soft_prompt_embeddings(
            model=spmodel.vlm,
            processor=processor,
            num_special_tokens_in_prefix=num_prefix,
            save_ckpt=train_config["output_path"],
        )
        hook_handle.remove()


if __name__ == "__main__":
    main()
