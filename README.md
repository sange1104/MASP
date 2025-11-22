# MASP: Multi-Aspect Soft Prompt Tuning for Emotion Reasoning in Vision-Language Models
 

## 🔍 Overview

Understanding human emotions from an image is a challenging yet essential task for vision-language models. While recent efforts have fine-tuned vision-language models to enhance emotional awareness, most approaches rely on global visual representations and fail to capture the nuanced, multi-faceted nature of emotional cues. Furthermore, most existing approaches adopt instruction tuning, which requires costly dataset construction and involves training a large number of parameters, thereby limiting scalability and efficiency. To address these challenges, we propose MASP, a novel framework for Multi-Aspect guided emotion reasoning with Soft Prompt tuning in vision-language models. MASP explicitly separates emotion-relevant visual cues via multi-aspect cross-attention modules and guides the language model using soft prompts, enabling efficient and scalable task adaptation without modifying the base model. Our method achieves state-of-the-art performance on various emotion recognition benchmarks, demonstrating that explicit modeling of multi-aspect emotional cues with soft prompt tuning leads to more accurate and interpretable emotion reasoning in vision-language models.

<p align="center">
  <img src="assets/masp_architecture.jpg" alt="MASP Architecture" width="900">
</p> 
 
 
## 📦 Setup 
```bash
git clone https://github.com/{id}/masp.git
cd masp
pip install -r requirements.txt
```

## 📂 Dataset Structure

The project assumes the following dataset directory layout.

Example: EmoSet

```bash
emoset
├── train
    ├── ...
├── test
│   ├── amusement
    ├── ...
└── annotation
    ├── amusement
    ├── ...
```

- Each emotion label corresponds to a subfolder.

- For datasets including annotation files (e.g., EmoSet), JSON files should follow the same hierarchy as the images. Other datasets (e.g., Emotion6) do not include an annotation folder — they only contain train/ and test/ splits.

- Update the root dataset path in config/train.yaml and config/eval.yaml before running training or evaluation.


## 🧪 Training

MASP training consists of two stages. Both stages share the same configuration file — modify dataset paths, hyperparameters, and training options in config/train.yaml before running.

1. Stage 1 — learn query vectors & cross-attention

Trains the query vectors and cross-attention modules to extract view-specific information from images. After training, the learned weights are saved and later loaded during Stage 2.

```bash
python main.py --stage stage1
```


2. Stage 2 — learn soft prompts

Loads the weights from Stage 1 and freezes them. Trains only the soft prompt for emotion prediction. After training, the checkpoint for the soft prompt is saved.

```bash
python main.py --stage stage2
```
All configurations can be modified in config/train.yaml.


## 📈 Evaluation

We provide pretrained checkpoints for simple reproduction of this method: [google drive](https://drive.google.com/drive/folders/1blXWP2I876a3wfLzyEZC_YXsFXWg0uiy?usp=sharing), or you can train the model from scratch.

| Component | File | Notes |
|----------|------|-------|
| Stage 1 — Aspect Module | aspect.pth | Query vectors + cross-attention |
| Stage 2 — Soft Prompt | soft_prompt_emotion6.pt | Trained soft prompt (Emotion6 only) |

After downloading, place them like this:

```
outputs
├── stage1
│   └── aspect.pth
└── stage2
    └── soft_prompt_emotion6.pt
```

Update checkpoint paths in the config:

```yaml
checkpoint:
  ckpt_path: "../outputs/stage1/aspect.pth"
  soft_prompt_path: "../outputs/stage2_train/soft_prompt_emotion6.pt"
```

Run the final evaluation of emotion recognition performance using the command below:

```bash
cd src
python evaluate.py
```

This script loads the trained Stage 2 MASP model and reports accuracy.
All configurations can be adjusted in config/eval.yaml.
