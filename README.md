# MASP: Multi-Aspect Soft Prompt Tuning for Emotion Reasoning in Vision-Language Models
 

## 🔍 Overview

Understanding human emotions from an image is a challenging yet essential task for vision-language models. While recent efforts have fine-tuned vision-language models to enhance emotional awareness, most approaches rely on global visual representations and fail to capture the nuanced, multi-faceted nature of emotional cues. Furthermore, most existing approaches adopt instruction tuning, which requires costly dataset construction and involves training a large number of parameters, thereby limiting scalability and efficiency. To address these challenges, we propose MASP, a novel framework for Multi-Aspect guided emotion reasoning with Soft Prompt tuning in vision-language models. MASP explicitly separates emotion-relevant visual cues via multi-aspect cross-attention modules and guides the language model using soft prompts, enabling efficient and scalable task adaptation without modifying the base model. Our method achieves state-of-the-art performance on various emotion recognition benchmarks, demonstrating that explicit modeling of multi-aspect emotional cues with soft prompt tuning leads to more accurate and interpretable emotion reasoning in vision-language models.

<p align="center">
  <img src="assets/masp_architecture.jpg" alt="MASP Architecture" width="900">
</p> 

## 📁 Project Structure
```
MASP/
├── config.yaml # Configuration file
├── data_utils.py # Data loading and preprocessing utilities
├── evaluate.py # Evaluation script
├── prompt_utils.py # Prompt generation and processing utilities
├── prompts.txt # List of emotion prompts
├── requirements.txt # Python package dependencies
├── softprompt_model.py # MASP model and soft prompt architecture
├── train.py # Training script
└── README.md
```


## 📦 Setup 
```bash
git clone https://github.com/{id}/masp.git
cd masp
pip install -r requirements.txt
```

## 🧪 Training
You can start training MASP using the following command:

```bash
python train.py \ 
  --your_datapath /path/to/your/dataset \
  --dataname EmoSet \
  --save_ckpt /path/to/save/softprompt_ckpt.pth
```


## 📈 Evaluation
```bash 
python evaluate.py \ 
  --your_datapath /path/to/your/dataset \
  --dataname EmoSet \
  --load_ckpt /path/to/your/softprompt_ckpt.pth \
  --load_spmodel_ckpt /path/to/your/model_ckpt.pth \
```

