# **ImageCraft: Direct Image-to-Speech Synthesis**

## **Overview**

ImageCraft is a deep learning project designed to generate spoken descriptions directly from images. The goal is to create a model that combines vision and text-to-speech capabilities for accessibility tools, multimedia storytelling, and human-computer interaction. It utilizes a vision transformer (SigLIP) for image encoding, Gemma for text decoding, and VoiceCraft for speech synthesis.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bzmNvc-XM9RPbkZEYFdap-nNJkrCvfzu#scrollTo=-SoOHUJHsfTD) [![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/nsandiman/uarizona-msis-capstone-group5-imagecraft)

![alt text](https://github.com/Jerdah/ImageCraft/blob/main/reports/figures/imagecraft-arch.jpeg)

## **Table of Contents**

1. [Project Objectives](#project-objectives)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Training](#training)
7. [Future Work](#future-work)
8. [References](#references)

## **Project Objectives**

The primary objectives of ImageCraft are:

- To create a multimodal pipeline that converts input images into meaningful spoken descriptions.
- To utilize transformer-based models, specifically a vision transformer (SigLIP) as an image encoder and a Gemma decoder.
- To facilitate image-to-speech for accessibility use cases.

## **Dataset**

### **MSCOCO**

The MSCOCO dataset is used for training and evaluation. It contains paired image-caption data, making it suitable for the image-to-speech task.

- **Download and Preparation**: The datasets are downloaded and organized into relevant folders for training (`data/processed/coco/train` and evaluation `data/processed/coco/test`).

**Download dataset for training**:

```bash
python -m src.data.download --dataset "coco" --dataset_size "10%"
```

## **Model Architecture**

ImageCraft consists of three major components:

1. **Vision Transformer (SigLIP)**: Calculates the image embeddings.
2. **Gemma Decoder**: Decodes text from the image features.
3. **VoiceCraft token infilling neural codec language model**: The speech synthesis model.

## **Installation**

To set up the environment and install the necessary dependencies, follow the steps below:

1. **Clone the Repository**:

```bash
git clone https://github.com/Jerdah/ImageCraft.git
cd ImageCraft
```

2. **Install System-Level Dependencies**:

```bash
apt-get install -y espeak-ng espeak espeak-data libespeak1 libespeak-dev festival* build-essential flac libasound2-dev libsndfile1-dev vorbis-tools libxml2-dev libxslt-dev zlib1g-dev
```

3. **Install Python libraries**:

```bash
pip install -r requirements.txt
```

4. **Metrics for evaluating automated image descriptions using tools such as SPICE, PTBTokenizer, METEOR and FENSE**:

```bash
aac-metrics-download
```

## **Usage**

### **Inference**

You can use the provided Gradio interface or run the inference script to generate speech from an image.

#### **Using Gradio (basic interface)**:

```python

import sys
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["USER"] = "imagecraft"

import gradio as gr
from src.model.modules.imagecraft import ImageCraft


model = ImageCraft.from_pretrained("nsandiman/imagecraft-ft-co-224")

default_image = "media/images/2.jpeg"
def generate(image_path):
    """Process image inputs and generate audio response."""
    transcript, audio_buffer = model.generate(image_path, output_type="buffer")

    return audio_buffer, transcript


imagecraft_app = gr.Interface(
    fn=generate,
    inputs=[
        gr.Image(
            type="filepath",
            label="Upload an image",
            sources=["upload"],
            value=default_image,
        ),
    ],
    outputs=[gr.Audio(label="Speech"), gr.Textbox(label="Text")],
    title="ImageCraft",
    description="Upload an image and get the speech responses.",
    allow_flagging="never",
)

imagecraft_app.launch()
```

![alt text](https://github.com/Jerdah/ImageCraft/blob/main/reports/figures/imagecraft-basic-ui.png)

#### **Using Gradio (evaluation interface)**:

```python


import sys
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["USER"] = "imagecraft"

import gradio as gr
from bert_score import score
import evaluate
from src.model.modules.imagecraft import ImageCraft


bertscore_metric = evaluate.load("bertscore")
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load('rouge')


default_image = "media/images/2.jpeg"
def imagecraft_interface(image_path, reference_text):
  """Process image inputs and generate audio response."""
  transcript, audio_buffer = model.generate(image_path, output_type="buffer")

  if not reference_text:
    evaluation_result = "No reference text provided for evaluation."
  else:
    reference_text = reference_text.strip().lower().rstrip('.')
    transcript = transcript.strip().lower().rstrip('.')

    bert_score_result = calculate_bert_score(reference_text, transcript)
    bleu_score_result = calculate_bleu_score(reference_text, transcript)
    rouge_score_result = calculate_rouge_score(reference_text, transcript)

    evaluation_result = f"BERT Score: {bert_score_result:.4f}\nBLEU Score: {bleu_score_result:.4f}\nROUGE Score: {rouge_score_result:.4f}"


  return audio_buffer, transcript, evaluation_result

def calculate_bert_score(reference, hypothesis):
  scores = bertscore_metric.compute(predictions=[hypothesis], references=[reference], lang="en")
  f1 = scores["f1"][0]
  return f1

def calculate_bleu_score(reference, hypothesis):
  results = bleu_metric.compute(predictions=[hypothesis], references=[[reference]])
  bleu = results["bleu"]
  return bleu

def calculate_rouge_score(reference, hypothesis):
  results = rouge_metric.compute(predictions=[hypothesis], references=[[reference]])
  return results["rougeL"]

imagecraft_app = gr.Interface(
  fn=imagecraft_interface,
  inputs=[
    gr.Image(
            type="filepath",
            label="Upload an image",
            sources=["upload"],
            value=default_image,
        ),
    gr.Textbox(label="Reference Text (for evaluation)")
  ],
  outputs=[
    gr.Audio(label="Speech"),
    gr.Textbox(label="Text"),
    gr.Textbox(label="Evaluation Results")
  ],
  title="ImageCraft",
  description="Upload an image and get the speech responses.",
  allow_flagging="never"
)

imagecraft_app.launch()

```

![alt text](https://github.com/Jerdah/ImageCraft/blob/main/reports/figures/imagecraft-evaluation-ui.png)

#### **Using CLI**:

```bash
# run inference and return the audio file path
python -m src.model.inference --image_path "media/images/1.jpeg" --output_type "file"
```

## **Training**

Specify training log collector:

- To use tensorboard, add the argument tensorboard to the command line.
- To use wandb, add the argument wandb to the command line.

Download the dataset (if it doesn't exist) and train the model.

```python

python -m src.model.train --dataset "coco" --dataset_size "20%" --batch_size 2 --max_epochs 10 --log_every_n_steps 2 --log_to "wandb"
```

## **Future Work**

- **Real-Time Processing**: Optimize the model for real-time inference on edge devices.
- **Improvement in Text Generation**: Integrate semantic analysis to enhance caption quality.

## **References**

- **VoiceCraft**: The VoiceCraft text-to-speech module used in this project is based on the repository provided by Facebook Research. For more details, visit the [VoiceCraft GitHub](https://github.com/jasonppy/VoiceCraft) repository.
- **Vision Transformer (SigLIP)**: The Vision Transformer architecture is inspired by "Sigmoid Loss for Language Image Pre-Training" by Zhai et al. (2023). [Paper link](https://arxiv.org/abs/2303.15343)

## **License**

This codebase is under CC BY-NC-SA 4.0 ([LICENSE-CODE](./LICENSE-CODE)). Note that we use some of the code from other repository that are under different licenses: ./src/model/modules/voicecraft.py is under CC BY-NC-SA 4.0; ./src/model/modules/codebooks_patterns.py is under MIT license; ./src/model/modules/tokenizer.py are under Apache License, Version 2.0; the phonemizer we used is under GNU 3.0 License.

## **Acknowledgments**

- Thanks to [nsandiman](https://github.com/nsandiman), [ravinamore-ml](https://github.com/ravinamore-ml), [Masengug](https://github.com/Masengug) and [Jerdah](https://github.com/Jerdah)
- We thank Umar Jamil for his work [pytorch-paligemma](https://github.com/hkproj/pytorch-paligemma), from where we took lot of inspiration.
