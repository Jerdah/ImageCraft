
# **ImageCraft: Direct Image-to-Speech Synthesis**





## **Overview**
ImageCraft is a deep learning project designed to generate spoken descriptions directly from images. The goal is to create a model that combines vision and text-to-speech capabilities for accessibility tools, multimedia storytelling, and human-computer interaction. It utilizes a Vision Transformer (ViT) for image encoding, a custom GPT model for text generation, and VoiceCraft for converting text into speech.

## **Table of Contents**
1. [Project Objectives](#project-objectives)
2. [Directory Structure](#directory-structure)
3. [Dataset](#dataset)
4. [Model Architecture](#model-architecture)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Training and Evaluation](#training-and-evaluation)
8. [Deployment](#deployment)
9. [Testing](#testing)
10. [Results and Visualization](#results-and-visualization)
12. [Future Work](#future-work)
15. [References](#references)

## **Project Objectives**
The primary objectives of ImageCraft are:
- To create a multimodal pipeline that converts input images into meaningful spoken descriptions.
- To utilize transformer-based models, specifically a Vision Transformer (ViT) as an image encoder and a custom GPT decoder to generate text descriptions.
- To facilitate real-time audio captioning of images for accessibility use cases.

## **Directory Structure**
The primary objectives of ImageCraft are:

```css
ImageCraft/
│
├── training_data/
│   ├── dataset/
│   │   ├── flickr30k/
│   │   └── mscoco/
│   ├── checkpoints/
│   └── pretrained_models/
│
├── VoiceCraft/
│   ├── models/
│   ├── data/
│   └── inference_tts_scale.py
│
├── notebooks/
├── src/
│   ├── data_preparation.py
│   ├── train.py
│   ├── inference.py
│   └── utils.py
│
└── requirements.txt
```
## **Dataset**
### **Flickr30k and MSCOCO**
The Flickr30k dataset is used for training and evaluation. It contains paired image-caption data, making it suitable for the image-to-speech task.

- **Download and Preparation**: The datasets are downloaded and organized into relevant folders for training (`/training_data/dataset/flickr30k` and `/training_data/dataset/mscoco`). During preparation, images are resized, and captions are tokenized using a custom tokenizer that adds special tokens like `[BOS]` (beginning of sequence) and `[EOS]` (end of sequence).

## **Model Architecture**
ImageCraft consists of three major components:
1. **Vision Transformer (ViT)**: Encodes images into feature vectors. The ViT is pre-trained, with selective layers fine-tuned during training.
2. **Custom GPT Decoder**: Generates text captions from the image features. This architecture consists of multiple transformer-based blocks similar to GPT.
3. **VoiceCraft Module**: Converts generated text into speech using a text-to-speech synthesis model.

## **Installation**
To set up the environment and install the necessary dependencies, follow the steps below:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Jerdah/ImageCraft.git
   cd ImageCraft
   ```
   
2.**Install System-Level Dependencies**:
```bash
apt-get install -y espeak-ng espeak espeak-data libespeak1 libespeak-dev festival* build-essential flac libasound2-dev libsndfile1-dev vorbis-tools libxml2-dev libxslt-dev zlib1g-dev
```

3. **Install Python Libraries**:
```bash
pip install -r requirements.txt
```

4. **Download Dataset from Kaggle**: 
kaggle datasets download -d hsankesara/flickr-image-dataset
kaggle datasets download -d mnassrib/ms-coco

### +**Installation Instructions Details**
Ensure you have all dependencies installed with specific versions:

- Python >= 3.8
- torch==2.0.1
- transformers==4.27.1
- gradio==3.0
  
If you encounter installation errors, refer to the `requirements.txt` or contact us for help.

## **Usage**
### **Inference**

You can use the provided Gradio interface or run the inference script to generate speech from an image.

#### **Using Gradio**:

```python

def imagecraft_interface(image_path, reference_text):
  """Process image inputs and generate audio response."""
  transcript, audio_buffer = model.generate(image_path, output_type="buffer")

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

# Define Gradio interface
gradio_interface = gr.Interface(
  fn=imagecraft_interface,
  inputs=[
    gr.Image(type="filepath", label="Upload an image"),
    gr.Textbox(label="Reference Text (for evaluation)")
  ],
  outputs=[
    gr.Audio(label="Speech"),
    gr.Textbox(label="Transcript"),
    gr.Textbox(label="Evaluation Results")
  ],
  title="ImageCraft",
  description="Upload an image and get the speech responses.",
  allow_flagging="never"
)

# Launch the Gradio app
gradio_interface.launch(debug=False)
```

#### **Using CLI**:

```bash
# run inference and return the audio file path
python -m src.model.inference --image_path "media/images/1.jpeg" --output_type "file"
```

## **Training and Evaluation**

### **Training**
The training pipeline uses the following setup:

- **Freezing Strategy**: Initially, only the GPT decoder is trained while the ViT encoder remains frozen. Later epochs unfreeze the ViT for end-to-end fine-tuning.
- **Metrics**: Training loss and test loss are monitored along with perplexity, which measures the quality of text predictions.

To train the model from scratch:

```python
#train the model
python -m src.model.train --dataset "flickr" --dataset_size "5%" --batch_size 2 --max_epochs 2
```

### **Evaluation Metrics**
The following metrics are used to evaluate model performance:

**Training Loss**: Measures the model's performance on the training set.
**Test Loss**: Measures the generalization ability on unseen data.
**Perplexity**: Evaluates how well the model predicts the sequence.

### **TensorBoard**:
Training metrics are logged to TensorBoard for easy visualization:

```bash
tensorboard --logdir runs
```

## **Deployment**

The model can be deployed using the REST API provided by Flask. Additionally, the model can be containerized using Docker for reproducibility and easy deployment on cloud platforms.

### **Run API**:
```bash
python app.py
```

Navigate to `http://localhost:5000` to use the web interface.

## **Testing**

There are no specific unit tests implemented in the code for different functions. Implementing unit tests with a framework like `pytest` is recommended for:

- **Data Preprocessing**: Validate transformations and tokenization.
- **Model Forward Passes**: Ensure that both ViT and GPT modules work as expected.
To add unit tests, consider creating a `tests/` directory with the following:

`test_data_preparation.py`
`test_model_forward.py`

## **Results and Visualization**

- **Training Curves**: Loss and perplexity are plotted using matplotlib after each epoch to visualize performance.
- **Generated Samples**: Audio samples from the model are saved and can be played back to evaluate the quality of speech generation.

  ### **Gradio demo app**

![alt text](https://github.com/Jerdah/ImageCraft/blob/main/reports/figures/gradio_app_demo.png)

## **Future Work**

- **Real-Time Processing**: Optimize the model for real-time inference on edge devices.
- **Improvement in Text Generation**: Integrate semantic analysis to enhance caption quality.

## **References**

- **VoiceCraft**: The VoiceCraft text-to-speech module used in this project is based on the repository provided by Facebook Research. For more details, visit the [VoiceCraft GitHub](https://github.com/jasonppy/VoiceCraft) repository.
- **Vision Transformer (ViT)**: The Vision Transformer architecture is inspired by "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Dosovitskiy et al. (2020). [Paper link](https://arxiv.org/abs/2010.11929)

## **Acknowledgments**

- Thanks to [nsandiman](https://github.com/nsandiman), [ravinamore-ml](https://github.com/ravinamore-ml), [Masengug](https://github.com/Masengug) and [Jerdah](https://github.com/Jerdah)
