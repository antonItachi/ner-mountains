# Mountain Name Recognition with Neural Networks

## Overview

This project focuses on recognizing mountain names in text using Named Entity Recognition (NER) techniques and neural networks. The goal is to extract mountain names from unstructured text data.

## Data Collection

1. Data was collected by web scraping websites containing information about mountains.
2. The collected text data was preprocessed to extract and annotate mountain names.

## Model Training

1. The BERT (Bidirectional Encoder Representations from Transformers) model was used for token classification.
2. The dataset was tokenized, and labels were aligned with tokens.
3. The model was fine-tuned on the annotated dataset.

## Model Inference

1. A trained model was used for inference.
2. The model predicted mountain names in input text.

## Improvements

1. To improve accuracy, diversify the dataset with more mountain-related sentences.
2. Experiment with different NER models and hyperparameters.
3. Explore techniques for handling variations in mountain name formats.

## Usage

To perform inference with the trained model, use the provided Python script.

```bash
python model_training.py
python model_inference.py
```

## Conclusion

This project demonstrates the process of recognizing mountain names in text using NER and neural networks. Further improvements can enhance the accuracy and robustness of the model.

Please refer to the documentation for detailed usage instructions.
