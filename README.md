# Spoken Question Answering Model

## Brief Description

This project implements a question-answering model for spoken documents using the BERT architecture. The model is trained on the Spoken-SQuAD dataset, which consists of spoken documents and text-based questions. The goal is to extract answers from the spoken transcripts based on given questions.

## Key Features

- Fine-tuning BERT for extractive question answering
- Custom dataset implementation for Spoken-SQuAD
- Data augmentation techniques
- Automatic mixed precision training
- Gradient accumulation for effective larger batch sizes
- Linear learning rate decay
- Evaluation using F1 score

## Requirements

- Python 3.x
- PyTorch
- Transformers library
- tqdm
- scikit-learn
- matplotlib

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Mahmoud118/spoken-qa-model.git
   ```

2. Install the required packages:
 
## Usage

1. Prepare the Spoken-SQuAD dataset and place it in the `Spoken-SQuAD` directory.

2. Run the main script:
   ```
   python main.py
   ```

3. The script will train the model and save the best model weights to `best_model.pth`.

4. To evaluate the model on the test set, use the `evaluate_on_test_set` function provided in the code.

## Model Architecture

The model uses the BERT-base-uncased architecture fine-tuned for question answering. It employs a custom dataset class `SpokenSQuADDataset` that handles the processing of spoken documents and questions.

## Training

The training process includes:
- Data augmentation
- Custom collate function for batching
- Automatic mixed precision training
- Gradient accumulation
- Linear learning rate decay
- Validation after each epoch

## Evaluation

The model is evaluated using the F1 score, which balances precision and recall. The evaluation is performed on a separate test set to assess the model's generalization capabilities.
