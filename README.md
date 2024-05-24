# Text Classification using RNN and Word Embeddings
 sentiment analysis on Amazon reviews using deep learning models (simpleRNN and LSTM). Preprocess the dataset, split it for training and validation, and build a vocabulary for word embedding. The project includes a real-time prediction feature for user-inputted reviews and provides a report detailing model summaries and best hyperparameters. Ideal for NLP enthusiasts exploring text classification techniques.

## Introduction

Sentiment analysis is a common task in natural language processing (NLP) that aims to determine the sentiment or opinion expressed in a piece of text. This project leverages deep learning techniques, specifically RNN and LSTM models, to classify the sentiment of Amazon reviews into positive, neutral, or negative categories.

## Features

- **Data Preprocessing:** Includes cleaning and preprocessing text data.
- **Tokenization and Padding:** Converts text data into a numerical format suitable for model training.
- **Model Creation:** Defines and initializes RNN and LSTM models.
- **Training and Validation:** Implements training loops with performance tracking.
- **GUI Application:** Provides a simple GUI for sentiment prediction using trained models.
- **Visualization:** Plots training and validation losses.

## Usage

1. **Data Preprocessing:** Preprocess the dataset to clean and tokenize the text data.
2. **Model Training:** Train the RNN and LSTM models on the preprocessed data.
3. **Sentiment Prediction:** Use the trained models to predict the sentiment of new reviews via a GUI application.

## Dataset

The project uses the Amazon reviews dataset, which contains customer reviews and ratings. The dataset includes the following columns:
- `sentiments`: The sentiment of the review (positive, neutral, negative).
- `cleaned_review`: The preprocessed review text.
- `cleaned_review_length`: The length of the cleaned review.
- `review_score`: The review score given by the customer.

## Model Training

The project defines and trains two types of models:
- **RNN Model:** A simple Recurrent Neural Network for sentiment classification.
- **LSTM Model:** A Long Short-Term Memory network, which is more effective for capturing long-term dependencies in text data.

Both models are trained using the Adam optimizer and cross-entropy loss function.

## GUI for Sentiment Prediction

A simple GUI application is provided for predicting the sentiment of new reviews. The user can input a review, select the model (RNN or LSTM), and get the predicted sentiment.

## Results

### RNN Model

- **Training Loss:** Shows a decreasing trend over epochs, indicating good learning progress.
- **Validation Loss:** Fluctuates but generally decreases, suggesting the model is learning but experiencing some overfitting.
- **Validation Accuracy:** Stabilizes around 86%, demonstrating reasonable performance for sentiment classification.

**RNN Model Performance:**

| Epoch | Train Loss | Val Loss | Val Accuracy |
|-------|------------|----------|--------------|
| 1     | 0.6734     | 0.5767   | 77.85%       |
| 2     | 0.5406     | 0.5487   | 79.70%       |
| 3     | 0.4409     | 0.5355   | 80.48%       |
| ...   | ...        | ...      | ...          |
| 20    | 0.0597     | 0.7657   | 86.08%       |

![RNN Performance](https://github.com/ahmedm-sallam/Text-Classification-using-RNN-and-Word-Embeddings/assets/97572668/409e2604-408f-4070-9ba6-0f0667e38e83)


### LSTM Model

- **Training Loss:** Decreases steadily, indicating effective learning.
- **Validation Loss:** Generally decreases, showing good generalization with slight overfitting.
- **Validation Accuracy:** Peaks around 89%, indicating superior performance over the RNN model.

**LSTM Model Performance:**

| Epoch | Train Loss | Val Loss | Val Accuracy |
|-------|------------|----------|--------------|
| 1     | 0.5736     | 0.4878   | 79.76%       |
| 2     | 0.3643     | 0.3995   | 85.21%       |
| 3     | 0.2299     | 0.4260   | 85.88%       |
| ...   | ...        | ...      | ...          |
| 20    | 0.0117     | 0.7123   | 88.72%       |

![LSTM Performance](https://github.com/ahmedm-sallam/Text-Classification-using-RNN-and-Word-Embeddings/assets/97572668/a20001aa-cd99-452e-afa1-9357b4f624b1)

