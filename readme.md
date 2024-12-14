# Sentiment Analysis with RNN, CNN, CNN+LSTM, and DistilBERT

## Introduction

This project explores various deep learning architectures for sentiment analysis on IMDB movie reviews. We compare four models:

1. **Recurrent Neural Network (RNN)** using SimpleRNN
2. **Convolutional Neural Network (CNN)**
3. **CNN + LSTM Hybrid Model**
4. **DistilBERT Transformer Model**

These models have been chosen to showcase the progression from traditional sequential models (RNNs) and local feature extractors (CNNs) to more sophisticated architectures (CNN+LSTM) and transformer-based models (DistilBERT).

## Dataset

The dataset consists of IMDB movie reviews, each labeled as positive (1) or negative (0). The dataset includes approximately **39,723** training samples. Each sample contains raw text and a corresponding binary sentiment label.

**Dataset Columns:**
- `text`: The movie review text.
- `label`: Binary sentiment label (0 = negative, 1 = positive).


## Requirements

- Python 3.7+
- Virtual environment or Conda environment recommended.

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Key Libraries:**
- `numpy` and `pandas` for data manipulation
- `nltk` for natural language processing
- `gensim` for training Word2Vec embeddings
- `tensorflow` and `keras` for deep learning models
- `transformers` (HuggingFace) for DistilBERT
- `scikit-learn` for evaluation metrics
- `matplotlib` and `seaborn` for data visualization

**Additional Setup:**
- Download NLTK stopwords and WordNet lemmatizer:
  ```python
  import nltk
  nltk.download('stopwords')
  nltk.download('wordnet')
  ```

## Preprocessing

1. **Text Cleaning & Normalization:**  
   Remove HTML tags, lowercasing, removing unwanted characters, and handling repetitive characters.

2. **Handling Contractions:**  
   A `contractions.csv` is used to map common English contractions to their expanded forms.

3. **Lemmatization & Stopword Removal:**  
   NLTK’s WordNetLemmatizer and stopword removal are applied to keep only informative tokens.

4. **Tokenization & Padding:**  
   Text is tokenized using `Tokenizer` from `keras.preprocessing.text`. Sequences are padded to a fixed length (e.g., 500 tokens).

5. **Word Embeddings (Word2Vec):**  
   A Word2Vec model is trained on the training set to create embedding vectors, which are then loaded into Keras Embedding layers.

## Models

**1. RNN Model (SimpleRNN):**  
- Embedding layer (pre-trained Word2Vec)
- SimpleRNN layer (100 units, ReLU)
- Dense layers for binary classification

**2. CNN Model:**  
- Embedding layer (pre-trained Word2Vec)
- Conv1D layer (100 filters, kernel size 5)
- GlobalMaxPooling1D
- Dense layers for classification

**3. CNN + LSTM Model:**  
- Embedding layer (pre-trained Word2Vec)
- Two Conv1D + MaxPooling layers
- LSTM layer (64 units)
- Dense layers with dropout and L2 regularization

**4. DistilBERT Model:**  
- Using `DistilBertTokenizer` and `TFDistilBertForSequenceClassification` from HuggingFace Transformers
- Fine-tuned on the IMDB dataset

## Training

**Key Parameters:**
- Optimizer: Adam
- Loss: Binary Crossentropy
- EarlyStopping: Monitors validation accuracy or loss to prevent overfitting
- Epochs: Up to 15 (for RNN/CNN/CNN+LSTM), up to 5 for DistilBERT (to keep training time manageable)


## Evaluation

The models are evaluated on a held-out test set using:
- **Accuracy**
- **F1 score**
- **Confusion Matrix**

**Performance Snapshot:**

| Model     | Accuracy  | F1 Score  |
|-----------|-----------|-----------|
| RNN       | ~85.86%   | ~86.53%   |
| CNN       | ~86.42%   | ~86.50%   |
| CNN+LSTM  | ~85.25%   | ~86.73%   |
| DistilBERT| ~88.95%   | ~89.04%   |

*Note: The provided numbers are illustrative based on the initial project results.*

## Observations

- **RNN:** Excels at sequential data but may struggle with long dependencies.
- **CNN:** Efficient at capturing local patterns and slightly outperforms RNN in our case.
- **CNN+LSTM:** Combines local pattern recognition (CNN) with sequential memory (LSTM). Shows competitive results.
- **DistilBERT:** Achieves the best accuracy and F1 score, leveraging powerful pre-trained contextual embeddings.

## Future Improvements

- **Hyperparameter Tuning:** Further optimization of the number of filters, units, and dropout rates.
- **Advanced Models:** Explore other transformer-based models (e.g., BERT, RoBERTa) or larger architectures.
- **Data Augmentation:** Use techniques like back-translation or synonym replacement to increase data variety.

