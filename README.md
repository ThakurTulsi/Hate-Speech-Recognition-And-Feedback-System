# Hate Speech Recognition

This project implements a machine learning model to classify text data as hate speech, offensive speech, or neutral. It preprocesses the text data, tokenizes the inputs, converts them into numerical representations, and trains a model to categorize the input comments accordingly. The project is built using Python libraries like Pandas, NumPy, and Scikit-Learn.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to identify and flag comments that contain hate speech or offensive language. It is a critical step toward creating safer online environments, by automatically detecting and addressing toxic or harmful content.

### Features:
- Cleaned and preprocessed text data.
- Tokenization and word vectorization using CountVectorizer.
- Model training and classification using machine learning algorithms.
- User-friendly prediction interface that classifies input text into three categories:
  - Hate Speech
  - Offensive Speech
  - Neither Hate nor Offensive

## Dataset

The dataset used consists of user comments, where each comment is labeled as:
- **Hate Speech**
- **Offensive Speech**
- **Neither Hate nor Offensive**
The dataset is downloaded from Kaggle and also attached in the files section of the repository. (Link: https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset?resource=download)

Text preprocessing includes:
- Lowercasing
- Removing punctuation
- Tokenizing the text
- Stopwords removal
- Lemmatization

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hate-speech-recognition.git
   ```
   
2. Navigate to the project directory:
   ```bash
   cd hate-speech-recognition
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Open the Jupyter Notebook `Hate Speech Recognition.ipynb`.
   
2. Run all the cells to load the dataset, preprocess the text, and train the model.

3. To make predictions:
   - Enter your comment in the provided input field.
   - The model will classify the input as either hate speech, offensive speech, or neutral.

4. Example output:
   ```
   You typed: Science is amazing!
   Category: Neither Hate nor Offensive
   Good Job! Spread kindness over hate or offensive messages.
   ```

## Model

The project uses **CountVectorizer** to convert text into numerical features and a machine learning algorithm (likely Logistic Regression or similar) to classify the text.

Steps involved:
1. Text preprocessing
2. Tokenization
3. Word vectorization
4. Model training
5. Model evaluation (accuracy, confusion matrix, etc.)

## Results

The model achieves reasonable accuracy in classifying text as hate, offensive, or neutral. Example metrics could include:
- Accuracy: `XX%`
- F1 Score: `XX%`
- Confusion Matrix: Visualized in the notebook.

## Contributing

Contributions, issues, and feature requests are welcome!

## License

This project is licensed under the MIT License.
