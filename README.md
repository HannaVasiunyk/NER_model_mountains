# Mountain NER (DistilBERT model)

## Overview

The **Mountain NER** is a Python-based project designed to generate sentences that reference various mountains. Each
generated sentence includes a mountain name labeled in the BIO (Begin, Inside, Outside) format, suitable for training
Named Entity Recognition (NER) models in Natural Language Processing (NLP) applications.

## Features

- **Random Sentence Generation**: Generates sentences containing mountain names from a predefined list.
- **BIO Labeling**: Labels mountain names in sentences using the BIO format, aiding in entity recognition tasks.
- **Output Files**:
    - `mountain_sentences.txt`: Contains the generated sentences.
    - `mountain_label_dataset.csv`: Contains the labeled dataset in a structured format.

## Project Structure

- `dataset_creation.ipynb`: Jupyter Notebook file that contains the code for generating a dataset of sentences featuring
  mountain names. It utilizes random sampling to create unique sentences and applies the BIO (Begin, Inside, Outside)
  labeling format for Named Entity Recognition (NER) tasks.
- `model_training.py`: Python script designed to train a Named Entity Recognition (NER) model using the DistilBERT
  architecture. The model is trained on a labeled dataset generated from the `dataset_creation.ipynb` notebook.
- `model_inference.py`: Python script designed for performing inference using a pre-trained Named Entity Recognition (
  NER) model based on the DistilBERT architecture. This script takes input sentences, processes them, and utilizes the
  trained model to identify and label occurrences of mountain names within the text. It serves as a practical tool for
  applying the trained NER model in real-world scenarios, allowing users to recognize mountain names in various
  contexts.
- `ner_model_demo.ipynb`: Jupyter Notebook file serves as an interactive demonstration of the Named Entity Recognition (
  NER) capabilities of a trained DistilBERT model. This notebook allows users to input sentences and visualize the
  model's ability to recognize and label mountain names, showcasing the effectiveness of the model in a user-friendly
  environment.

## Requirements

To run this notebook and the associated project files, you'll need to install the necessary dependencies listed in the
`requirements.txt` file. The file includes the following libraries:

- pandas===2.2.3
- numpy===1.24.3
- jupyter===1.1.1
- torch===2.2.2
- scikit-learn===1.5.2
- simpletransformers===0.70.1

## How to Run the Project

1. **Clone the Repository**
    - Start by cloning the project repository to your local machine:
      ```bash
      git clone <repository_url>
      cd <repository_directory>
      ```

2. **Set Up a Virtual Environment (Optional)**
    - It's recommended to create a virtual environment to avoid conflicts with other projects:
      ```bash
      python -m venv venv
      source venv/bin/activate  # On Windows use `venv\Scripts\activate`
      ```

3. **Install Dependencies**
    - Install the required libraries from the `requirements.txt` file:
      ```bash
      pip install -r requirements.txt
      ```
4. **Run the dataset creation notebook:**

- Open dataset_creation.ipynb in Jupyter Notebook and execute the cells to create the dataset.

5. **Train the model:**

- Run model_training.py in your Python environment to train the model on the generated dataset.

6. **Check the inference script:**

- Run model_inference.py to load the trained model and see its predictions on new data.
  
7. **Check the demo notebook:**

- Open ner_model_demo.ipynb in Jupyter Notebook to review the compiled demo of the NER model.