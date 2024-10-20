import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from simpletransformers.ner import NERModel, NERArgs


def load_dataset(file_path):
    """Load the dataset from a CSV file and return as a DataFrame."""
    try:
        data = pd.read_csv(file_path, sep='\t', encoding='latin1', header=None, on_bad_lines='skip')
        data.columns = ['sentence_id', 'word', 'label']
        return data
    except Exception as e:
        raise IOError(f"Error loading data from {file_path}: {e}")

def split_and_prepare_data(data):
    """Prepare training and test datasets."""

    # Check if required columns are present
    if not all(col in data.columns for col in ['sentence_id', 'word']):
        raise KeyError("Missing required columns in DataFrame: 'sentence_id' and/or 'word'.")

    print(f"DataFrame shape: {data.shape}")
    print(f"DataFrame columns: {data.columns.tolist()}")

    x = data[["sentence_id", "word"]]
    y = data["label"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

    train_data = pd.DataFrame({
        "sentence_id": x_train["sentence_id"],
        "words": x_train["word"],
        "labels": y_train
    })

    test_data = pd.DataFrame({
        "sentence_id": x_test["sentence_id"],
        "words": x_test["word"],
        "labels": y_test
    })

    return train_data, test_data


def initialize_ner_model(labels):
    """Create and return a NER model using DistilBERT."""

    args = NERArgs(
        num_train_epochs=10,
        learning_rate=2e-4,
        overwrite_output_dir=True,
        train_batch_size=16,
        eval_batch_size=16,
        output_dir='trained_model'
    )

    return NERModel('distilbert', 'distilbert-base-cased', labels=labels, args=args, use_cuda=False)

def train_and_evaluate_ner_model(model, train_data, test_data):
    """Train and evaluate the NER model."""
    model.train_model(train_data, eval_data=test_data, acc=accuracy_score)
    result, model_outputs, preds_list = model.eval_model(test_data)
    return result


def main():
    """Main function to load data, prepare datasets, create model, and evaluate it."""
    file_path = "https://raw.githubusercontent.com/HannaVasiunyk/NER_model_mountains/refs/heads/main/mountain_label_dataset.csv"

    # Load data
    data = load_dataset(file_path)

    # Prepare training and test data
    train_data, test_data = split_and_prepare_data(data)

    # Create model
    dataset_labels = data["label"].unique().tolist()
    model = initialize_ner_model(dataset_labels)

    # Train and evaluate the model
    evaluation_result = train_and_evaluate_ner_model(model, train_data, test_data)

    print("Model evaluation:")
    print(evaluation_result)


if __name__ == "__main__":
    main()