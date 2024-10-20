import re

from simpletransformers.ner import NERModel

# Initialize the NER model with distilbert type
model_directory = "trained_model"
model = NERModel(model_type='distilbert', model_name=model_directory, use_cuda=False)


def highlight_mountain_names(input_sentence, model_prediction):
    """
    Highlights mountain names in the input sentence using the model's predictions.

    Args:
        input_sentence (str): The sentence containing potential mountain names.
        model_prediction (list): The model's output predictions for the sentence.

    Returns:
        str: The input sentence with highlighted mountain names.
    """
    mountains = []  # List to store identified mountain names
    current_mountain = []  # Temporary list to build the current mountain name

    for word in model_prediction[0]:
        word_label = next(iter(word.values()))
        word_key = next(iter(word.keys()))

        # Check for the beginning of a mountain name
        if word_label == 'B-MOUNTAIN':
            if current_mountain:
                # Store the previous mountain name if exists
                mountains.append(' '.join(current_mountain))
                current_mountain = []  # Reset for a new mountain
            current_mountain.append(word_key)  # Add the new mountain name
        elif word_label == 'I-MOUNTAIN' and current_mountain:
            # Continue adding to the current mountain name
            current_mountain.append(word_key)

    # Add the last mountain name if it exists
    if current_mountain:
        mountains.append(' '.join(current_mountain))
        bold_start = '\033[1m'
        bold_end = '\033[0m'

    # Replace mountain names in the input sentence with highlighted versions
    for mountain in mountains:
        input_sentence = re.sub(r'(?<!\w)' + re.escape(mountain) + r'(?!\w)',
                                 f'{bold_start}{mountain}{bold_end}(Mountain)', input_sentence)

    return input_sentence


def predict_and_highlight_mountains(input_sentence):
    """
    Processes the input sentence, predicting mountain names and highlighting them.

    Args:
        input_sentence (str): The input sentence to be processed.

    Returns:
        str: The highlighted sentence with mountain names.
    """
    prediction, model_output = model.predict([input_sentence])

    # Highlight mountain names in the input sentence
    highlighted_sentence = highlight_mountain_names(input_sentence, prediction)
    return highlighted_sentence


def main():
    """Main function to demonstrate mountain name highlighting."""
    input_sentence = (
        """
        Climbing Gasherbrum IV is a challenge even for experienced mountaineers.
        Matterhorn and Mount Ararat offers breathtaking views from its peak.
        The base of Snowdon stretches across several regions.
        The rugged terrain of Mount Sniezka makes it difficult to ascend.
        Local legends surround the history of Mont Maudit.
        """
    )

    # Process the input sentence
    highlighted_sentence = predict_and_highlight_mountains(input_sentence)
    print(highlighted_sentence)


if __name__ == "__main__":
    main()
