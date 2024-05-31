import json
import random

import nltk
from tqdm import tqdm


def load_inlp_data(persistent_dir, bias_type, seed=0):
    """Loads sentences used to train INLP classifiers.

    Args:
        persistent_dir (`str`): Directory where all data is stored.
        bias_type (`str`): The bias type to generate the dataset for.
            Must be either gender, nationality, or religion.
    """
    random.seed(seed)

    if bias_type == "gender":
        data = _load_gender_data(persistent_dir)
    elif bias_type == "nationality":
        data = _load_nationality_data(persistent_dir)
    else:
        data = _load_religion_data(persistent_dir)
    return data


def _load_gender_data(persistent_dir):

    male_sentences = []
    female_sentences = []

    male_sentences_clipped = []
    female_sentences_clipped = []
    neutral_sentences_clipped = []

    count_male_sentences = 0
    count_female_sentences = 0
    count_neutral_sentences = 0

    with open(f"{persistent_dir}/data/text/gender_male.txt", "r") as f:
        lines = f.readlines()
    random.shuffle(lines)

    for line in tqdm(lines, desc="Loading INLP data"):
        # Each line contains a paragraph of text.
        sent, idx = line.split('\t')
        sentences = nltk.sent_tokenize(sent)

        for sentence in sentences:

            tokens = sentence.split(" ")

            # Convert tokens to lower case.
            tokens = [token.lower() for token in tokens]

            # Skip sentences that are too short.
            if len(tokens) < 5:
                continue

            male_sentences.append(sentence)
            index = random.randint(idx, len(tokens))
            male_sentences_clipped.append(" ".join(tokens[: index + 1]))
            count_male_sentences += 1


    with open(f"{persistent_dir}/data/text/gender_female.txt", "r") as f:
        lines = f.readlines()
    random.shuffle(lines)

    for line in tqdm(lines, desc="Loading INLP data"):
        # Each line contains a paragraph of text.
        sent, idx = line.split('\t')
        sentences = nltk.sent_tokenize(sent)

        for sentence in sentences:

            tokens = sentence.split(" ")

            # Convert tokens to lower case.
            tokens = [token.lower() for token in tokens]

            # Skip sentences that are too short.
            if len(tokens) < 5:
                continue

            female_sentences.append(sentence)
            index = random.randint(idx, len(tokens))
            female_sentences_clipped.append(" ".join(tokens[: index + 1]))
            count_female_sentences += 1
            

    with open(f"{persistent_dir}/data/text/gender_neutral.txt", "r") as f:
        lines = f.readlines()
    random.shuffle(lines)

    for line in tqdm(lines, desc="Loading INLP data"):
        # Each line contains a paragraph of text.
        sent, idx = line.split('\t')
        sentences = nltk.sent_tokenize(sent)

        for sentence in sentences:

            tokens = sentence.split(" ")

            # Convert tokens to lower case.
            tokens = [token.lower() for token in tokens]

            # Skip sentences that are too short.
            if len(tokens) < 5:
                continue

                # Start from the fourth token.
            index = random.randint(4, len(tokens))
            neutral_sentences_clipped.append(" ".join(tokens[:index]))
            count_neutral_sentences += 1

    print("INLP dataset collected:")
    print(f" - Num. male sentences: {count_male_sentences}")
    print(f" - Num. female sentences: {count_female_sentences}")
    print(f" - Num. neutral sentences: {count_neutral_sentences}")

    data = {
        "male": male_sentences_clipped,
        "female": female_sentences_clipped,
        "neutral": neutral_sentences_clipped,
    }

    return data


def _load_nationality_data(persistent_dir):

    nationality_sentences = []
    nationality_sentences_clipped = []
    neutral_sentences_clipped = []

    count_nationality_sentences = 0
    count_neutral_sentences = 0

    with open(f"{persistent_dir}/data/text/nationality_biased.txt", "r") as f:
        lines = f.readlines()
    random.shuffle(lines)

    for line in tqdm(lines, desc="Loading INLP data"):
        # Each line contains a paragraph of text.
        sent, idx = line.split('\t')
        sentences = nltk.sent_tokenize(sent)

        for sentence in sentences:

            tokens = sentence.split(" ")

            # Convert tokens to lower case.
            tokens = [token.lower() for token in tokens]

            # Skip sentences that are too short.
            if len(tokens) < 5:
                continue

            nationality_sentences.append(sentence)
            index = random.randint(idx, len(tokens))
            nationality_sentences_clipped.append(" ".join(tokens[: index + 1]))
            count_nationality_sentences += 1

    with open(f"{persistent_dir}/data/text/nationality_neutral.txt", "r") as f:
        lines = f.readlines()
    random.shuffle(lines)

    for line in tqdm(lines, desc="Loading INLP data"):
        # Each line contains a paragraph of text.
        sentences = nltk.sent_tokenize(line)

        for sentence in sentences:

            tokens = sentence.split(" ")

            # Convert tokens to lower case.
            tokens = [token.lower() for token in tokens]

            # Skip sentences that are too short.
            if len(tokens) < 5:
                continue

            index = random.randint(4, len(tokens))
            neutral_sentences_clipped.append(" ".join(tokens[:index]))
            count_neutral_sentences += 1
            continue

    print("INLP dataset collected:")
    print(f" - Num. bias sentences: {count_nationality_sentences}")
    print(f" - Num. neutral sentences: {count_neutral_sentences}")

    data = {"bias": nationality_sentences_clipped, "neutral": neutral_sentences_clipped}

    return data


def _load_religion_data(persistent_dir):

    religion_sentences = []
    religion_sentences_clipped = []
    neutral_sentences_clipped = []

    count_religion_sentences = 0
    count_neutral_sentences = 0

    with open(f"{persistent_dir}/data/text/religion_biased.txt", "r") as f:
        lines = f.readlines()
    random.shuffle(lines)

    for line in tqdm(lines, desc="Loading INLP data"):
        sent, idx = line.split('\t')
        sentences = nltk.sent_tokenize(sent)

        for sentence in sentences:
            tokens = sentence.split(" ")

            # Convert tokens to lower case.
            tokens = [token.lower() for token in tokens]

            # Skip sentences that are too short.
            if len(tokens) < 5:
                continue

            religion_sentences.append(sentence)
            index = random.randint(idx, len(tokens))
            religion_sentences_clipped.append(" ".join(tokens[: index + 1]))
            count_religion_sentences += 1

    with open(f"{persistent_dir}/data/text/religion_neutral.txt", "r") as f:
        lines = f.readlines()
    random.shuffle(lines)

    for line in tqdm(lines, desc="Loading INLP data"):
        # Each line contains a paragraph of text.
        sentences = nltk.sent_tokenize(line)

        for sentence in sentences:
            tokens = sentence.split(" ")

            # Convert tokens to lower case.
            tokens = [token.lower() for token in tokens]

            # Skip sentences that are too short.
            if len(tokens) < 5:
                continue

            index = random.randint(4, len(tokens))
            neutral_sentences_clipped.append(" ".join(tokens[:index]))
            count_neutral_sentences += 1
            continue

    print("INLP dataset collected:")
    print(f" - Num. bias sentences: {count_religion_sentences}")
    print(f" - Num. neutral sentences: {count_neutral_sentences}")

    data = {"bias": religion_sentences_clipped, "neutral": neutral_sentences_clipped}

    return data
