import re
import string
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

from cleanlab.classification import CleanLearning


def main():
    data = pd.read_csv("./banking-intent-classification.csv")

    raw_texts, raw_labels = data["text"].values, data["label"].values

    raw_train_texts, raw_test_texts, raw_train_labels, raw_test_labels = train_test_split(
        raw_texts, raw_labels, test_size=0.1
    )
    num_classes = len(set(raw_train_labels))

    encoder = LabelEncoder()
    encoder.fit(raw_train_labels)

    train_labels = encoder.transform(raw_train_labels)
    test_labels = encoder.transform(raw_test_labels)

    transformer = SentenceTransformer("google/electra-small-discriminator")

    train_texts = transformer.encode(raw_train_texts)
    test_texts = transformer.encode(raw_test_texts)

    model = LogisticRegression(max_iter=400)

    cv_n_folds = 5
    cl = CleanLearning(model, cv_n_folds=cv_n_folds)

    label_issues = cl.find_label_issues(X=train_texts, labels=train_labels)

    print(label_issues.head())


if __name__ == "__main__":
    main()
