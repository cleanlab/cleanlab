import pandas as pd
import spacy
import torch
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import pandas as pd
import spacy
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.utils import simple_preprocess


class Coherence:
    def __init__(
        self, bert_model: str = "bert-base-uncased", spacy_model: str = "en_core_web_sm"
    ) -> None:
        # Load spaCy for sentence tokenization
        self.__spacy_model = spacy_model
        try:
            self.nlp = spacy.load(self.__spacy_model)
        except OSError:
            print("Downloading spaCy 'en_core_web_sm' model...")
            from spacy.cli import download

            download(self.__spacy_model)
            nlp = spacy.load("en_core_web_sm")
        # Load BERT model and tokenizer
        self.__bert_model = bert_model
        self.tokenizer = BertTokenizer.from_pretrained(self.__bert_model)
        self.model = BertModel.from_pretrained(self.__bert_model)

    def get_sentence_embeddings(self, sentences: list) -> np.ndarray:
        """Generate BERT embeddings for a list of sentences."""
        embeddings = []
        for sentence in sentences:
            # Tokenize and encode the sentence
            inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
            outputs = self.model(**inputs)
            # Extract the CLS token as the sentence embedding
            cls_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
            embeddings.append(cls_embedding.flatten())
        return np.array(embeddings)

    def score(self, text: str) -> float:
        """Calculate the coherence score of a single text based on sentence similarity."""
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]

        # Generate embeddings for each sentence
        sentence_embeddings = self.get_sentence_embeddings(sentences)

        # Calculate pairwise cosine similarities between consecutive sentences
        coherence_scores = []
        for i in range(len(sentence_embeddings) - 1):
            similarity = cosine_similarity([sentence_embeddings[i]], [sentence_embeddings[i + 1]])[
                0
            ][0]
            coherence_scores.append(similarity)

        # Return the average coherence score for the text
        return np.mean(coherence_scores) if coherence_scores else 0.0

    def assess_dataframe(self, dataframe: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Evaluate coherence for each text in the DataFrame."""
        coherence_scores = []
        for text in dataframe[text_column]:
            coherence_score = self.score(text)
            coherence_scores.append(coherence_score)

        # Add coherence scores as a new column
        dataframe["coherence_score"] = coherence_scores
        return dataframe


class GensimCoherence:
    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_sm")

    def preprocess_text(self, text):
        """Tokenize text into sentences and preprocess each sentence."""
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        tokenized_sentences = [simple_preprocess(sentence) for sentence in sentences]
        return tokenized_sentences

    def score(self, text):
        """Calculate coherence score for tokenized sentences."""
        tokenized_sentences = self.preprocess_text(text)
        dictionary = Dictionary(tokenized_sentences)

        coherence_model = CoherenceModel(
            topics=tokenized_sentences,
            texts=tokenized_sentences,
            dictionary=dictionary,
            coherence="c_v",
        )
        coherence_score = coherence_model.get_coherence()
        return coherence_score

    def assess_dataframe(self, dataframe: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Evaluate coherence for each text in the DataFrame."""
        coherence_scores = []
        for text in dataframe[text_column]:
            tokenized_sentences = self.preprocess_text(text)
            coherence_score = self.score(tokenized_sentences)
            coherence_scores.append(coherence_score)

        # Add coherence scores as a new column
        dataframe["coherence_score"] = coherence_scores
        return dataframe
