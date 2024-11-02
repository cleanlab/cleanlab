import pandas as pd
from spellchecker import SpellChecker

class SpellingChecker:
    def __init__(self, weight) -> None:
        self.spell_checker = SpellChecker()
        self.__weight = weight

    def score(self, text: str) -> float:
        """Check the spelling of the words in the given text and return a polar spelling score."""
        words = text.split()
        total_words = len(words)

        if total_words == 0:
            return 1.0  # No words, perfect score

        # Count correctly spelled words
        correctly_spelled = sum(1 for word in words if word in self.spell_checker)

        spelling_score = (correctly_spelled / total_words) ** self.__weight

        return spelling_score

    def assess_dataframe(self, dataframe: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Evaluate spelling for each text in the DataFrame."""
        spelling_scores = []
        for text in dataframe[text_column]:
            spelling_score = self.score(text)
            spelling_scores.append(spelling_score)

        # Add spelling scores as a new column
        dataframe['spelling_score'] = spelling_scores
        return dataframe
