import pandas as pd
import re
from rouge_score import rouge_scorer

from tqdm import tqdm 

class Filter:
    def __init__(self, filter_function, vectorized=False, noise = "") -> None:
        self.filter_function = filter_function
        self.vectorized = vectorized
        self.noise = noise

    def filter(self, cleaned):
        if not self.vectorized:
            cleaned = cleaned.apply(self.filter_function)
        else:
            cleaned = self.filter_function(cleaned)
        return cleaned


class Cleaner:
    def __init__(self) -> None:
        self.filters = []

    def set_filters(self, filters):
        self.filters = filters

    def add_filters(self, filters):
        self.filters.extend(filters)

    def add_filter(self, filter):
        self.add_filters([filter])

    def clean(self, df, column='message_body_raw') -> pd.Series:
        """
        returns pd.Series of cleaned messages.
        """
        cleaned = df[column].copy()
        for filter in self.filters:
            messages_for_filter = df[df[filter.noise]]["cleaned"] if filter.noise != "all" else cleaned
            cleaned.loc[messages_for_filter.index] = filter.filter(messages_for_filter)    
        return cleaned

    def test(self, df, raw_column='message_body_raw', clean_column='message_body_clean'):
        df = df.reset_index()

        df['cleaned'] = df[raw_column]
        df['cleaned'] = self.clean(df, raw_column)

        df[clean_column] = df[clean_column].apply(
            lambda x: ' '.join(re.findall(r'\w+|[^\s\w]+', x)))

        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = df[[clean_column, 'cleaned']].apply(
            lambda x: scorer.score(x[clean_column], x['cleaned']), axis=1)

        df['f1_score'] = scores.apply(lambda x: x['rougeL'].fmeasure)
        df['recall'] = scores.apply(lambda x: x['rougeL'].recall)

        return df, df['f1_score'].mean()
