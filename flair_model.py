from email.utils import parsedate_to_datetime
from flair.datasets import ColumnCorpus
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.data import Corpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings, TransformerWordEmbeddings
from flair.trainers import ModelTrainer

import pandas as pd
import os


class FlairModel:
    def __init__(self, hidden_size=256, name=''):
        self.hidden_size = hidden_size
        self.name = name

    def train(
        self,
        train,
        test,
        dev,
        x_name='message_body_raw',
        y_name='message_body_clean',
        epochs=10
    ):
        os.makedirs(f"resources/data/{self.name}", exist_ok=True)

        self._generate_flair_txt(
            train, f'resources/data/{self.name}/train.txt', x_name, y_name)
        self._generate_flair_txt(
            test, f'resources/data/{self.name}/test.txt', x_name, y_name)
        self._generate_flair_txt(
            dev, f'resources/data/{self.name}/dev.txt', x_name, y_name)

        columns = {0: 'text', 1: 'signal'}
        corpus: Corpus = ColumnCorpus(f'resources/data/{self.name}/', columns)

        tag_type = 'signal'
        tag_dictionary = corpus.make_label_dictionary(label_type=tag_type)

        embedding_types = [
            WordEmbeddings('glove'),
            FlairEmbeddings('news-forward-fast'),
            FlairEmbeddings('news-backward-fast'),
        ]
        embeddings = StackedEmbeddings(embeddings=embedding_types)

        # embeddings = TransformerWordEmbeddings(model='distilbert-base-uncased',
        #                                        layers="-1",
        #                                        subtoken_pooling="first",
        #                                        fine_tune=True,
        #                                        use_context=True,
        #                                        )

        tagger = SequenceTagger(hidden_size=self.hidden_size,
                                embeddings=embeddings,
                                tag_dictionary=tag_dictionary,
                                tag_type=tag_type)

        trainer = ModelTrainer(tagger, corpus)

        trainer.train('resources/taggers/' + self.name,
                      train_with_dev=True,
                      max_epochs=epochs)

    def _generateFlairString(self, raw, clean):
        counter = 0
        signal = []
        string = ''
        for word in raw.rstrip().split():
            if word == clean.split()[counter]:
                if counter + 1 < len(clean.split()):
                    counter += 1
                    signal.append(word)
                else:
                    signal.append(word)
                    counter = 0
                    string += f"{signal[0]}\tB-Key\n"
                    for s in signal[1:]:
                        string += f"{s}\tI-Key\n"
            else:
                string += f"{word}\tO\n"
        return string + '\n'

    def _generate_flair_txt(self, df, path, x_name, y_name):
        """
        Generates the text file necessary for training a flair model.

        Args:
            df: contains message_body_raw, message_body_clean columns
            path: the path in which to store the txt file

        Returns:
            None
        """

        f = open(path, 'w')
        f.write('')
        f.close()
        f = open(path, 'a')

        flair_strings = df.apply(lambda x: self._generateFlairString(
            x[x_name], x[y_name]), axis=1)

        for string in flair_strings.values:
            f.write(string)
        f.close()

    def predict(self, sentences):
        try:
            tagger = SequenceTagger.load(
                f"resources/taggers/{self.name}/final-model.pt")
        except:
            print("No model found")
            return None

        sentence_list = [Sentence(sentence_str)
                         for sentence_str in sentences]
        tagger.predict(sentence_list)

        results = []

        for sentence in sentence_list:
            parsed_sentence = " ".join(
                [e.text for e in sentence.get_spans('signal')])
            if len(parsed_sentence) > 0:
                results.append(parsed_sentence)
            else:
                results.append(sentence.to_plain_string())

        return results
