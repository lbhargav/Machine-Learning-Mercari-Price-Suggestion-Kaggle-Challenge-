from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

from src.constants import TF_IDF_VECTORIZER, COUNT_VECTORIZER

class PreprocessTextData:
    def __init__(self, df, features_dict):
        """
        df: pandas dataframe
        features_dict: dict with key encoding type and values list
        """
        self.df = df
        self.features_dict = features_dict
        self.output_df = self.df.copy(deep=True)

    #  Count-vectorization
    def _get_count_vectorizer(self):
        """
        Function to get count vectorizer for text data
        """
        cv = CountVectorizer(max_features=2)
        for col in self.features_dict.get(TF_IDF_VECTORIZER):
            self.output_df = pd.concat(
                [self.output_df, pd.DataFrame(cv.fit_transform(self.df[col]).toarray(),
                                              columns=cv.get_feature_names())], axis=1)
        return self.output_df

    #  Tf-idf vectorization
    def _get_tfidf_vectorizer(self):
        """
        Function to get tf-idf matrix for given text data
        """
        tfidf_vectorizer = TfidfVectorizer(max_features=20,
                                           ngram_range=(1, 3),
                                           stop_words='english')
        for col in self.features_dict.get(TF_IDF_VECTORIZER):
            self.output_df = pd.concat([self.output_df, pd.DataFrame(tfidf_vectorizer.fit_transform(self.df[col]).toarray(),
                                          columns=tfidf_vectorizer.get_feature_names())], axis=1)
            self.output_df.head(4)
        return self.output_df

    def text_fit_transform(self):
        for encoding_type in self.features_dict.keys():
            if COUNT_VECTORIZER == encoding_type:
                self.output_df = self._get_count_vectorizer()
            elif TF_IDF_VECTORIZER == encoding_type:
                self.output_df = self._get_tfidf_vectorizer()
            else:
                raise Exception("Encoding type not supported")
        return self.output_df
