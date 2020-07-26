import pandas as pd

from sklearn.preprocessing import LabelEncoder
from src.constants import LABEL_ENCODING, ONE_HOT_ENCODING

class CategoricalFeatures:
    def __init__(self, df, categorical_features_dict):
        """
        df: pandas dataframe
        categorical_features_dict: dict with key encoding type and values list
        of columns
        """
        self.df = df
        self.categorical_features_dict = categorical_features_dict
        # check for null values
        for col in self.categorical_features_dict.values():
            if self.df[col].isnull().values.any():
                raise Exception("Dataframe has null values column {}".format(col))
        self.output_df = self.df.copy(deep=True)

    def _label_encoding(self):
        """
        Function to performs one hot encoding
        """
        global lb_encoder
        lb_encoder = LabelEncoder()
        for col in self.categorical_features_dict.get(LABEL_ENCODING):
            self.output_df[col] = lb_encoder.fit_transform(self.df[col])
        return self.output_df

    def _one_hot_encoding(self):
        """
        Function to performs one hot encoding
        """
        for col in self.categorical_features_dict.get(ONE_HOT_ENCODING):
            self.output_df = pd.concat([self.output_df, pd.get_dummies(self.df[col], prefix=str(col))], axis=1)
        return self.output_df

    def fit_transform(self):
        for encoding_type in self.categorical_features_dict.keys():
            if LABEL_ENCODING == encoding_type:
                self.output_df = self._label_encoding()
            elif ONE_HOT_ENCODING == encoding_type:
                self.output_df = self._one_hot_encoding()
            else:
                raise Exception("Encoding type not supported")
        return self.output_df



