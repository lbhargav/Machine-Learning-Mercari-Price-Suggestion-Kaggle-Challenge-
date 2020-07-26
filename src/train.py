import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from src.categorical import CategoricalFeatures
from src.preprocess_text_data import PreprocessTextData

from src.constants import TF_IDF_VECTORIZER

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def split_cat(text):
    try:
        return text.split("/")
    except:
        return ("No Label", "No Label", "No Label")

# handling missing values for brand_name and item_description
def fill_missing_values(df):
    """
    Function to handle missing values
    :param df: dataframe
    """
    df['brand_name'].fillna(value='no brand', inplace=True)
    df['item_description'].fillna(value='no description', inplace=True)
    return df

def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))

if __name__ == "__main__":
    df = pd.read_table("../input/train.tsv", nrows=1000)
    df_test =  pd.read_table("../input/test.tsv", nrows=1000)
    sample = pd.read_csv("../input/sample_submission.csv")
    train_len = len(df)
    # target varible
    Y = np.log1p(df['price'])

    full_data = pd.concat([df, df_test], ignore_index=True)

    # splitting category column into three sub categories
    full_data['general_cat'], full_data['sub_cat1'], full_data['sub_cat2'] = zip(*full_data['category_name']
                                                                                 .apply(lambda x: split_cat(x)))

    fill_missing_values(full_data)
    categorical_features_dict = {"label_encoding": ['brand_name', 'sub_cat1', 'sub_cat2'],
                                 "one_hot_encoding": ['item_condition_id', 'shipping', 'general_cat']}
    label_categorical_features = CategoricalFeatures(full_data, categorical_features_dict=categorical_features_dict)
    data_transformed = label_categorical_features.fit_transform()
    text_features_dict = {"tf_idf_vectorization" : ['item_description', 'name']}
    text_vectorized_fetaures = PreprocessTextData(data_transformed, text_features_dict)
    full_data_transformed = text_vectorized_fetaures.text_fit_transform()

    columns_to_drop = ['category_name', 'train_id'] + categorical_features_dict.get("one_hot_encoding") + text_features_dict.get(TF_IDF_VECTORIZER)
    full_data_transformed.drop(columns_to_drop, axis=1, inplace=True)

    X = full_data_transformed[:train_len]
    X.drop(['price', 'test_id'], axis=1, inplace=True)

    print("predictions on test data")
    X_test = full_data_transformed[train_len:]
    submission: pd.DataFrame = X_test[['test_id']]
    X_test.drop(['price', 'test_id'], axis=1, inplace=True)

    train_sparse_matrix, valid_sparse_matrix, y_train, y_valid = train_test_split(X, Y, test_size=0.1, random_state=144)
    d_train = lgb.Dataset(train_sparse_matrix, label=y_train)

    params = {
        'learning_rate': 0.15,
        'application': 'regression',
        'max_depth': 13,
        'num_leaves': 400,
        'verbosity': -1,
        'metric': 'RMSE',
        'data_random_seed': 1,
        'bagging_fraction': 0.8,
        'feature_fraction': 0.6,
        'nthread': 4,
        'lambda_l1': 10,
        'lambda_l2': 10
    }
    classifier = lgb.train(params, train_set=d_train, num_boost_round=3200, verbose_eval=100)
    predictions = classifier.predict(valid_sparse_matrix)

    pred = []
    for prediction in predictions:
        if prediction < 0:
            prediction = 0
        pred.append(float(prediction))
    predNP = np.array(pred)

    print(rmsle(y_valid, predNP))