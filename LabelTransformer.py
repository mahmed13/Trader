from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import numpy as np

class LabelTransformer(BaseEstimator, TransformerMixin):

    def transform(self, df, label_feature='close', periods=1, **transform_params):
        # label
        df['temp'] = df[label_feature].pct_change(periods)
        df['label'] = np.where(df['temp'] > 0, 1, 0)
        df['label'] = df['label'].shift(-periods)

        # split X, y
        y = df['label']
        df.drop(['temp', 'label', 'pair_name'], axis=1, inplace=True)
        X = df

        # remove nan values
        X = X[:-periods]
        y = y[:-periods]

        return X, y

    def fit(self, X, y=None, **fit_params):
        return self
