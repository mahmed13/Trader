from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd

class TrimDataTransformer(BaseEstimator, TransformerMixin):

    def transform(self, X,**transform_params):
        return X

    def fit(self, X, y=None, **fit_params):
        return self
