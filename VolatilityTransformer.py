from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd

class VolatilityTransformer(BaseEstimator, TransformerMixin):

    def transform(self, X, **transform_params):
        Xdum = pd.DataFrame(((X['high'] - X['low']) / X['close']).values.reshape(-1,1), columns=['volatility'], index=X.index)
        return Xdum

    def fit(self, X, y=None, **fit_params):
        return self
