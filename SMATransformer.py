from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import pandas as pd

class SMATransformer(BaseEstimator, TransformerMixin):

    def transform(self, X, windows = [3, 5, 10, 20, 50], feature = 'close', **transform_params):
        Xdum = pd.DataFrame(index=X.index)
        for window in windows:
            Xdum[('sma_%d' % window)] = SMATransformer.moving_average(self, X, window, feature)
            # replacing first window*288 number of values becuase theyre all 0
            Xdum[('sma_%d' % window)] = np.where(Xdum[('sma_%d' % window)] == 0, X[feature], Xdum[('sma_%d' % window)])

        return Xdum

    def fit(self, X, y=None, **fit_params):
        return self

    def moving_average(self, df, window, feature):
        window *= 288 # 5 min periods -> 1 day
        values = np.array(df[feature])
        weights = np.repeat(1.0, window)/window
        smas = np.convolve(values, weights, 'valid')
        smas = np.concatenate((np.zeros(shape=(window-1,)), smas))
        return smas


