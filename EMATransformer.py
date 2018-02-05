from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import pandas as pd

class EMATransformer(BaseEstimator, TransformerMixin):

    def transform(self, X, windows = [3, 5, 10, 20, 50], feature = 'close' ,**transform_params):

        Xdum = pd.DataFrame(index=X.index)
        for window in windows:
            Xdum[('ema_%d' % window)] = EMATransformer.exp_moving_average(self, X, window, feature)
            # replacing first window*288 number of values becuase theyre all 0
            Xdum[('ema_%d' % window)] = np.where(Xdum[('ema_%d' % window)] == 0, X[feature], Xdum[('ema_%d' % window)])
        return Xdum

    def fit(self, X, y=None, **fit_params):
        return self

    def exp_moving_average(self, df, window, feature):
        window *= 288 # 5 min periods -> 1 day
        values = np.array(df[feature])

        weights = np.exp(np.linspace(-1.,0.,window))
        weights /= weights.sum()

        a = np.convolve(values, weights)[:len(values)]
        a[:window] = a[window]
        return a

