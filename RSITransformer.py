from talib import abstract
from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import math

class RSITransformer(BaseEstimator, TransformerMixin):

    def transform(self, X, period_count=14, hot_thresh=30, cold_thresh=70, **transform_params):
        return RSITransformer.analyze_rsi(X, period_count=period_count, hot_thresh=hot_thresh, cold_thresh=cold_thresh)

    def fit(self, X, y=None, **fit_params):
        return self



    def analyze_rsi(historial_data, period_count,
                    hot_thresh, cold_thresh, all_data=True):
        """Performs an RSI analysis on the historical data
        Args:
            historial_data (pandas.Dataframe): A matrix of historical OHCLV data.
            period_count (int, optional): Defaults to 14. The number of data points to consider for
                our simple moving average.
            hot_thresh (float, optional): Defaults to None. The threshold at which this might be
                good to purchase.
            cold_thresh (float, optional): Defaults to None. The threshold at which this might be
                good to sell.
            all_data (bool, optional): Defaults to True. If True, we return the RSI associated
                with each data point in our historical dataset. Otherwise just return the last one.
        Returns:
            dict: A dictionary containing a tuple of indicator values and booleans for buy / sell
                indication.
        """
        dataframe = historial_data
        rsi_values = abstract.RSI(dataframe, period_count)
        #print('Max RSI',rsi_values.max(), 'Min RSI', rsi_values.min())
        rsi_result_data = []
        for rsi_value in rsi_values:

            # if math.isnan(rsi_value):
            #     continue

            is_hot = False
            if hot_thresh is not None:
                is_hot = rsi_value < hot_thresh

            is_cold = False
            if cold_thresh is not None:
                is_cold = rsi_value > cold_thresh

            # construct dict of results
            data_point_result = {
                'rsi_value': rsi_value,
                'rsi_is_cold': is_cold,
                'rsi_is_hot': is_hot
            }
            rsi_result_data.append(data_point_result)

        # convert back to dataframe
        out_df = pd.DataFrame(rsi_result_data, index=dataframe.index)

        if all_data:
            return out_df
        else:
            try:
                return out_df[-1]
            except IndexError:
                return out_df

