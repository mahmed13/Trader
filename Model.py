# Create a pipeline that extracts features from the data then creates a model
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin, BaseEstimator
from VolatilityTransformer import VolatilityTransformer
from SMATransformer import SMATransformer
from EMATransformer import EMATransformer
from DatesTransformer import DatesTransformer
from TrimDataTransformer import TrimDataTransformer
from LabelTransformer import LabelTransformer
from PeakTransformer import PeakTransformer
from RSITransformer import RSITransformer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from NonPricingData import NonPricingData

import matplotlib.pyplot as plt


import os
import glob
import time
import pandas as pd
import numpy as np


class Model:
    def run(self):
        DATA_DIR = "data_modified"

        # creates data directory
        if not os.path.exists(DATA_DIR):
            os.mkdir(DATA_DIR)

        dataFilenames = glob.glob('data/USDT_' + '*.csv')
        n_periods = [1, 2, 6, 12, 36, 72, 144]
        # read data from file
        for per in n_periods[0:1]:
            for fileName in dataFilenames[1:2]:
                try:
                    #print('Reading data from file',fileName)
                    df = pd.read_csv(fileName).iloc[-100000:] # modeling starts; 6mo = 52,416 periods
                except:
                    print('Failed opening', fileName)
                    continue

                df.set_index('date', inplace=True)

                print('pair=',df['pair_name'].iloc[0])
                print('n_periods=',per)


                # Pass in candlestick data to RSI Transformer
                df_rsi = RSITransformer().transform(df[['open','high','low','close']], period_count=10)

                #PeakTransformer().transform(df.iloc[-300:], feature='close')

                # feature engineer
                df_npd = NonPricingData().daily_trends('bitcoin')['bitcoin']
                print(df_npd.tail(10).index.dtypes)
                return 0
                df_sma = SMATransformer().transform(df, windows=[3, 5, 10, 20, 50, 100, 200], feature='close')
                df_ema = EMATransformer().transform(df, windows=[3, 5, 10, 20, 50, 100, 200], feature='close')
                df_vol = VolatilityTransformer().transform(df)
                df_dates = DatesTransformer().transform(df)

                df = pd.concat([df, df_rsi], axis=1)
                df = pd.concat([df, df_sma], axis=1)
                df = pd.concat([df, df_ema], axis=1)
                df = pd.concat([df, df_vol], axis=1)
                #df = pd.concat([df, df_dates], axis=1)

                print(df.groupby('rsi_is_cold')['rsi_is_cold'].count())
                show_plots = True

                if show_plots:
                    fig, axes = plt.subplots(2, 1, sharex=True)
                    ax1, ax2 = axes[0], axes[1]
                    df['close'].plot(ax=ax1, linewidth=0.25)
                    df['rsi_value'].plot(ax=ax2, linewidth=0.25)
                    ax2.axhline(30, lw=0.25, color='g', linestyle='dashed')
                    ax2.axhline(70, lw=0.25, color='r', linestyle='dashed')
                    plt.show()

                # trim
                df = df.iloc[-30000:]
                df.dropna(inplace=True)
                print(df.head(10))
                X, y = LabelTransformer().transform(df=df, label_feature='close', periods=per)


                # create feature union
                features = []

                #features.append(('standardize', StandardScaler()))
                #features.append(('pca', PCA()))
                #features.append(('kbest', SelectKBest(k=8)))

                #feature_union = FeatureUnion(features)


                # create pipeline
                estimators = []
                #estimators.append(('feature_union', feature_union))
                #estimators.append(('pca', PCA()))
                #estimators.append(('decision', DecisionTreeClassifier(criterion = "gini", max_depth=3, min_samples_leaf=5)))
                estimators.append(('logistic', LogisticRegression()))

                model = Pipeline(estimators)

                from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

                parameters = {'logistic__C': [20, 3]}
                #parameters = {}

                seed = 1
                np.random.seed(seed)

                tscv_n_splits = 4
                tscv = TimeSeriesSplit(n_splits=tscv_n_splits)
                clf = GridSearchCV(model, param_grid=parameters, n_jobs=1, cv=tscv, return_train_score=True)
                print('training')
                clf.fit(X, y)

                print('Best param=',clf.best_params_)
                print('Best avg score=',clf.best_score_)
                print('nth train score=', max(clf.cv_results_[('split%d_train_score' % (tscv_n_splits-1))]),
                      '\nnth test score=', max(clf.cv_results_[('split%d_test_score' % (tscv_n_splits-1))]))
                print('More details...\n',clf.cv_results_)



                # # evaluate pipeline
                # seed = 7
                # kfold = KFold(n_splits=10, random_state=seed)
                # results = cross_val_score(model, X, y, cv=kfold)
                # print(results.mean())
                # print(results)


def main():
    Model().run()

if __name__ == '__main__':
    main()