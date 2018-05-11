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
from sklearn import preprocessing
import warnings
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
from xgboost import XGBClassifier


import matplotlib.pyplot as plt


import os
import glob
import time
import pandas as pd
import numpy as np


class Model:
    def run(self):
        PERIOD_LENGTH =  5 # minutes
        DATA_DIR = "data_5"

        # creates data directory
        if not os.path.exists(DATA_DIR):
            os.mkdir(DATA_DIR)

        dataFilenames = glob.glob(DATA_DIR+'/USDT_' + '*.csv')
        n_periods = [1, 2, 6, 12, 36, 72, 144, 288, 432, 288*2]
        #n_periods = [12, 36, 72, 144, 288, 432, 288*2]
        #n_periods = [144*i for i in range(1,6)]
        # read data from file
        for per in n_periods:
            for fileName in dataFilenames[1:2]:
                try:
                    #print('Reading data from file',fileName)
                    df = pd.read_csv(fileName)
                    print(len(df))
                    df= df.iloc[-100000:] # modeling starts; 6mo = 52,416 periods
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
                df_npd = NonPricingData().daily_trends('bitcoin', period_length=PERIOD_LENGTH)
                #df_hash = NonPricingData().hash_rate()
                df_sma = SMATransformer().transform(df, windows=[1, 3, 5, 10, 20, 50, 100], feature='close')
                df_ema = EMATransformer().transform(df, windows=[1, 3, 5, 10, 20, 50, 100], feature='close')
                df_vol = VolatilityTransformer().transform(df)
                df_dates = DatesTransformer().transform(df)

                df = pd.concat([df, df_rsi], axis=1)
                df = pd.concat([df, df_sma], axis=1)
                df = pd.concat([df, df_ema], axis=1)
                df = pd.concat([df, df_vol], axis=1)
                df = pd.concat([df, df_dates], axis=1)
                df = pd.merge(df, df_npd, left_index=True, right_index=True)
                #df = pd.merge(df, df_hash, left_index=True, right_index=True)
                #print(df.groupby('rsi_is_cold')['rsi_is_cold'].count())
                show_plots = False

                if show_plots:
                    fig, axes = plt.subplots(2, 1, sharex=True)
                    ax1, ax2 = axes[0], axes[1]
                    df['close'].plot(ax=ax1, linewidth=0.25)
                    df['rsi_value'].plot(ax=ax2, linewidth=0.25)
                    ax2.axhline(30, lw=0.25, color='g', linestyle='dashed')
                    ax2.axhline(70, lw=0.25, color='r', linestyle='dashed')
                    plt.show()

                # trim
                print(len(df))
                df = df.iloc[-30000:]
                print(len(df))
                df.dropna(inplace=True)
                df.drop('volume', 1, inplace = True)
                df.drop('quoteVolume', 1, inplace = True)
                print(list(df.columns.values))
                X, y = LabelTransformer().transform(df=df, label_feature='close', periods=per)


                # create feature union
                # features = []
                # features.append(('standardize', StandardScaler()))
                # features.append(('pca', PCA()))
                # features.append(('kbest', SelectKBest(k=16)))
                #
                # feature_union = FeatureUnion(features)


                # create pipeline
                estimators = []
                #estimators.append(('feature_union', feature_union))
                #estimators.append(('pca', PCA()))
                #estimators.append(('decision', DecisionTreeClassifier(criterion = "gini", max_depth=3, min_samples_leaf=5)))
                #estimators.append(('logistic', LogisticRegression()))
                estimators.append(('xgb', XGBClassifier()))
                model = Pipeline(estimators)

                from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

                parameters = {'logistic__C': [20, 7, 3, 0.1, 0.01]}
                parameters = {}

                seed = 1
                np.random.seed(seed)

                tscv_n_splits = 4
                tscv = TimeSeriesSplit(n_splits=tscv_n_splits)
                clf = GridSearchCV(model, param_grid=parameters, n_jobs=1, cv=tscv, return_train_score=True)
                print('Training')
                clf.fit(X, y)

                print("Label distribution=", sum(y)/len(y),"/",1-sum(y)/len(y))
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



if __name__ == '__main__':
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    Model().run()
