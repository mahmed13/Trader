from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import pandas as pd
import datetime as datetime
from sklearn.preprocessing import OneHotEncoder
import holidays


class DatesTransformer(BaseEstimator, TransformerMixin):

    def transform(self, X, **transform_params):
        #output_df = DatesTransformer.dates(self, X)
        output_df = DatesTransformer().holidays(X)
        return output_df

    def fit(self, X, y=None, **fit_params):
        return self

    def label_days(self, day):
        if day <= 10:
            return 1
        if day > 10 and day <= 20:
            return 2
        if day > 20:
            return 3

    def holidays(self, X):
        output_df = pd.DataFrame(list(map(DatesTransformer.isHoliday, X.index.values)), columns=['isHoliday'], index=X.index)
        return output_df

    def isHoliday(date):
        if int(date) in holidays.UnitedStates():
            return True
        return False

    def dates(self, df):
        # output df - only contains features made in this method
        output_df = pd.DataFrame(index=df.index)

        # day of week
        days = lambda x: datetime.datetime.fromtimestamp(x).weekday()
        enc = OneHotEncoder()
        mapped_days = np.array(df.index.map(days), dtype=pd.Series).reshape(-1, 1)
        one_hot = enc.fit_transform(mapped_days)
        one_hot = np.array(one_hot.toarray())

        # padding
        mapped_days = [item for sublist in mapped_days for item in sublist]
        missing_days = list({0, 1, 2, 3, 4, 5, 6} - set(mapped_days))
        missing_days.sort()
        for i in missing_days:
            one_hot = np.insert(one_hot, i - 1, np.zeros((1, one_hot.shape[0])), 1)

        output_df[['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']] = pd.DataFrame(one_hot,
                                                                                                        dtype=np.int32,
                                                                                                        index = df.index)
        # month
        months = lambda x: datetime.datetime.fromtimestamp(x).month
        enc = OneHotEncoder()
        mapped_months = np.array(df.index.map(months), dtype=pd.Series).reshape(-1, 1)
        one_hot = enc.fit_transform(mapped_months)
        one_hot = np.array(one_hot.toarray())

        # padding
        mapped_months = [item for sublist in mapped_months for item in sublist]
        missing_months = list({1,2,3,4,5,6,7,8,9,10,11,12} - set(mapped_months))
        missing_months.sort()
        for i in missing_months:
            one_hot = np.insert(one_hot,i-1,np.zeros((1, one_hot.shape[0])), 1)


        output_df[['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november',
            'december']] = pd.DataFrame(np.array(one_hot),
                                        dtype=np.int32,
                                        index=df.index)

        # phase of month - note: empirically a very weak indicator
        month_phases = lambda x: DatesTransformer.label_days(self, datetime.datetime.fromtimestamp(x).day)

        enc = OneHotEncoder()
        mapped_phases = np.array(df.index.map(month_phases), dtype=pd.Series).reshape(-1, 1)
        one_hot = enc.fit_transform(mapped_phases)
        one_hot = np.array(one_hot.toarray())

        # padding
        mapped_phases = [item for sublist in mapped_phases for item in sublist]
        missing_phases = list({1, 2, 3} - set(mapped_phases))
        missing_phases.sort()
        for i in missing_phases:
            one_hot = np.insert(one_hot, i - 1, np.zeros((1, one_hot.shape[0])), 1)

        output_df[['beg_month', 'mid_month', 'end_month']] = pd.DataFrame(one_hot, dtype=np.int32,index=df.index)

        return output_df

if __name__ == '__main__':
    pass