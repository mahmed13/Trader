import time
import json
import numpy as np
import pandas as pd
from pytrends.request import TrendReq
from datetime import datetime, timedelta
from blockchain import util
from pytrends.request import TrendReq


class NonPricingData(): # Update needed: add update feature similar to poloniexdata.py

    # Generates google trend data with day granularity given a single search term.
    def daily_trends(self, search_term):
        ## PARAMS ##
        # The maximum for a timeframe for which we get daily data is 270.
        # Therefore we could go back 269 days. However, since there might
        # be issues when rescaling, e.g. zero entries, we should have an
        # overlap that does not consist of only one period. Therefore,
        # I limit the step size to 250. This leaves 19 periods for overlap.
        maxstep = 269
        overlap = 40
        step = maxstep - overlap + 1
        kw_list = [search_term]
        start_date = datetime(2013, 12, 9).date()
        #start_date = datetime(2016, 3, 9).date()

        ## FIRST RUN ##
        # Run the first time (if we want to start from today, otherwise we need to ask for an end_date as well
        today = datetime.today().date()
        old_date = today
        # Go back in time
        new_date = today - timedelta(days=step)

        # request trend data
        interest_over_time_df = NonPricingData().TrendDataRequest(new_date=new_date,old_date=old_date,kw_list=kw_list)

        ## RUN ITERATIONS ##
        while new_date > start_date:

            ### Save the new date from the previous iteration.
            # Overlap == 1 would mean that we start where we
            # stopped on the iteration before, which gives us
            # indeed overlap == 1.
            old_date = new_date + timedelta(days=overlap - 1)

            ### Update the new date to take a step into the past
            # Since the timeframe that we can apply for daily data
            # is limited, we use step = maxstep - overlap instead of
            # maxstep.
            new_date = new_date - timedelta(days=step)

            # If we went past our start_date, use it instead
            if new_date < start_date:
                new_date = start_date

            # request trend data
            temp_df = NonPricingData().TrendDataRequest(new_date=new_date, old_date=old_date, kw_list=kw_list)

            # check request contents
            if (temp_df.empty):
                raise ValueError(
                    'Google sent back an empty dataframe. Possibly there were no searches at all during the this period! '
                    'Set start_date to a later date.')

            # Renormalize and concate
            interest_over_time_df = NonPricingData().renormalize(kw_list=kw_list,new_date=new_date,
                                                                 old_date=old_date, overlap=overlap,
                                                                 original_df=interest_over_time_df,
                                                                 prepend_df=temp_df)

        # timestamp indecies -> unix; Create new date column for testing
        interest_over_time_df['date'] = interest_over_time_df.index
        interest_over_time_df.index = pd.Series(convert_timestamp_to_unix_time(interest_over_time_df[search_term])).astype(np.int64)

        interest_over_time_df = NonPricingData().spread_period(interest_over_time_df, search_term)

        return interest_over_time_df

    # (helper function) handles making Google Trends API calls
    def TrendDataRequest(self, new_date, old_date, kw_list):
        # trend object
        pytrend = TrendReq()

        # Create new timeframe for which we download data
        timeframe = new_date.strftime('%Y-%m-%d') + ' ' + old_date.strftime('%Y-%m-%d')

        # API Call -- Download Trend Data
        pytrend.build_payload(kw_list=kw_list, timeframe=timeframe)

        # return df
        return pytrend.interest_over_time()

    # (helper function) renormalization using scalining and concatination
    def renormalize(self, kw_list, new_date, old_date, overlap, original_df, prepend_df):
        # Renormalize the dataset and drop last line
        for kw in kw_list:
            beg = new_date
            end = old_date - timedelta(days=1)

            # Since we might encounter zeros, we loop over the
            # overlap until we find a non-zero element
            for t in range(1, overlap + 1):
                if prepend_df[kw].iloc[-t] != 0:
                    # theses two elements in the df represent the same day, different sample for normalization
                    scaling = original_df[kw].iloc[t - 1] / prepend_df[kw].iloc[-t]
                    break
                elif t == overlap:
                    print('Did not find non-zero overlap, set scaling to zero! Increase Overlap!')
                    scaling = 0

            # Apply scaling
            prepend_df.loc[beg:end, kw] = prepend_df.loc[beg:end, kw] * scaling
        interest_over_time_df = pd.concat([prepend_df[:-overlap], original_df])

        return interest_over_time_df

    # (helper function) copys all daily trend data to period, Try making smoother lines between days
    def spread_period(self, df, feature):    # needs work..
        df = df[feature]
        PERIODS_IN_A_DAY = 288
        start_date = df.index.values[0]
        end_date = df.index.values[-1]

        new_index = list(range(start_date, end_date+300*300, 300))

        df = pd.concat([df]*PERIODS_IN_A_DAY)
        df.sort_index(inplace=True)

        lol = df.values.tolist()

        # fix
        new_index = new_index[0:len(df)]

        df = pd.DataFrame(lol, columns=[feature+'_trend'], index=new_index)
        return df

    def hash_rate(self):
        response = util.call_api('charts/hash-rate?timespan=2years&format=json', base_url='https://api.blockchain.info/')
        hash_json = json.loads(response)

        times = np.zeros(len(hash_json['values']))
        hash_rates = np.zeros(len(hash_json['values']))
        for i in range(len(hash_json['values'])):
            times[i] = hash_json['values'][i]['x']
            hash_rates[i] = hash_json['values'][i]['y']
        times = list(map(int, times))
        hash_rates = list(map(int, hash_rates))
        df = pd.DataFrame(np.column_stack([times, hash_rates]), columns=['time', 'hash_rate'])
        df = df.set_index('time')

        df = NonPricingData().spread_period(df, 'hash_rate')

        return df

# (helper function)
def convert_timestamp_to_unix_time(timestamps):
    unix_times = []
    for i in range(len(timestamps.index)):
        unix_times.append(time.mktime(list(timestamps.index)[i].timetuple()))

    return unix_times


if __name__ == '__main__':
    print('NonPricingData [Testing mode]')
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')

    # testing
    search_terms = ['fortnite','Lamar Odom', 'Bitcoin']
    search_terms = ['bitcoin']

    df = NonPricingData().hash_rate()


    for search_term in search_terms:
        #df = NonPricingData().daily_trends(search_term)
        print(df.tail())
        #NonPricingData().spread_period(df)

        if True:
            # plot
            plt.plot(df)
            plt.xticks(rotation=20)
            plt.ylabel('Google Search Trends: \''+search_term+'\'')
            plt.show()

