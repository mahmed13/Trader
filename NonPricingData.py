import time
import json
import numpy as np
import pandas as pd
from pytrends.request import TrendReq
from datetime import datetime, timedelta
from blockchain import util
from pytrends.request import TrendReq


class NonPricingData(): # Update needed: add update feature similar to poloniexdata.py

    # (helper function) handles making Google Trends API calls
    # returns interest level df
    def TrendDataRequest(self, new_date, old_date, kw_list):
        # trend object
        pytrend = TrendReq()

        # Create new timeframe for which we download data
        timeframe = new_date.strftime('%Y-%m-%d') + ' ' + old_date.strftime('%Y-%m-%d')

        # API Call -- Download Trend Data
        pytrend.build_payload(kw_list=kw_list, timeframe=timeframe)

        return pytrend.interest_over_time()

    # Generates google trend data with day granularity given a single search term.
    def daily_trends(self, search_term):
        # The maximum for a timeframe for which we get daily data is 270.
        # Therefore we could go back 269 days. However, since there might
        # be issues when rescaling, e.g. zero entries, we should have an
        # overlap that does not consist of only one period. Therefore,
        # I limit the step size to 250. This leaves 19 periods for overlap.
        maxstep = 269
        overlap = 40
        step = maxstep - overlap + 1
        kw_list = [search_term]
        start_date = datetime(2011, 12, 9).date()
        start_date = datetime(2016, 3, 9).date()

        ## FIRST RUN ##
        # Login to Google. Only need to run this once, the rest of requests will use the same session.
        pytrend = TrendReq()

        # Run the first time (if we want to start from today, otherwise we need to ask for an end_date as well
        today = datetime.today().date()
        old_date = today

        # Go back in time
        new_date = today - timedelta(days=step)
        #new_date = today - timedelta(days=5)

        # request trend data
        interest_over_time_df = NonPricingData().TrendDataRequest(new_date=new_date,old_date=old_date,kw_list=kw_list)

        ## RUN ITERATIONS
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


            if (temp_df.empty):
                raise ValueError(
                    'Google sent back an empty dataframe. Possibly there were no searches at all during the this period! Set start_date to a later date.')

            # Renormalize the dataset and drop last line
            for kw in kw_list:
                beg = new_date
                end = old_date - timedelta(days=1)

                # Since we might encounter zeros, we loop over the
                # overlap until we find a non-zero element
                for t in range(1, overlap + 1):
                    # print('t = ',t)
                    # print(temp_df[kw].iloc[-t])
                    if temp_df[kw].iloc[-t] != 0:
                        # theses two elements in the df represent the same day, different sample for normalization
                        scaling = interest_over_time_df[kw].iloc[t - 1] / temp_df[kw].iloc[-t]
                        # print('Found non-zero overlap!')
                        break
                    elif t == overlap:
                        print('Did not find non-zero overlap, set scaling to zero! Increase Overlap!')
                        scaling = 0
                # Apply scaling
                temp_df.loc[beg:end, kw] = temp_df.loc[beg:end, kw] * scaling
            interest_over_time_df = pd.concat([temp_df[:-overlap], interest_over_time_df])

        # return dataset
        return interest_over_time_df


    def hash_rate(self):
        response = util.call_api('charts/hash-rate?format=json', base_url='https://api.blockchain.info/')
        hash_json = json.loads(response)

        times = np.zeros(len(hash_json['values']))
        hash_rates = np.zeros(len(hash_json['values']))
        for i in range(len(hash_json['values'])):
            times[i] = hash_json['values'][i]['x']
            hash_rates[i] = hash_json['values'][i]['y']

        return times, hash_rates

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

    for search_term in search_terms:
        df = NonPricingData().daily_trends(search_term)[search_term]

        # plot
        plt.plot(df)
        plt.xticks(rotation=20)
        plt.ylabel('Google Search Trends: \''+search_term+'\'')
        plt.show()

