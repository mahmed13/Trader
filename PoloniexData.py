import os
import time
import pandas as pd
import multiprocessing
#from functools import partial

FETCH_URL = "https://poloniex.com/public?command=returnChartData&currencyPair=%s&start=%d&end=%d&period=300"
DATA_DIR = "data"
COLUMNS = ["pair_name","date","high","low","open","close","volume","quoteVolume","weightedAverage"]

class PoloniexTickers:
    def get_data(self, pair):
        time.sleep(2)
        datafile = os.path.join(DATA_DIR, pair+".csv") # stores actual prices
        timefile = os.path.join(DATA_DIR, pair) # keeps track of time

        # determine whether .csv file exist / coin is new
        if os.path.exists(datafile):
            newfile = False
            start_time = int(open(timefile).readline()) + 1
            end_time = 9999999999
        else:
            newfile = True
            start_time = 1388534400    # 2014.01.01
            end_time = 1458534400 # intermediate step for old coins
            outf = open(datafile, "a")
            outf.close()
            ft = open(timefile, "w")
            ft.write("%d\n" % end_time)
            ft.close()


        # create url from inputs
        url = FETCH_URL % (pair, start_time, end_time)
        print('[',int(time.time()),']',"Scraping prices from ticker %s from time %d to %d" % (pair, start_time, end_time), url)

        try:
            # set all values
            df = pd.read_json(url, convert_dates=False)
        except:
            print('[',int(time.time()),']','ERROR: Reading JSON failed on pair', pair, url)

        # insert pair name column
        df['pair_name'] = pair
        cols = df.columns.tolist()
        df = df[[cols[-1]] + cols[:-1]]

        # ?? No new ticker
        if df["date"].iloc[-1] == 0 and not newfile:
            return False

        # write new endtime to timefile
        ft = open(timefile,"w")
        # Avoids newer coins like BCH from incorrectly having a start_time of 1 on the next pass
        if df['date'].iloc[-1] != 0:
            end_time = df["date"].iloc[-1]
        ft.write("%d\n" % end_time)
        ft.close()

        # write new data to csv
        outf = open(datafile, "a")
        if newfile:
            df.to_csv(outf, index=False, columns=COLUMNS)
        else:
            df.to_csv(outf, index=False, columns=COLUMNS, header=False)
        outf.close()

        # run scraper again because coin is new
        if newfile:
            PoloniexTickers().get_data(pair)

        return True

    def update(self, tickers):
        # creates data directory
        if not os.path.exists(DATA_DIR):
            os.mkdir(DATA_DIR)
        try:
            # get all pair names
            df = pd.read_json("https://poloniex.com/public?command=return24hVolume")
        except:
            print('[',int(time.time()),']','ERROR: Getting all pair names')

        pairs = [pair for pair in df.columns if pair.startswith('USDT_')]
        # output if ticker doesnt exist
        for ticker in tickers:
            if ticker not in df.columns:
                print('[',int(time.time()),']','ERROR: Ticker',ticker,'does not exist')
                print('List of all accepted pairs:',pairs)

        # tickers = [] -> update all tickers, else run specific tickers
        if len(tickers) != 0:
            pairs = [pair for pair in pairs if pair in tickers]

        # multiprocessing
        pool = multiprocessing.Pool(4)
        out = pool.map(self.get_data, pairs)
        print('OUTPUT =',out)
        return out

def main():
    pass

if __name__ == '__main__':
    PoloniexTickers().update([])