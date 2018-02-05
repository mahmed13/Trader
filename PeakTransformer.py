from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import numpy as np
import pandas as pd
import scipy as sc
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import math

class PeakTransformer(BaseEstimator, TransformerMixin):

    def transform(self, X, feature='close', **transform_params):
        h = PeakTransformer.ent(X[feature])
        print(h)
        # prob_density, logprob = PeakTransformer.kde(x = X['close'].values.reshape(-1,1), w= 1.9)

        X = X[feature].values
        w = 6         # gaussian kde histogram window size
        k = 2           # number of temporal neighbors to compare
        S = []
        for i in range(k,len(X)-k-1):

            N, N_prime = PeakTransformer.setbuilder(k,i,X)
            #print(N_prime)
            #print(N)
            #print('entropy H=', PeakTransformer.entropy(N, w))
            #print('entropy H=', PeakTransformer.entropy(N_prime, w))
            S.append(PeakTransformer.entropy(N, w) - PeakTransformer.entropy(N_prime, w))

        S = [abs(number) for number in S]
        print('len(X)', len(X),'len(X)-k', len(X)-k)
        print('len(S)', len(S), 'k=',k)
        max_S = max(S)
        min_S = min(S)
        mean_S = sum(S) / float(len(S))
        peaks_i = []
        for i in range(len(S)):
            if S[i] > (mean_S+max_S*3)/4:
                peaks_i.append(i+k-1)
        peaks = [X[i] for i in peaks_i]

        print('max',max_S)
        print('min',min_S)
        print('mean',mean_S)

        max_i = S.index(max(S))+k
        print('max index', max_i)
        min_i = S.index(min(S))+k
        print('min index',min_i)


        plt.plot(X)
        plt.plot(peaks_i, peaks, 'g^')
        plt.plot([min_i], [X[min_i]], 'r^')
        #plt.plot(X[max_i-k:max_i+k])
        plt.ylabel('BTC/USDT Price')
        plt.xlabel('Periods')
        plt.show()

        # x = X['close'].values
        # prob_density = [PeakTransformer.kde_test(X['close'].values, i, 50) for i in range(len(x)-50)]
        # print(prob_density)
        # plt.fill_between(x[:-50], prob_density, alpha=0.5)
        # plt.plot(x, np.full_like(x, -0.000009), '|k', markeredgewidth=0.01)
        # # plt.ylim(0.00, 0.30)
        # plt.show()

        return X

    def fit(self, X, y=None, **fit_params):
        return self

    def setbuilder(k, i, T):
        N_plus = T[i + 1:k + i + 1]                 # k right temporal neighbors
        N_minus = T[i - k:i]                        # k left temporal neighbors
        N_prime = T[i - k:k + i + 1]                # range of 2k elements centered around x_i (including x_i
        N = np.concatenate([N_minus, N_plus])       # range of 2k elements centered around x_i (excluding x_i)

        return N, N_prime

    def gaussian_kernel(x):
        if abs(x) < 1 :
            return (3/4.0)*(1 - math.pow(x,2))
        else:
            return (1.0/(math.sqrt(2) * math.pi)) * math.exp((-1/2.0)*math.pow(x,2))

    """Fit the model according to the given training data.

        Parameters
        ----------
        a : np.array
            array of values to be used for estimate
            
        i : int
            Index of value in the 'a' np.array to be estimated

        w : int
            Window size parameter for kde 

        Returns
        -------
        P_w(a_i) : list
            Probability density
            Returns PD estimate.
    """
    def kde_test(a, i, w):
        kernel_sum = 0
        h = (a[i]- a[i+w]) # actual bandwidth
        #h = (a[i] + w)
        if h == 0:
            return 0
        for j in range(len(a)):
            kernel_sum += PeakTransformer.gaussian_kernel((a[i]-a[j])/(h))
        return 1.0/(len(a)*abs(h)) * kernel_sum

    def kde(x, w):
        # instantiate and fit the KDE model
        kde = KernelDensity(bandwidth=w, kernel='gaussian')
        kde.fit(x)

        # score_samples returns the log of the probability density
        logprob = kde.score_samples(x)
        prob_density = np.exp(logprob)

        #print('log(P_w(a))',logprob)
        #print('P_w(a)=',prob_density)
        #print('sum(P_w(a))=', sum(prob_density))
        #print('max prob=', max(prob_density))
        #print('Value*(prob_density)=', x[:,0]*prob_density)
        H_w = (-prob_density*logprob)
        #print('H = Sum of:', H_w)
        #print('H =', sum(H_w))

        #plt.fill_between(x[:,0], prob_density, alpha=0.5)
        #plt.plot(x, np.full_like(x, -0.000009), '|k', markeredgewidth=0.01)
        #plt.ylim(0.00, 0.30)
        #plt.show()

        return prob_density, logprob

    def entropy(X, w):
        prob_density, logprob = PeakTransformer.kde(x= X.reshape(-1,1), w= w)

        h = -1.0 * prob_density * logprob
        h = sum(h)
        return h

    # Input a pandas series
    def ent(data):
        p_data = data.value_counts() / len(data)  # calculates the probabilities
        entropy = sc.stats.entropy(p_data)  # input probabilities to get the entropy
        return entropy

    def rolling_entropy(window, bins):
        cx = np.histogram(window, bins)[0]
        c_normalized = cx/float(np.sum(cx))
        c_normalized = c_normalized[np.nonzero(c_normalized)]
        h = -sum(c_normalized * np.log(c_normalized))
        return h