import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
import scipy.stats as sts
import yfinance as yf
from scipy.optimize import linprog
from scipy.optimize import minimize
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import datetime
plt.style.use('ggplot')

class Portfolio:
    def __init__(self, names, dates):
        self.n = len(names)
        self.names = names
        self.data = pd.DataFrame()
        self.new_data = pd.DataFrame()
        self.w = np.array([1/self.n]*self.n)
        self.month_return = 0
        self.month_COV = 0
        self.month_risk = 0
        self.dates = dates

    def get_ts(self):
        self.data = pd.DataFrame(columns=self.names)
        for name in self.names:
            self.data[name] = yf.download(name, self.dates[0], self.dates[1])['Adj Close']

    def get_ts_csv(self):
        self.data = pd.read_csv('quotes.csv', sep='\t', parse_dates=True, index_col='Date')

    def graphics(self):
        if self.data.empty:
            print('At first, use get_ts() for download data')
        else:
            fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
            (self.data.pct_change()+1).cumprod().plot(ax=ax)
            plt.show()

    def resample_to_months(self):
        self.new_data = self.data.resample('M').last()

#     def get_month_return(self):
#         self.month_return = self.new_data.pct_change().mean()
#         return self.month_return

    def get_pct_change(self):
        return self.new_data.pct_change(1).apply(lambda x: np.log(1+x))

    def get_month_return(self):
        return self.new_data.pct_change(1).apply(lambda x: np.log(1+x)).mean()

#     def get_month_COV(self):
#         self.month_COV = self.new_data.pct_change().cov()
#         return self.month_COV

    def get_month_COV(self):
        return self.new_data.pct_change(1).apply(lambda x: np.log(1+x)).cov()

    def get_month_risk(self, w):
        self.month_risk = np.dot(np.dot(w, self.get_month_COV()), w)
        return self.month_risk

    def get_returnP(self):
        return np.dot(self.w, self.get_month_return())

    def get_returnP_free(self, rf):
        return np.dot(self.w[:self.n], self.get_month_return()) + self.w[self.n] * rf

    def get_riskP(self):
        return np.dot(np.dot(self.w, self.get_month_COV()), self.w)

    def get_riskP_free(self):
        return np.dot(np.dot(self.w[:self.n], self.get_month_COV()), self.w[:self.n])

    def corr_matrix(self):
        fig = plt.figure(figsize=(8,6))
        cmap = sns.diverging_palette(220, 0, as_cmap=True)
        sns.heatmap(self.new_data.pct_change(1).apply(lambda x: np.log(1+x)).corr(), cmap=cmap, annot = True, square=True)

        b1,t1=plt.ylim()
        b1+=0.5
        t1-=0.5
        plt.ylim(b1, t1)
        plt.show()
