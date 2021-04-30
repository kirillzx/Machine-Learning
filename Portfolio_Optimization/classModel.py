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
from classOptimizeP import OptimizationP

class Model:
    def __init__(self, type_opt, names, dates, m, rf, t, curr, budget, rb=False):
        self.type_opt = type_opt
        self.portfolio = OptimizationP(names, dates)
        self.rf = rf
        self.t = t
        self.costs = []
        self.res = []
        self.m = m
        self.curr = curr
        self.budget = budget
        self.forecast_data = []
        self.names = names
#         self.portfolio.get_ts()
        self.portfolio.get_ts_csv()
        self.portfolio.resample_to_months()
        self.income = []
        self.rb = rb
        self.cov_matrix = 0
        self.r = 0
        self.risk = 0.01
        self.fee_array = []
        self.fee = 0.03
        self.tax = 0.13

    def opt(self):
        if self.type_opt == 'multi-criteria':
            self.portfolio.multi_criteria_opt((0, 1), 0.01, self.rf, self.t)
        else:
            self.portfolio.optimizeP(self.type_opt, (0, 1), 0.01, self.rf)

        self.cov_matrix = self.portfolio.get_month_COV()
        self.risk = np.dot(self.portfolio.w[:self.portfolio.n], \
                           np.dot(self.cov_matrix, self.portfolio.w[:self.portfolio.n]))
        self.r = np.dot(self.portfolio.w[:self.portfolio.n], self.portfolio.get_month_return())

    def buy(self): #get quantity of assets at moment t0
        self.opt()
        self.costs = []
        self.res = []
        self.costs = np.array(self.portfolio.data.iloc[-1])

        u = np.dot(self.portfolio.w, self.budget)

        for i in range(self.m):
            self.res.append(u[i] / self.costs[i])
            self.fee_array.append((self.fee*self.costs[i])*(u[i] / self.costs[i]))

        for i in range(self.m, self.portfolio.n):
            self.res.append(u[i] / (self.curr[-1]*self.costs[i]))

        return self.res

    def quant(self):
        temp = self.buy()
        return f'Quantity {self.type_opt}: {[round(i) for i in temp]}'

    def forecast(self):
        self.forecast_data = []
        for name in self.names:
            model = ARIMA(self.portfolio.new_data[name], order=(1,2,1), freq='M')
            model_fit = model.fit()
            forecast = model_fit.forecast(5)
            self.forecast_data.append(np.array(forecast))

        self.forecast_data = np.array(self.forecast_data)
        return self.forecast_data

    def portfolio_return(self):
        self.forecast()
        if self.rb:
            total_income = self.budget
            b = self.budget
            self.rb_data = self.portfolio.data
            for j in range(len(self.forecast_data[0])):
                self.portfolio.data.loc[f'pr{j}'] = self.forecast_data.T[j]
                self.budget = total_income
                self.income = []
                s = self.buy()
                for i in range(self.m):
                    self.income.append(self.forecast_data[i, j]*s[i] +\
                                   s[i]*self.costs[i]*np.random.normal(loc=self.r, scale=np.sqrt(self.risk)))

                for i in range(self.m, self.portfolio.n):
                    self.income.append(self.forecast_data[i, j]*self.curr[-1]*s[i] +\
                                       s[i]*self.costs[i]*np.random.normal(loc=self.r, scale=np.sqrt(self.risk)))
                total_income = sum(self.income)
            return total_income - b

        elif not self.rb:
            self.income = []
            s = self.buy()

            for i in range(self.m):
                self.income.append(self.forecast_data[i, -1]*s[i] +\
                                   s[i]*self.costs[i]*np.random.normal(loc=self.r, scale=np.sqrt(self.risk)))
                self.fee_array[i] = self.fee_array[i] + (self.fee*self.forecast_data[i, -1]*s[i])

            for i in range(self.m, self.portfolio.n):
                self.income.append(self.forecast_data[i, -1]*self.curr[-1]*s[i] +\
                                   s[i]*self.costs[i]*np.random.normal(loc=self.r, scale=np.sqrt(self.risk)))

            return sum(self.income) - self.budget - sum(self.fee_array) - (sum(self.income) - self.budget)*self.tax

        else:
            print('Specify True or False rebalance!')
