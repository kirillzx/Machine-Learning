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
from classPortfolio import Portfolio


class OptimizationP(Portfolio):
    def __init__(self, names, dates):
        super().__init__(names, dates)
        self.bnds = tuple([(0.05, 0.15) for i in range(self.n)])
        self.init = np.repeat(0.05, self.n)
        self.cons = 0
        self.opt1 = 0
        # p = Portfolio(names, dates)
        self.free_cov_matrix = 0
        self.free_month_return = 0
        self.cov_matrix = 0
        self.return1 = 0
        self.dates = dates

    def get_help(self):
        print('maxReturn - maximize return of the portfolio\nminRisk - minimize risk of the portfolio',\
              '\nmaxSharpe - maximize a Sharpe ratio\nmaxSortino - maximize a Sortino ratio',\
             '\nminGenFun - minimize generalized function')

    def optimizeP(self, type_opt, boundsP, initP, rf=0.0037, t=0.5):
        self.cov_matrix = self.get_month_COV()
        self.return1 = self.get_month_return()

        if type_opt == 'maxReturn':
            self.bnds = tuple([(boundsP[0], boundsP[1]) for i in range(self.n)])
            self.init = np.repeat(initP, self.n)
            self.cons = ({'type':'eq', 'fun': lambda x: np.dot(np.ones(self.n), x)-1},\
                        {'type':'ineq', 'fun': lambda x: -np.dot(x, np.dot(self.cov_matrix, x))+0.01})

            self.opt1 = minimize(lambda x: -np.dot(x, self.return1), self.init,\
                                 bounds=self.bnds, constraints=self.cons)
            self.w = self.opt1.x

            return self.opt1

        elif type_opt == 'minRisk':
            self.bnds = tuple([(boundsP[0], boundsP[1]) for i in range(self.n)])
            self.init = np.repeat(initP, self.n)
            self.cons = ({'type':'eq', 'fun': lambda x: np.dot(np.ones(self.n), x)-1})
#                         {'type':'ineq', 'fun': lambda x: np.dot(self.return1, x)-0.02})

            self.opt1 = minimize(lambda x: np.dot(np.dot(x, self.cov_matrix), x), self.init,\
                                 method='SLSQP', bounds=self.bnds, constraints=self.cons)
            self.w = self.opt1.x

            return self.opt1

        elif type_opt == 'maxSharpe':

            def sharpe_f(x):
                return -(np.dot(self.return1, x) - rf)/np.dot(np.dot(x, self.cov_matrix), x)

            self.bnds = tuple([(boundsP[0], boundsP[1]) for i in range(self.n)])
            self.init = np.repeat(initP, self.n)
            self.cons = ({'type':'eq', 'fun': lambda x: np.dot(np.ones(self.n), x)-1})

            self.opt1 = minimize(sharpe_f, self.init, method='SLSQP', bounds=self.bnds, constraints=self.cons)
            self.w = self.opt1.x

            return self.opt1

        elif type_opt == 'maxSortino':
            r = self.return1
            array = []

            for i in range(len(r)):
                if r[i] < rf:
                    array.append(r[i])
            r2 = 0

            for i in range(len(array)):
                r2 += (array[i] - rf)**2

            def sortino_f(x):
                return -(np.dot(self.return1, x) - rf)/(r2/self.n)

            self.bnds = tuple([(boundsP[0], boundsP[1]) for i in range(self.n)])
            self.init = np.repeat(initP, self.n)
            self.cons = ({'type':'eq', 'fun': lambda x: np.dot(np.ones(self.n), x)-1})

            self.opt1 = minimize(sortino_f, self.init, bounds=self.bnds, constraints=self.cons)
            self.w = self.opt1.x

            return self.opt1

        elif type_opt == 'minGenFun':

            def genFun(x):
                return -t * np.dot(self.return1, x) + np.dot(np.dot(x, self.cov_matrix), x)

            self.bnds = tuple([(boundsP[0], boundsP[1]) for i in range(self.n)])
            self.init = np.repeat(initP, self.n)
            self.cons = ({'type':'eq', 'fun': lambda x: np.dot(np.ones(self.n), x)-1})

            self.opt1 = minimize(genFun, self.init, bounds=self.bnds, constraints=self.cons)
            self.w = self.opt1.x

            return self.opt1

        elif type_opt == 'minGenFunFree':

            self.free_cov_matrix = np.vstack((np.hstack((self.cov_matrix, np.zeros((self.n,1)))), np.zeros(self.n+1)))
            self.free_month_return = np.append(self.return1, rf)

            def genFunFree(x):
                return -t * np.dot(self.free_month_return, x) + np.dot(np.dot(x, self.free_cov_matrix), x)

            self.bnds = tuple([(boundsP[0], boundsP[1]) for i in range(self.n+1)])
            self.init = np.repeat(initP, self.n+1)
            self.cons = ({'type':'eq', 'fun': lambda x: np.dot(np.ones(self.n+1), x)-1})

            self.opt1 = minimize(genFunFree, self.init, bounds=self.bnds, constraints=self.cons)
            self.w = self.opt1.x

            return self.opt1

        else:
            return 'Choose the type of optimization. Use Object.get_help() to learn more.'

    def multi_criteria_opt(self, boundsP, initP, rf, t):
        self.cov_matrix = self.get_month_COV()
        self.return1 = self.get_month_return()

        f1 = -self.optimizeP('maxReturn', boundsP, initP, rf, t).fun
        f2 = -self.optimizeP('maxSharpe', boundsP, initP, rf, t).fun
        f3 = self.optimizeP('minGenFun', boundsP, initP, rf, t).fun

        r1 = np.append(self.return1, f1)
        r2 = np.append(self.return1, f2)
        r3 = np.append(self.return1, f3)

        def sharpe_f(x):
                return (np.dot(self.return1, x) - rf)/np.sqrt(np.dot(np.dot(x, self.cov_matrix), x))

        def genFun(x, t):
                return -t * np.dot(self.return1, x) + 0.5 * np.dot(np.dot(x, self.cov_matrix), x)

        def fun_x(x):
            array = np.zeros(self.n)
            array = np.append(array, 1)
            return np.dot(array, x)

        self.bnds = [(boundsP[0], boundsP[1]) for i in range(self.n)]
        self.bnds.append((0, 1))
        self.bnds = tuple(self.bnds)

        self.init = np.repeat(initP, self.n+1)
        self.cons = [{'type':'eq', 'fun': lambda x: np.dot(x[:self.n], np.ones(self.n))-1},
                    {'type':'ineq', 'fun': lambda x: np.array([-f1 + np.dot(x[:self.n], \
                                                                self.return1) + f1*x[self.n],
                                                              -f2 + sharpe_f(x[:self.n]) + f2*x[self.n]
                                                              -genFun(x[:self.n], t) + f3 + f3*x[self.n]])}]

        self.opt1 = minimize(lambda x: x[self.n], self.init, bounds=self.bnds, constraints=self.cons)
        self.w = self.opt1.x[:self.n]

        return self.opt1

    def optimal_t(self):
        l = np.ones(self.n)
        inv_COV = np.linalg.inv(self.get_month_COV())
        r = self.get_month_return()

        h0 = np.dot(l, inv_COV)/np.dot(np.dot(l, inv_COV), l)
        h1 = np.dot(inv_COV, r) - np.dot(inv_COV, l)*np.dot(np.dot(l, inv_COV), r)/np.dot(np.dot(l, inv_COV), l)

        alpha0 = np.dot(r, h0)
        alpha1 = np.dot(r, h1)
        beta0 = np.dot(np.dot(h0, self.get_month_COV()), h0)

        return (h0, h1, alpha0, alpha1, beta0)

    def efficient_frontier(self, boundsP, initP):
        temp = round(10000*self.optimizeP('minRisk', boundsP, initP).fun)
        self.cov_matrix = self.get_month_COV()
        self.return1 = self.get_month_return()

        self.bnds = tuple([(boundsP[0], boundsP[1]) for i in range(self.n)])
        self.init = np.repeat(initP, self.n)

        eff_frontier = []

        for i in range(temp, 100):
            self.cons = ({'type':'eq', 'fun': lambda x: np.dot(np.ones(self.n), x)-1},\
                        {'type':'ineq', 'fun': lambda x: -np.dot(x, np.dot(self.cov_matrix, x)) + i/10000})
            opt = minimize(lambda x: -np.dot(self.return1, x), self.init,\
                                 bounds=self.bnds, constraints=self.cons)

            eff_frontier.append([i/100, round(-opt.fun*100, 3)])

        return np.array(eff_frontier).T

    def not_efficient_region(self, boundsP, initP):
        temp = round(10000*self.optimizeP('minRisk', boundsP, initP).fun)
        self.cov_matrix = self.get_month_COV()
        self.return1 = self.get_month_return()

        self.bnds = tuple([(boundsP[0], boundsP[1]) for i in range(self.n)])
        self.init = np.repeat(initP, self.n)

        not_eff_reg = []

        for i in range(temp, 100):
            self.cons = ({'type':'eq', 'fun': lambda x: np.dot(np.ones(self.n), x)-1},\
                        {'type':'ineq', 'fun': lambda x: -np.dot(x, np.dot(self.cov_matrix, x)) + i/10000})
            opt = minimize(lambda x: np.dot(self.return1, x), self.init,\
                                 bounds=self.bnds, constraints=self.cons)

            not_eff_reg.append([i/100, opt.fun*100])

        return np.array(not_eff_reg).T
