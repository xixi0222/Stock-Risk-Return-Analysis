import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt 
import random
import numpy as np
from numpy import matlib
import scipy.optimize as sc_optim
from structs import *

class portfolio():
    def __init__(self):
        self.file_path = DATA_PATH
        self.two_stock_name = TWO_STOCK_NAMES
        self.index_name = INDEX_NAME
        self.cc_return_index, self.cc_return_two_return = self.calculate_return()

        self.weight_a = 0
        self.weight_b = 0
        self.alpha_2_comb = 0
        self.beta_2_comb = 0

    def calculate_return(self):
        file_path = self.file_path
        index_name = self.index_name
        two_stock_names = self.two_stock_name
        d = pd.read_excel(file_path, sheet_name=None)
        d_weekly = d['Weekly'].drop(0)

        close_index = d_weekly[index_name]
        # Rt = ln(Pt) − ln(Pt−1)
        ret_index = np.log(close_index / close_index.shift(1)).drop(1)

        two_ret = []
        for name in two_stock_names:
            close_single_stock = d_weekly[name]
            ret_single_stock = np.log(close_single_stock / close_single_stock.shift(1)).drop(1)
            two_ret.append(ret_single_stock)

        return ret_index, two_ret

    def calculate_2_stock_weight(self):
        ## Point estimation: Get alpha & beta

        two_ret = self.cc_return_two_return
        X = np.array(self.cc_return_index).reshape(len(self.cc_return_index), 1)  # 转化特征变量为二维数组

        two_alpha_set = []
        two_beta_set = []

        for ret in two_ret:
            model = linear_model.LinearRegression()
            y = list(ret)
            model.fit(X, y)

            beta = model.coef_
            alpha = model.intercept_

            two_alpha_set.append(alpha)
            two_beta_set.append(beta)
            
        # calculate cov(a, b)
        a = two_ret[0]
        b = two_ret[1]
        a_avg = sum(a)/len(a)
        b_avg = sum(b)/len(b)
        cov_ab = sum([(x - a_avg)*(y - b_avg) for x,y in zip(a, b)])/(len(a)-1)
        # Or we can use numpy to calculate the cov(a, b)
        # print(np.cov(two_ret)[0][1])

        # calculate sigma to get min(Var(sigma(X) + (1-sigma)Y))
        # X: a; Y: b
        sigma = (np.var(b) - cov_ab) / (np.var(a) + np.var(b) - 2 * cov_ab)
        alpha_2_comb = sigma * two_alpha_set[0] + (1-sigma) * two_alpha_set[1]
        beta_2_comb = (sigma * two_beta_set[0] + (1-sigma) * two_beta_set[1])[0]

        print("We use ", sigma, " * X + ",1 - sigma, " Y")
        print("To get min(Var(sigma(X) + (1-sigma)Y)), the alpha': ", alpha_2_comb, ", the beta': ", beta_2_comb)

        self.weight_a = sigma
        self.weight_b = 1- sigma
        self.alpha_2_comb = alpha_2_comb
        self.beta_2_comb = beta_2_comb

    def output_result_to_excel(self):
        statistics = pd.DataFrame(index=['stats'])
        statistics['two_stock_name'] = str(self.two_stock_name[0])+', '+str(self.two_stock_name[1])
        statistics['index_name'] = self.index_name
        statistics['weight_a'] = self.weight_a
        statistics['weight_b'] = self.weight_b
        statistics['alpha_2_comb'] = self.alpha_2_comb
        statistics['beta_2_comb'] = self.beta_2_comb

        return statistics.T