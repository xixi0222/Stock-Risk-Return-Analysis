import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt 
import random
import numpy as np
from numpy import matlib
import scipy.optimize as sc_optim
from structs import * 

class Single_Index_Model():
    def __init__(self):
        self.file_path = DATA_PATH
        self.single_stock_name = SINGLE_STOCK_NAME
        self.index_name = INDEX_NAME
        self.cc_return_index, self.cc_return_single_stock = self.calculate_return()
        self.cc_return_single_stock_mean = 0 
        self.cc_return_single_stock_var = 0
        self.cc_return_single_stock_std = 0

        self.predict_cc_return = 0
        self.alpha = 0
        self.beta = 0

        self.ci_classical_method_alpha_l = 0
        self.ci_classical_method_alpha_h = 0
        self.ci_classical_method_beta_l = 0
        self.ci_classical_method_beta_h = 0

        self.ci_non_para_bs_percentile_alpha_l = 0
        self.ci_non_para_bs_percentile_alpha_h = 0
        self.ci_non_para_bs_percentile_beta_l = 0
        self.ci_non_para_bs_percentile_beta_h = 0

        self.ci_non_para_bs_t_alpha_l = 0
        self.ci_non_para_bs_t_alpha_h = 0
        self.ci_non_para_bs_t_beta_l = 0
        self.ci_non_para_bs_t_beta_h = 0

        self.ci_para_bs_percentile_alpha_l = 0
        self.ci_para_bs_percentile_alpha_h = 0
        self.ci_para_bs_percentile_beta_l = 0
        self.ci_para_bs_percentile_beta_h = 0

        self.ci_para_bs_t_alpha_l = 0
        self.ci_para_bs_t_alpha_h = 0
        self.ci_para_bs_t_beta_l = 0
        self.ci_para_bs_t_beta_h = 0

        self.ci_para_bs_se_alpha_l = 0
        self.ci_para_bs_se_alpha_h = 0
        self.ci_para_bs_se_beta_l = 0
        self.ci_para_bs_se_beta_h = 0

    def calculate_return(self):
        d = pd.read_excel(self.file_path, sheet_name=None)
        d_weekly = d['Weekly'].drop(0)

        close_index, close_single_stock = d_weekly[self.index_name], d_weekly[self.single_stock_name]

        # CC Return = ln(Pt) − ln(Pt−1)
        cc_return_index, cc_return_single_stock = np.log(close_index / close_index.shift(1)).drop(1), \
                                                  np.log(close_single_stock / close_single_stock.shift(1)).drop(1)
        return cc_return_index, cc_return_single_stock


    def point_estimation(self):
        cc_return_index = self.cc_return_index
        cc_return_single_stock = self.cc_return_single_stock
        model = linear_model.LinearRegression()
        X = np.array(cc_return_index).reshape(len(cc_return_index), 1)  # 转化特征变量为二维数组
        y = list(cc_return_single_stock)
        model.fit(X, y)

        beta = model.coef_
        alpha = model.intercept_
        # You can use predicted_y to draw some graphs
        predicted_y = beta * list(cc_return_index) + alpha

        self.predict_cc_return = predicted_y
        self.alpha = alpha
        self.beta = beta

        print("alpha: ", alpha)
        print("beta: ", beta)

    def confidence_interval(self):
        self.ci_classical_method()
        self.ci_bootstrapping_method()

    def ci_classical_method(self):
        # Confidence interval: Classical method - SE
        print("---------2.2.1 Classical method-----------")

        ret_len = len(self.cc_return_index)
        y = list(self.cc_return_single_stock)
        X = np.array(self.cc_return_index).reshape(len(self.cc_return_index), 1)
        alpha = self.alpha
        beta = self.beta

        y_var = np.var(y)
        sample_mean = np.mean(X)
        sample_var = np.var(X)

        se_alpha = y_var * ((1 / ret_len) + (sample_mean ** 2) / (ret_len * sample_var))
        se_beta = y_var * (1 / (ret_len * sample_var))
        print("----置信区间 95% using +/- 2 * SE")
        print("alpha CI:")
        print(alpha - 2 * se_alpha, alpha + 2 * se_alpha)
        print("beta CI:")
        print(beta - 2 * se_beta, beta + 2 * se_beta)

        self.ci_classical_method_alpha_l = alpha - 2 * se_alpha
        self.ci_classical_method_alpha_h = alpha + 2 * se_alpha
        self.ci_classical_method_beta_l = beta - 2 * se_beta
        self.ci_classical_method_beta_h = beta + 2 * se_beta

    def ci_bootstrapping_method(self):
        print("-------2.2.2 Use Bootstrap method--------")
        B = 1000
        sigma = 0.05
        print("Using confidance level: ", sigma)
        lower_bound_rate = sigma / 2
        higher_bound_rate = 1 - lower_bound_rate
        lower_bound = int(lower_bound_rate * B)
        higher_bound = int(higher_bound_rate * B)
        beta_set = []
        alpha_set = []
        beta_t_set = []
        alpha_t_set = []

        ret_len = len(self.cc_return_index)
        y = list(self.cc_return_single_stock)
        X = np.array(self.cc_return_index).reshape(ret_len, 1)
        alpha = self.alpha
        beta = self.beta

        y_mean = np.mean(y)
        y_var = np.var(y)
        y_std = np.std(y)

        sample_mean = np.mean(X)
        sample_var = np.var(X)
        sample_std = np.std(X)

        se_alpha = y_var * ((1 / ret_len) + (sample_mean ** 2) / (ret_len * sample_var))
        se_beta = y_var * (1 / (ret_len * sample_var))

        # Non-parametric bootstrap
        print("-------2.2.2.1 Use Non-parametric Bootstrap method--------")
        beta_set.clear()
        alpha_set.clear()
        beta_t_set.clear()
        alpha_t_set.clear()

        for i in range(B):
            rs_idx = [random.randint(0, ret_len - 1) for _ in range(ret_len)]
            rs_X = X[rs_idx]
            rs_y = np.array(y)[rs_idx]
            
            rs_y_var = np.var(rs_y)
            
            rs_model = linear_model.LinearRegression()
            rs_model.fit(rs_X, rs_y)
            rs_beta = rs_model.coef_
            rs_alpha = rs_model.intercept_
            
            rs_mean = np.mean(rs_X)
            rs_var = np.var(rs_X)
            rs_se_alpha = rs_y_var * ((1 / ret_len) + ((rs_mean ** 2) / (ret_len * rs_var)))
            rs_se_beta = rs_y_var * (1 / (ret_len * rs_var))
            
            rs_alpha_t = (rs_alpha - alpha) / rs_se_alpha
            rs_beta_t = (rs_beta - beta) / rs_se_beta
            
            beta_set.append(rs_beta)
            alpha_set.append(rs_alpha)
            alpha_t_set.append(rs_alpha_t)
            beta_t_set.append(rs_beta_t)

        beta_set.sort()
        alpha_t_set.sort()
        beta_t_set.sort()

        # non-parametric bootstrap percentile method
        print("------2.2.2.1.1 Use non-parametric bootstrap percentile method-------")

        print("Alpha CI:")
        print(alpha_set[lower_bound], alpha_set[higher_bound])
        print("Beta CI:")
        print(beta_set[lower_bound], beta_set[higher_bound])

        self.ci_non_para_bs_percentile_alpha_l = alpha_set[lower_bound]
        self.ci_non_para_bs_percentile_alpha_h = alpha_set[higher_bound]
        self.ci_non_para_bs_percentile_beta_l = beta_set[lower_bound]
        self.ci_non_para_bs_percentile_beta_h = beta_set[higher_bound]

        # non-parametric bootstrap t method
        print("------2.2.2.1.2 Use non-parametric bootstrap t method-------")

        print("Alpha CI:")
        print(alpha - alpha_t_set[higher_bound] * se_alpha, alpha - alpha_t_set[lower_bound] * se_alpha)
        print("Beta CI:")
        print(beta - beta_t_set[higher_bound] * se_beta, beta - beta_t_set[lower_bound] * se_beta)

        self.ci_non_para_bs_t_alpha_l = alpha - alpha_t_set[higher_bound] * se_alpha
        self.ci_non_para_bs_t_alpha_h = alpha - alpha_t_set[lower_bound] * se_alpha
        self.ci_non_para_bs_t_beta_l = beta - beta_t_set[higher_bound] * se_beta
        self.ci_non_para_bs_t_beta_h = beta - beta_t_set[lower_bound] * se_beta

        # Parameter bootstrap
        print("----------2.2.2.2 Use parametric bootstrap method-----------")
        beta_set.clear()
        alpha_set.clear()
        beta_t_set.clear()
        alpha_t_set.clear()

        para_x_samples = sample_std * matlib.randn((B, ret_len)) + sample_mean
        para_eps_samples = y_std * matlib.randn((B, ret_len)) + y_mean
        para_y_samples = beta[0] * para_x_samples + alpha + para_eps_samples

        for i in range(B):
            rs_y = para_y_samples[i].reshape((-1,1))
            rs_X = np.array(para_x_samples[i]).reshape(ret_len, 1) 
            rs_y_var = np.var(rs_y)
            rs_model = linear_model.LinearRegression()
            rs_model.fit(rs_X, rs_y)
            rs_beta = rs_model.coef_
            rs_alpha = rs_model.intercept_
            rs_mean = np.mean(rs_X)
            rs_std = np.std(rs_X)
            rs_var = np.var(rs_X)
            rs_se_alpha = rs_y_var * ((1 / ret_len) + ((rs_mean ** 2) / (ret_len * rs_var)))
            rs_se_beta = rs_y_var * (1 / (ret_len * rs_var))
            
            rs_alpha_t = (rs_alpha - alpha) / rs_se_alpha
            rs_beta_t = (rs_beta - beta) / rs_se_beta
            
            beta_set.append(rs_beta)
            alpha_set.append(rs_alpha)
            alpha_t_set.append(rs_alpha_t)
            beta_t_set.append(rs_beta_t)

        alpha_set.sort()
        beta_set.sort()
        alpha_t_set.sort()
        beta_t_set.sort()

        # Parametric bootstrap method - percentile method
        print("--------2.2.2.2.1 Use parametric bootstrap method - percentile method----------")
        print("Alpha CI:")
        print(alpha_set[lower_bound], alpha_set[higher_bound])
        print("Beta CI:")
        print(beta_set[lower_bound], beta_set[higher_bound])

        self.ci_para_bs_percentile_alpha_l = alpha_set[lower_bound]
        self.ci_para_bs_percentile_alpha_h = alpha_set[higher_bound]
        self.ci_para_bs_percentile_beta_l = beta_set[lower_bound]
        self.ci_para_bs_percentile_beta_h = beta_set[higher_bound]

        # Parametric bootstrap method - t method

        print("---------2.2.2.2.2 Use parametric bootstrap method - t method----------")

        print("Alpha CI:")
        print(alpha - alpha_t_set[higher_bound] * se_alpha,\
              alpha - alpha_t_set[lower_bound] * se_alpha)
        print("Beta CI:")
        print(beta - beta_t_set[higher_bound] * se_beta,\
              beta - beta_t_set[lower_bound] * se_beta)

        self.ci_para_bs_t_alpha_l = alpha - alpha_t_set[higher_bound] * se_alpha
        self.ci_para_bs_t_alpha_h = alpha - alpha_t_set[lower_bound] * se_alpha
        self.ci_para_bs_t_beta_l = beta - beta_t_set[higher_bound] * se_beta
        self.ci_para_bs_t_beta_h = beta - beta_t_set[lower_bound] * se_beta

        # parametric bootstrap method - SEboot method
        # calculate SEboot using resampling 

        print("---------2.2.2.2.3 Use parametric bootstrap method - SEboot method----------")

        se_boot_alpha = (sum([(i - np.mean(alpha_set)) ** 2 for i in alpha_set]) / (B - 1)) ** 0.5
        se_boot_beta = (sum([(i - np.mean(beta_set)) ** 2 for i in beta_set]) / (B - 1)) ** 0.5

        print("----置信区间 95% using +/- 2 * SE")
        print("Alpha CI:")
        print(alpha - 2 * se_boot_alpha, alpha + 2 * se_boot_alpha)
        print("Beta CI:")
        print(beta - 2 * se_boot_beta, beta + 2 * se_boot_beta)

        self.ci_para_bs_se_alpha_l = alpha - 2 * se_boot_alpha
        self.ci_para_bs_se_alpha_h = alpha + 2 * se_boot_alpha
        self.ci_para_bs_se_beta_l = beta - 2 * se_boot_beta
        self.ci_para_bs_se_beta_h = beta + 2 * se_boot_beta

    def output_result_to_excel(self):
        statistics = pd.DataFrame(index=['stats'])
        statistics['single_stock_name'] = self.single_stock_name
        statistics['index_name'] = self.index_name
        statistics['alpha'] = self.alpha
        statistics['beta'] = self.beta

        statistics['ci_classical_method_alpha_l'] = self.ci_classical_method_alpha_l
        statistics['ci_classical_method_alpha_h'] = self.ci_classical_method_alpha_h
        statistics['ci_classical_method_beta_l'] = self.ci_classical_method_beta_l
        statistics['ci_classical_method_beta_h'] = self.ci_classical_method_beta_h

        statistics['ci_non_para_bs_percentile_alpha_l'] = self.ci_non_para_bs_percentile_alpha_l
        statistics['ci_non_para_bs_percentile_alpha_h'] = self.ci_non_para_bs_percentile_alpha_h
        statistics['ci_non_para_bs_percentile_beta_l'] = self.ci_non_para_bs_percentile_beta_l
        statistics['ci_non_para_bs_percentile_beta_h'] = self.ci_non_para_bs_percentile_beta_h

        statistics['ci_non_para_bs_t_alpha_l'] = self.ci_non_para_bs_t_alpha_l
        statistics['ci_non_para_bs_t_alpha_h'] = self.ci_non_para_bs_t_alpha_h
        statistics['ci_non_para_bs_t_beta_l'] = self.ci_non_para_bs_t_beta_l
        statistics['ci_non_para_bs_t_beta_h'] = self.ci_non_para_bs_t_beta_h 

        statistics['ci_para_bs_percentile_alpha_l'] = self.ci_para_bs_percentile_alpha_l
        statistics['ci_para_bs_percentile_alpha_h'] = self.ci_para_bs_percentile_alpha_h
        statistics['ci_para_bs_percentile_beta_l'] = self.ci_para_bs_percentile_beta_l
        statistics['ci_para_bs_percentile_beta_h'] = self.ci_para_bs_percentile_beta_h

        statistics['ci_para_bs_t_alpha_l'] = self.ci_para_bs_t_alpha_l
        statistics['ci_para_bs_t_alpha_h'] = self.ci_para_bs_t_alpha_h
        statistics['ci_para_bs_t_beta_l'] = self.ci_para_bs_t_beta_l
        statistics['ci_para_bs_t_beta_h'] = self.ci_para_bs_t_beta_h

        statistics['ci_para_bs_se_alpha_l'] = self.ci_para_bs_se_alpha_l
        statistics['ci_para_bs_se_alpha_h'] = self.ci_para_bs_se_alpha_h
        statistics['ci_para_bs_se_beta_l'] = self.ci_para_bs_se_beta_l
        statistics['ci_para_bs_se_beta_h'] = self.ci_para_bs_se_beta_h

        return statistics.T
