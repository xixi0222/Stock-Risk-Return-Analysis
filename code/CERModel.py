import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt 
import random
import numpy as np
from numpy import matlib
import scipy.optimize as sc_optim
from structs import * 

class CERModel():
    def __init__(self):
        self.file_path = DATA_PATH
        self.single_stock_name = SINGLE_STOCK_NAME
        self.cc_return_single_stock = self.calculate_return()
        self.cc_return_single_stock_mean = 0 
        self.cc_return_single_stock_var = 0
        self.cc_return_single_stock_std = 0
        
        self.ci_classical_method_mean_l = 0
        self.ci_classical_method_mean_h = 0
        self.ci_classical_method_std_l = 0
        self.ci_classical_method_std_h = 0

        self.ci_non_para_bs_method_percentile_mean_l = 0
        self.ci_non_para_bs_method_percentile_mean_h = 0
        self.ci_non_para_bs_method_percentile_std_l = 0
        self.ci_non_para_bs_method_percentile_std_h = 0
        self.ci_non_para_bs_method_percentile_var_l = 0
        self.ci_non_para_bs_method_percentile_var_h = 0

        self.ci_non_para_bs_method_t_mean_l = 0
        self.ci_non_para_bs_method_t_mean_h = 0
        self.ci_non_para_bs_method_t_std_l = 0
        self.ci_non_para_bs_method_t_std_h = 0

        self.ci_para_bs_method_percentile_mean_l = 0
        self.ci_para_bs_method_percentile_mean_h = 0
        self.ci_para_bs_method_percentile_std_l = 0
        self.ci_para_bs_method_percentile_std_h = 0
        self.ci_para_bs_method_percentile_var_l = 0
        self.ci_para_bs_method_percentile_var_h = 0

        self.ci_para_bs_method_t_mean_l = 0
        self.ci_para_bs_method_t_mean_h = 0
        self.ci_para_bs_method_t_std_l = 0
        self.ci_para_bs_method_t_std_h = 0

        self.ci_para_bs_method_se_mean_l = 0
        self.ci_para_bs_method_se_mean_h = 0
        self.ci_para_bs_method_se_std_l = 0
        self.ci_para_bs_method_se_std_h = 0
        self.ci_para_bs_method_se_var_l = 0
        self.ci_para_bs_method_se_var_h = 0


    def calculate_return(self):
        d = pd.read_excel(self.file_path, sheet_name=None)
        d_weekly = d['Weekly'].drop(0)

        close_single_stock = d_weekly[self.single_stock_name]

        # CC Return = ln(Pt) − ln(Pt−1)
        ret_single_stock = np.log(close_single_stock / close_single_stock.shift(1)).drop(1)
        return ret_single_stock

    def print_data(self):
        print(self.single_stock_name)
        print(self.cc_return_single_stock)

    def point_estimation(self):
        self.cc_return_single_stock_mean = self.cc_return_single_stock.mean()
        self.cc_return_single_stock_var = self.cc_return_single_stock.var()
        self.cc_return_single_stock_std = self.cc_return_single_stock.std()
        print("Stock Name:", self.single_stock_name)
        print("Mean: ", self.cc_return_single_stock_mean)
        print("Var: ", self.cc_return_single_stock_var)
        print("Std: ", self.cc_return_single_stock_std)

    def confidence_interval(self):
        self.ci_classical_method()
        self.ci_bootstrapping_method()

    def ci_classical_method(self):
        # Prepare some value needed for later calculation
        X = self.cc_return_single_stock
        sample_mean = self.cc_return_single_stock_mean
        sample_std = self.cc_return_single_stock_std
        ret_len = len(X)
        
        # Confidence interval: Classical method 
        print("-------1.2.1 Use Classical method--------")
        se_mean = sample_std / (ret_len ** 0.5)
        se_std = sample_std / ((2 * ret_len) ** 0.5)
        print("----Confidence Interval 95% using +/- 2 * SE")
        print("Mean CI:")
        print("Lower bound: ", sample_mean - 2 * se_mean, "Higher bound: ", sample_mean + 2 * se_mean)
        print("Std CI:")
        print("Lower bound: ", sample_std - 2 * se_std, "Higher bound: ", sample_std + 2 * se_std)

        self.ci_classical_method_mean_l = sample_mean - 2 * se_mean
        self.ci_classical_method_mean_h = sample_mean + 2 * se_mean
        self.ci_classical_method_std_l = sample_std - 2 * se_std
        self.ci_classical_method_std_h = sample_std + 2 * se_std

    def ci_bootstrapping_method(self):
        print("-------1.2.2 Use Bootstrap method--------")
        B = 10
        sigma = 0.1  # Common confidence level: 0.1, 0.05, 0.01
        lower_bound_rate = sigma / 2
        higher_bound_rate = 1 - lower_bound_rate
        lower_bound = int(lower_bound_rate * B)
        higher_bound = int(higher_bound_rate * B)
        X = self.cc_return_single_stock
        ret_len = len(X)

        sample_mean = self.cc_return_single_stock_mean
        sample_std = self.cc_return_single_stock_std
        sample_var = self.cc_return_single_stock_var

        mean_set = []
        var_set = []
        std_set = []
        mean_t_set = []
        std_t_set = []
        var_t_set = []

        # Non-parametric bootstrap
        print("-------1.2.2.1 Use Non-parametric Bootstrap method--------")
        mean_set.clear()
        var_set.clear()
        std_set.clear()
        mean_t_set.clear()
        std_t_set.clear()
        var_t_set.clear()
        for i in range(B):
            rs_idx = [random.randint(0, ret_len - 1) for _ in range(ret_len)]
            rs_X = X[rs_idx]
            
            rs_mean = np.mean(rs_X)
            rs_std = np.std(rs_X)
            rs_var = np.var(rs_X)

            rs_mean_t = (rs_mean - sample_mean) / rs_std * (ret_len ** 0.5)
            rs_std_t = (rs_std - sample_std) / rs_std * ((2 * ret_len) ** 0.5)
            rs_var_t = 0
            # 方差没有SE。可以用卡方分布，又因为卡方分布对于样本数据只和样本方差这单一变量有关，所以排序不会改变。
            # 答案会和quantile方法一样，所以在这里不重复
            
            mean_set.append(rs_mean)
            std_set.append(rs_std)
            var_set.append(rs_var)
            
            mean_t_set.append(rs_mean_t)
            std_t_set.append(rs_std_t)
            var_t_set.append(rs_var_t)
            
        mean_set.sort()
        var_set.sort()
        std_set.sort()

        mean_t_set.sort()
        std_t_set.sort()
        var_t_set.sort()

        # Non-parametric bootstrap - percentile method

        print("------1.2.2.1.1 Use non-parametric bootstrap percentile method-------")
        print("Mean CI:")
        print(mean_set[lower_bound], mean_set[higher_bound])
        print("Variance CI:")
        print(var_set[lower_bound], var_set[higher_bound])
        print("Std CI:")
        print(std_set[lower_bound], std_set[higher_bound])

        self.ci_non_para_bs_method_percentile_mean_l = mean_set[lower_bound]
        self.ci_non_para_bs_method_percentile_mean_h = mean_set[higher_bound]
        self.ci_non_para_bs_method_percentile_std_l = std_set[lower_bound]
        self.ci_non_para_bs_method_percentile_std_h = std_set[higher_bound]
        self.ci_non_para_bs_method_percentile_var_l = var_set[lower_bound]
        self.ci_non_para_bs_method_percentile_var_h = var_set[higher_bound]

        # Non-parametric bootstrap - t method
        print("-------1.2.2.1.2 Use non-parametric bootstrap t method--------")
        print("Mean CI:")
        print(sample_mean - mean_t_set[higher_bound] * sample_std / (ret_len ** 0.5), 
            sample_mean - mean_t_set[lower_bound] * sample_std / (ret_len ** 0.5))
        print("Std CI:")
        print(sample_std - std_t_set[higher_bound] * sample_std / ((2 * ret_len) ** 0.5), 
            sample_std - std_t_set[lower_bound] * sample_std / ((2 * ret_len) ** 0.5))
        # 方差没有SE。可以用卡方分布，又因为卡方分布对于样本数据只和样本方差这单一变量有关，所以排序不会改变。
        # 答案会和quantile方法一样，所以在这里不重复

        self.ci_non_para_bs_method_t_mean_l = sample_mean - mean_t_set[higher_bound] * sample_std / (ret_len ** 0.5)
        self.ci_non_para_bs_method_t_mean_h = sample_mean - mean_t_set[lower_bound] * sample_std / (ret_len ** 0.5)
        self.ci_non_para_bs_method_t_std_l = sample_std - std_t_set[higher_bound] * sample_std / ((2 * ret_len) ** 0.5)
        self.ci_non_para_bs_method_t_std_h = sample_std - std_t_set[lower_bound] * sample_std / ((2 * ret_len) ** 0.5)

        # Parameter bootstrap
        print("----------1.2.2.2 Use parametric bootstrap method-----------")
        
        # Parametric bootstrap method - percentile method
        print("--------1.2.2.2.1 Use parametric bootstrap method - percentile method----------")

        mean_set.clear()
        var_set.clear()
        std_set.clear()
        mean_t_set.clear()
        std_t_set.clear()
        var_t_set.clear()

        para_x_samples = sample_std * matlib.randn((B, ret_len)) + sample_mean

        mean_set = [np.mean(i) for i in para_x_samples]
        std_set = [np.std(i) for i in para_x_samples]
        var_set = [np.var(i) for i in para_x_samples]
            
        mean_set.sort()
        std_set.sort()
        var_set.sort()

        print("Mean CI:")
        print(mean_set[lower_bound], mean_set[higher_bound])
        print("Variance CI:")
        print(var_set[lower_bound], var_set[higher_bound])
        print("Std CI:")
        print(std_set[lower_bound], std_set[higher_bound])

        self.ci_para_bs_method_percentile_mean_l = mean_set[lower_bound]
        self.ci_para_bs_method_percentile_mean_h = mean_set[higher_bound]
        self.ci_para_bs_method_percentile_std_l = std_set[lower_bound]
        self.ci_para_bs_method_percentile_std_h = std_set[higher_bound]
        self.ci_para_bs_method_percentile_var_l = var_set[lower_bound]
        self.ci_para_bs_method_percentile_var_h = var_set[higher_bound]

        # Parametric bootstrap method - t method

        print("---------1.2.2.2.2 Use parametric bootstrap method - T method----------")
        mean_set.clear()
        var_set.clear()
        std_set.clear()
        mean_t_set.clear()
        std_t_set.clear()
        var_t_set.clear()

        para_x_samples = sample_std * matlib.randn((B, ret_len)) + sample_mean

        for i in para_x_samples:
            rs_X = i
            rs_mean = np.mean(rs_X)
            rs_std = np.std(rs_X)
            rs_var = np.var(rs_X)

            rs_mean_t = (rs_mean - sample_mean) / rs_std * (ret_len ** 0.5)
            rs_std_t = (rs_std - sample_std) / rs_std * ((2 * ret_len) ** 0.5)
            rs_var_t = 0
            
            mean_set.append(rs_mean)
            std_set.append(rs_std)
            var_set.append(rs_var)
            
            mean_t_set.append(rs_mean_t)
            std_t_set.append(rs_std_t)
            var_t_set.append(rs_var_t)
            
        mean_set.sort()
        var_set.sort()
        std_set.sort()

        mean_t_set.sort()
        std_t_set.sort()
        var_t_set.sort()

        print("Mean CI:")
        print(sample_mean - mean_t_set[higher_bound] * sample_std / (ret_len ** 0.5), 
            sample_mean - mean_t_set[lower_bound] * sample_std / (ret_len ** 0.5))
        print("Std CI:")
        print(sample_std - std_t_set[higher_bound] * sample_std / ((2 * ret_len) ** 0.5), 
            sample_std - std_t_set[lower_bound] * sample_std / ((2 * ret_len) ** 0.5))

        self.ci_para_bs_method_t_mean_l = sample_mean - mean_t_set[higher_bound] * sample_std / (ret_len ** 0.5)
        self.ci_para_bs_method_t_mean_h = sample_mean - mean_t_set[lower_bound] * sample_std / (ret_len ** 0.5)
        self.ci_para_bs_method_t_std_l = sample_std - std_t_set[higher_bound] * sample_std / ((2 * ret_len) ** 0.5)
        self.ci_para_bs_method_t_std_h = sample_std - std_t_set[lower_bound] * sample_std / ((2 * ret_len) ** 0.5)

        # parametric bootstrap method - SEboot method
        # calculate SEboot using resampling 
        print("---------1.2.2.2.3 Use parametric bootstrap method - SEboot method----------")
        mean_set.clear()
        var_set.clear()
        std_set.clear()
        mean_t_set.clear()
        std_t_set.clear()
        var_t_set.clear()

        para_x_samples = sample_std * matlib.randn((B, ret_len)) + sample_mean

        mean_set = [np.mean(i) for i in para_x_samples]
        std_set = [np.std(i) for i in para_x_samples]
        var_set = [np.var(i) for i in para_x_samples]

        se_mean = (sum([(i - np.mean(mean_set)) ** 2 for i in mean_set]) / (B - 1)) ** 0.5
        se_std = (sum([(i - np.mean(std_set)) ** 2 for i in std_set]) / (B - 1)) ** 0.5
        se_var = (sum([(i - np.mean(var_set)) ** 2 for i in var_set]) / (B - 1)) ** 0.5

        print("----置信区间 95% using +/- 2 * SE")
        print("Mean CI:")
        print(sample_mean - 2 * se_mean, sample_mean + 2 * se_mean)
        print("Std CI:")
        print(sample_std - 2 * se_std, sample_std + 2 * se_std)
        print("Variance CI:")
        print(sample_var - 2 * se_var, sample_var + 2 * se_var)

        self.ci_para_bs_method_se_mean_l = sample_mean - 2 * se_mean
        self.ci_para_bs_method_se_mean_h = sample_mean + 2 * se_mean
        self.ci_para_bs_method_se_std_l = sample_std - 2 * se_std
        self.ci_para_bs_method_se_std_h = sample_std + 2 * se_std
        self.ci_para_bs_method_se_var_l = sample_var - 2 * se_var
        self.ci_para_bs_method_se_var_h = sample_var + 2 * se_var

    def output_result_to_excel(self):

        statistics = pd.DataFrame(index=['stats'])
        statistics['single_stock_name'] = self.single_stock_name
        statistics['cc_return_single_stock_mean'] = self.cc_return_single_stock_mean
        statistics['cc_return_single_stock_std'] = self.cc_return_single_stock_std
        statistics['cc_return_single_stock_var'] = self.cc_return_single_stock_var

        statistics['ci_classical_method_mean_l'] = self.ci_classical_method_mean_l
        statistics['ci_classical_method_mean_h'] = self.ci_classical_method_mean_h
        statistics['ci_classical_method_std_l'] = self.ci_classical_method_std_l
        statistics['ci_classical_method_std_h'] = self.ci_classical_method_std_h

        statistics['ci_non_para_bs_method_percentile_mean_l'] = self.ci_non_para_bs_method_percentile_mean_l
        statistics['ci_non_para_bs_method_percentile_mean_h'] = self.ci_non_para_bs_method_percentile_mean_h
        statistics['ci_non_para_bs_method_percentile_std_l'] = self.ci_non_para_bs_method_percentile_std_l
        statistics['ci_non_para_bs_method_percentile_std_h'] = self.ci_non_para_bs_method_percentile_std_h
        statistics['ci_non_para_bs_method_percentile_var_l'] = self.ci_non_para_bs_method_percentile_var_l
        statistics['ci_non_para_bs_method_percentile_var_l'] = self.ci_non_para_bs_method_percentile_var_l

        statistics['ci_non_para_bs_method_t_mean_l'] = self.ci_non_para_bs_method_t_mean_l
        statistics['ci_non_para_bs_method_t_mean_h'] = self.ci_non_para_bs_method_t_mean_h
        statistics['ci_non_para_bs_method_t_std_l'] = self.ci_non_para_bs_method_t_std_l
        statistics['ci_non_para_bs_method_t_std_h'] = self.ci_non_para_bs_method_t_std_h

        statistics['ci_para_bs_method_percentile_mean_l'] = self.ci_para_bs_method_percentile_mean_l
        statistics['ci_para_bs_method_percentile_mean_h'] = self.ci_para_bs_method_percentile_mean_h
        statistics['ci_para_bs_method_percentile_std_l'] = self.ci_para_bs_method_percentile_std_l
        statistics['ci_para_bs_method_percentile_std_h'] = self.ci_para_bs_method_percentile_std_h
        statistics['ci_para_bs_method_percentile_var_l'] = self.ci_para_bs_method_percentile_var_l
        statistics['ci_para_bs_method_percentile_var_h'] = self.ci_para_bs_method_percentile_var_h

        statistics['ci_para_bs_method_t_mean_l'] = self.ci_para_bs_method_t_mean_l
        statistics['ci_para_bs_method_t_mean_h'] = self.ci_para_bs_method_t_mean_h
        statistics['ci_para_bs_method_t_std_l'] = self.ci_para_bs_method_t_std_l
        statistics['ci_para_bs_method_t_std_h'] = self.ci_para_bs_method_t_std_h

        statistics['ci_para_bs_method_se_mean_l'] = self.ci_para_bs_method_se_mean_l
        statistics['ci_para_bs_method_se_mean_h'] = self.ci_para_bs_method_se_mean_h
        statistics['ci_para_bs_method_se_std_l'] = self.ci_para_bs_method_se_std_l
        statistics['ci_para_bs_method_se_std_h'] = self.ci_para_bs_method_se_std_h
        statistics['ci_para_bs_method_se_var_l'] = self.ci_para_bs_method_se_var_l
        statistics['ci_para_bs_method_se_var_h'] = self.ci_para_bs_method_se_var_h

        return statistics.T