import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt 
import random
import numpy as np
from numpy import matlib
import scipy.optimize as sc_optim
from CERModel import CERModel
from Single_Index_Model import Single_Index_Model
from portfolio import portfolio

if __name__ == "__main__":

    print("##### Single Stock Analysis ######")

    ### Mean-Variance model (CER Model)
    print("-------------------1. Use Mean-Variance (CER Model) model-----------------")
    cer_model = CERModel()
    # cer_model.print_data()

    ## Point estimation: Get mean, var, std
    print("-----------1.1 Point estimation-----------")
    cer_model.point_estimation()

    ## Confidence interval
    print("-----------1.2 Confidence interval-----------")
    cer_model.confidence_interval()

    ## Output all results generated into excel file
    cer_statistics = cer_model.output_result_to_excel()
    with pd.ExcelWriter("result/cer_model_result.xlsx") as writer:
        cer_statistics.to_excel(writer, sheet_name='statistics')

    ### Single Index Model
    print("-------------------2. Use Single Index Model-----------------")
    si_model = Single_Index_Model()
            
    ## Point estimation: Get alpha & beta
    print("-----------2.1 Point estimation-----------")
    si_model.point_estimation()
    
    ## Confidence interval
    print("-----------2.2 Confidence interval-----------")
    si_model.confidence_interval()

    ## Output all results generated into excel file
    si_statistics = si_model.output_result_to_excel()
    with pd.ExcelWriter("result/si_model_result.xlsx") as writer:
        si_statistics.to_excel(writer, sheet_name='statistics')

    print("##### Portfolio ######")
    ## Use portfolio to reduce the risk.
    ## For 2 stock, we can calculate the proportion of each stock to get a minimum risk
    ## The minimum risk is min(Var(sigma(X) + (1-sigma)Y))

    print("-------------------3. Two Stock Portfolio-----------------")
    port = portfolio()
    port.calculate_2_stock_weight()
    port_statistics = port.output_result_to_excel()
    with pd.ExcelWriter("result/two_stock_portfolio_result.xlsx") as writer:
        port_statistics.to_excel(writer, sheet_name='statistics')

    



    


    








