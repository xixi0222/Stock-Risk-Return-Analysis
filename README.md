# Stock-Risk-Return-Analysis

#### Brief Intruduction:
1. Implement CER Model & Single Index Model with Point Estimation &amp; Interval Estimation.  
2. Use Classical method and Non-parametric Bootstrap (Percentile method and T method).  
3. Solve the best portfolio for 2 stocks for minimum variance. 

*Data source: Wind*

#### Model Explanation

1. CER Model:   
	* The CER model assumes that return of an asset over time is independent and identically normally distributed with a constant (time invariant) mean and variance. The model allows for the returns on different assets to be contemporaneously correlated but that the correlations are constant over time.
	* Point Estimation: Mean, Variance & Standard deviation.
	* Interval estimation: It shows the confidence interval of statistics (Mean, Variance & Standard deviation) at a confidence level.
2. Single Index Model: 
	* It shows that the stock return is influenced by the market (beta), has a firm specific expected value (alpha) and firm-specific unexpected component (residual). Each stock's performance is in relation to the performance of a market index. Security analysts often use the SIM for such functions as computing stock betas, evaluating stock selection skills, and conducting event studies.
	* Point Estimation: Alpha & Beta.
	* Interval estimation: It shows the confidence interval of statistics (Alpha & Beta) at a confidence level.
3. Portfolio:
        * Get the portfolio of 2 stocks which has minimum risk indicated by Var\[return of portfolio].

#### Data Description:

1. In this Excel file, I choose 10 stocks with the largest market value. These stocks can better reflect the corporations and economic situations of the US. For the index, Nasdaq, S&P500 and Dow Jones industrial index are chosen.

#### Analysis Structure Tree:

* CER Model
	* Point estimation
		* Classical method
	* Interval estimation
		*  Non-parametric Bootstrap
			* Percentile method
			* T method
		* Parametric Bootstrap
			* Percentile method
			* T method 
			* SEboot method

* Single Index Model
	* Point estimation
		* Classical method
	* Interval estimation
		*  Non-parametric Bootstrap
			* Percentile method
			* T method
		* Parametric Bootstrap
			* Percentile method
			* T method 
			* SEboot method

