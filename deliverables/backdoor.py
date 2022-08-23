from unicodedata import decimal

from sklearn.ensemble import RandomForestRegressor
from file_organize import dataFrameFinal
import statsmodels.api as sm
import pandas as pd
import numpy as np

def backdoor_mean(Y, A, Z, value, data):
    """
    Compute the counterfactual mean E[Y(a)] for a given value of a via backdoor adjustment
    
    Inputs
    ------
    Y: string corresponding variable name of the outcome
    A: string corresponding variable name
    Z: list of variable names to adjust for
    value: float corresponding value to set A to
    data: pandas dataframe
    
    Return
    ------
    ACE: float corresponding to the causal effect
    """
    
    formula = Y + "~" + A
    if len(Z) > 0:
        formula += " + " + "+".join(Z)
    
    model = sm.GLM.from_formula(formula=formula, data=data, family=sm.families.Gaussian()).fit()
    data_a = data.copy()
    data_a[A] = value
    return np.mean(model.predict(data_a))
    
def backdoor_ML(Y, A, Z, data):
    """
    Compute the counterfactual mean E[Y(a)] for a given value of a via backdoor adjustment
    
    Inputs
    ------
    Y: string corresponding variable name of the outcome
    A: string corresponding variable name
    Z: list of variable names to adjust for
    data: pandas dataframe
    
    Return
    ------
    ACE: float corresponding to the causal effect
    """
    
    formula = Y + "~" + A
    if len(Z) > 0:
        formula += " + " + "+".join(Z)
    
    outcome = get_numpy_matrix(data, [Y])
    predictors = get_numpy_matrix(data, [A] + Z)

    model = RandomForestRegressor(bootstrap=False)
    model.fit(predictors, outcome)
    data_A0 = data.copy()
    data_A0[A] = 0

    model = RandomForestRegressor(bootstrap=False)
    model.fit(predictors, outcome)
    data_A1 = data.copy()
    data_A1[A] = 1

    prediction_A0 = get_numpy_matrix(data_A0, [A] + Z)
    prediction_A0 = model.predict(prediction_A0)

    prediction_A1 = get_numpy_matrix(data_A1, [A] + Z)
    prediction_A1 = model.predict(prediction_A1)
    
    return np.mean(prediction_A1) - np.mean(prediction_A0)
    
def get_numpy_matrix(data, variables):
    """
    Takes a pandas dataframe and a list of variable names, and returns
    just the raw matrix for those specific variables
    """
    
    matrix = data[variables].to_numpy()

    # if there's only one variable, ensure we return a matrix with one column
    # rather than just a column vector
    if len(variables) == 1:
        return matrix.reshape(len(data),)
    return matrix

def compute_confidence_intervals_MLbackdoor(Y, A, Z, data, method_name, num_bootstraps=100, alpha=0.05, value=None):
    """
    Compute confidence intervals for backdoor adjustment via bootstrap
    
    Returns tuple (q_low, q_up) for the lower and upper quantiles of the confidence interval.
    """
    
    Ql = alpha/2
    Qu = 1 - alpha/2
    estimates = []
    
    for i in range(num_bootstraps):
        
        # resample the data with replacement
        data_sampled = data.sample(len(data), replace=True)
        data_sampled.reset_index(drop=True, inplace=True)
        
        # add estimate from resampled data
        if method_name == "backdoor_mean":
            estimates.append(backdoor_mean(Y, A, Z, value, data_sampled))
        if method_name == "backdoor_ML":
            estimates.append(backdoor_ML(Y, A, Z, data_sampled))

        else:
            print("Invalid method")
            estimates.append(1)

    # calculate the quantiles
    
    quantiles = np.quantile(estimates, q=[Ql, Qu])
    q_low = quantiles[0]
    q_up = quantiles[1]
    print("BACKDOOR_ML AVERAGE CAUSAL EFFECT FOR " + str(A) + ":", np.mean(estimates), "("+ str(q_low) + ", " + str(q_up) + ")")
    return q_low, q_up

def backdoor_adjustment(Y, A, Z, data):
    """
    Compute the average causal effect E[Y(A=1)] - E[Y(A=0)] via backdoor adjustment
    
    Inputs
    ------
    Y: string corresponding variable name of the outcome
    A: string corresponding variable name
    Z: list of variable names to adjust for
    data: pandas dataframe
    
    Return
    ------
    ACE: float corresponding to the causal effect
    """
    
    formula = Y + "~" + A
    if len(Z) > 0:
        formula += " + " + "+".join(Z)
    
    model = sm.GLM.from_formula(formula=formula, data=data, family=sm.families.Gaussian()).fit()
    data_A0 = data.copy()
    data_A1 = data.copy()
    data_A0[A] = 0
    data_A1[A] = 1
    return(np.mean(model.predict(data_A1)) - np.mean(model.predict(data_A0)))

def compute_confidence_intervals_backdoor(Y, A, Z, data, num_bootstraps=200, alpha=0.05):
    """
    Compute confidence intervals for backdoor adjustment via bootstrap

    Returns tuple (q_low, q_up) for the lower and upper quantiles of the confidence interval.
    """

    Ql = alpha/2
    Qu = 1 - alpha/2
    estimates = []

    for i in range(num_bootstraps):
        new = data.sample(n = len(data.index), replace = True)
        estimates.append(backdoor_adjustment(Y, A, Z, new))

    q_low = np.quantile(estimates, Ql)
    q_up = np.quantile(estimates, Qu)
    print("BACKDOOR AVERAGE CAUSAL EFFECT FOR " + str(A) + ":", np.mean(estimates), "("+ str(q_low) + ", " + str(q_up) + ")")
    return q_low, q_up


def main():
    mainInfo = dataFrameFinal.gatherData()
    columns = ["BROOKLYN", "MANHATTAN", "STATEN_ISLAND", "BRONX", "QUEENS"]
    additional = ["PERCENT_WHITE", 'POPULATION_DENSITY']

    for x in columns:
        backdoor_set = columns.copy()
        backdoor_set.remove(x)
        backdoor_set = backdoor_set + additional
        compute_confidence_intervals_backdoor("SALE_PRICE", x, backdoor_set, mainInfo)


    for x in columns:
        backdoor_set = columns.copy()
        backdoor_set.remove(x)
        backdoor_set = backdoor_set + additional
        compute_confidence_intervals_MLbackdoor("SALE_PRICE", x, backdoor_set, mainInfo, method_name="backdoor_ML")

if __name__ == "__main__":
    main()
