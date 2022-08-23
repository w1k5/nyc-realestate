
from fcit import fcit
from statsmodels.formula.api import ols
from file_organize import dataFrameFinal
import numpy as np


def get_numpy_matrix(data, variables):
    """
    Takes a pandas dataframe and a list of variable names, and returns
    just the raw matrix for those specific variables
    """
    matrix = data[variables].to_numpy()

    # if there's only one variable, ensure we return a matrix with one column
    # rather than just a column vector
    if len(variables) == 1:
        return matrix.reshape(len(data),1)
    return matrix



def main():
    data = dataFrameFinal.gatherData()
    data.dropna()
    columns = ["BROOKLYN", "MANHATTAN", "STATEN_ISLAND", "BRONX", "QUEENS"]
    additional = ["PERCENT_WHITE", 'POPULATION_DENSITY']

    for x in columns:
        #testing presence between treatment and each outcome
        predictor = get_numpy_matrix(data, [x])

        backdoor_set = columns.copy()
        backdoor_set.remove(x)
        backdoor_set = backdoor_set + additional
        final_adj = get_numpy_matrix(data, backdoor_set)

        outcome = get_numpy_matrix(data, ["SALE_PRICE"])

        runs = []
        for y in range(5):
            runs.append(fcit.test(predictor, outcome, final_adj, discrete=(True, False)))
        print(str(x) + " HAS A FCIT RESULT OF " + str(np.mean(runs)))


if __name__ == "__main__":
    main()
