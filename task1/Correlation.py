import pandas as pd
import numpy as np
'exec(%matplotlib inline)'
import matplotlib.pyplot as plt
import seaborn as sn
from scipy import stats


# Normality	of	X	and	Y;
# “Linear”	dependency;
# Homogeneity	of	Variance:
# Homoscedasticity; Levene’s	test:	scipy.stats.levene(…)


carData = pd.read_csv('../data/regression_lab.csv')


def perform_correlation(col1, col2):
    pearson_coef, p_value = stats.pearsonr(col1, col2)
    print("Pearson Correlation Coefficient: ", pearson_coef, "and a P-value of:", p_value)


perform_correlation(carData['mpg'], carData['acceleration'])
perform_correlation(carData['mpg'], carData['weight'])
perform_correlation(carData['mpg'], carData['displacement'])
horsepower = pd.to_numeric(carData["horsepower"], errors='coerce')
horsepower.dropna()
perform_correlation(carData['mpg'], horsepower)