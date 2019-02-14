# Importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


carData = pd.read_csv('../data/cleaned-auto-mpg.csv')


def check_assumptions(x, y, data):
    print(data[[x, y]].describe())

    # Checking the assumptions
    data.plot.scatter(x, y)
    plt.show()
    # Homogeneity of Variance
    print(stats.levene(data[x], data[y]))


def check_correlation(x, y, data):
    # Pearson correlation
    print(stats.pearsonr(data[x], data[y]))

    # Pandas correlation of two columns
    print(data[x].corr(data[y]))


check_assumptions("mpg", "weight", carData)
check_correlation("mpg", "weight", carData)





