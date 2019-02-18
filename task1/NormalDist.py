import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def empirical_dist_func(data):
    """Compute ECDF for a one-dimensional array of measurements."""

    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y


def plot_xy(x, y, label):
    plt.figure(figsize=(8,7))
    sns.set()
    plt.plot(x, y, marker=".", linestyle="none")
    plt.xlabel(label)
    plt.ylabel("Cumulative Distribution Function")


def generate_sample(data):
    samples = np.random.normal(np.mean(data), np.std(data), size=1000)
    x_theor, y_theor = empirical_dist_func(samples)

    plt.plot(x_theor, y_theor)
    plt.legend(('Normal Distribution', 'Empirical Data'), loc='lower right')
    plt.show()


def normal_test(data, label):
    print("Normal test for " + label)
    print(stats.normaltest(data))


def execute_normal_distribution_test(column_names, data):
    for name in column_names:
        x, y = empirical_dist_func(data[name])
        plot_xy(x, y, name)

        generate_sample(data[name])

        normal_test(data[name], name)


carData = pd.read_csv('../data/cleaned-outliers-auto-mpg.csv')
numeric_column_names = ['mpg', 'displacement', 'weight', 'acceleration', 'horsepower']

execute_normal_distribution_test(numeric_column_names, carData)
