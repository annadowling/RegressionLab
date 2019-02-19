import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
'exec(%matplotlib inline)'
matplotlib.style.use('ggplot')
from scipy import stats
import seaborn as sn

carData = pd.read_csv('../data/cleaned-outliers-auto-mpg.csv')


def plot_all(data, columns):
    autos_stats = data[columns]

    # Linearity
    plt.figure(figsize=(6, 6))
    sn.pairplot(autos_stats)
    plt.show()


def find_correlations(data, dropped_columns):
    print(data.drop(dropped_columns, axis=1).corr(method='pearson'))

    sn.heatmap(data.drop(dropped_columns, axis=1).corr(method='pearson'), annot=True)
    plt.show()


excluded_columns = ['model year', 'USA', 'Europe', 'Asia', 'cylinders', 'car name']
included_columns = ['mpg', 'displacement', 'weight', 'horsepower']
plot_all(carData, included_columns)
find_correlations(carData, excluded_columns)


def check_assumptions(x, comparison_columns, data):
    for y in comparison_columns:
        # Check Linearity
        data.plot.scatter(x, y)
        plt.show()
        # Homogeneity of Variance
        print(stats.levene(data[x], data[y]))

        # Pearson correlation
        check_correlation(data[x], data[y])


def check_correlation(x, y):
    pearson_coef, p_value = stats.pearsonr(x, y)
    print("Pearson Correlation Coefficient: ", pearson_coef, "and a P-value of:", p_value)


check_assumptions('mpg', ['displacement', 'weight', 'horsepower'], carData)
check_assumptions('displacement', ['mpg', 'weight', 'horsepower'], carData)
check_assumptions('weight', ['displacement', 'mpg', 'horsepower'], carData)
check_assumptions('horsepower', ['displacement', 'weight', 'mpg'], carData)

