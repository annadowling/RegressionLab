import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
'exec(%matplotlib inline)'
matplotlib.style.use('ggplot')
from scipy import stats
from sklearn.linear_model import LinearRegression


carData = pd.read_csv('../data/cleaned-auto-mpg.csv')
linearRegressor = LinearRegression()


def check_assumptions(x, y, data):
    print(data[[x, y]].describe())

    # Linearity
    data.plot.scatter(x, y)
    plt.show()
    # Homogeneity of Variance
    print(stats.levene(data[x], data[y]))


def check_correlation(x, y, data):
    # Pearson correlation
    pearson_coef, p_value = stats.pearsonr(data[x], data[y])
    print("Pearson Correlation Coefficient: ", pearson_coef, "and a P-value of:", p_value)


    # Pandas correlation of two columns
    print("Pandas Correlation")
    print(data[x].corr(data[y]))


check_assumptions("weight", "displacement", carData)
check_correlation("weight", "displacement", carData)





