import pandas as pd
'exec(%matplotlib inline)'
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def detect_outlier(data):
    outliers = []
    threshold = 3
    mean_1 = np.mean(data)
    std_1 = np.std(data)

    for y in data:
        z_score = (y - mean_1)/std_1
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers


def print_outliers(column_names, data):
    for name in column_names:
        print('Outliers for column: ' + name)
        outliers = detect_outlier(data[name])
        print(outliers)


def generate_boxplot(column_names, data):
    flierprops = dict(markerfacecolor='red', markersize=10,
                      linestyle='none')
    colors = sns.color_palette("pastel")
    for name in column_names:
        sns.boxplot(x=data[name], palette=colors, flierprops=flierprops)
        plt.show()


column_names = ['mpg', 'displacement', 'weight', 'acceleration', 'horsepower', 'model year', 'cylinders']
carData = pd.read_csv('../data/cleaned-auto-mpg.csv')

print_outliers(column_names, carData)
generate_boxplot(column_names, carData)




