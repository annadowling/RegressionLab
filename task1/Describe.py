import pandas as pd
'exec(%matplotlib inline)'
import matplotlib.pyplot as plt


def transform_origin(df):
    print("Transform Origin data")
    origin = df.pop('origin')
    df['USA'] = (origin == 1)*1
    df['Europe'] = (origin == 2)*1
    df['Asia'] = (origin == 3)*1


def cleanup_hp_data(df):
    print("Cleaning up horsepower data")
    # transform the horsepower column
    horsepower_missing_ind = df[df.horsepower=='?'].index

    df.loc[horsepower_missing_ind, 'horsepower'] = float('nan')
    df.horsepower = df.horsepower.apply(pd.to_numeric)
    df.loc[horsepower_missing_ind, 'horsepower'] = int(df.horsepower.mean())

    # Drop NaN values from data
    df.dropna()

    return df;


carData = pd.read_csv('../data/regression_lab.csv')

print(carData.head(5))
transform_origin(carData)

print(carData.head(5))

print(carData.dtypes)

# Print the unique values in horsepower column
print(carData.horsepower.unique())
carData = cleanup_hp_data(carData)


# display stats of the features
print(carData.describe())

# display stats of the mpg: So the minimum value is 9 and maximum is 46, but on average it is 23.44 with a variation of 7.8
print(carData.mpg.describe())

byMPG = carData.groupby('model year')['mpg'].mean()
print(byMPG)

carData.hist(bins=50, figsize=(30,20))
plt.savefig("attribute_histogram_plots")
plt.show()

carData.to_csv('../data/cleaned-auto-mpg.csv', index=False)


