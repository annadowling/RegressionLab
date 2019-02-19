import pandas as pd
import scipy.stats as scs

def chi_square_of_df_cols(df, col1, col2):
    return scs.chi2_contingency(pd.crosstab(df[col2], df[col1]))


def TestIndependence(df ,colX ,colY):
    X = df[colX].astype(str)
    Y = df[colY].astype(str)

    dfObserved = pd.crosstab(Y,X)
    chi2, p, dof, expected = scs.chi2_contingency(dfObserved.values)
    print("chi2")
    print(chi2)
    print("p-value")
    print(p) # p-value
    print("Degree of Freedom") # The number of independent variates which make up the statistic (eg chi-square) is known as degree of freedom of that statistic.
    print(dof)
    print("Expected Array")
    print(expected) # expected array

carData = pd.read_csv('../data/cleaned-outliers-auto-mpg.csv')

TestIndependence(carData, 'model year', 'cylinders')
TestIndependence(carData, 'Europe', 'cylinders')
TestIndependence(carData, 'Asia', 'cylinders')
TestIndependence(carData, 'USA', 'cylinders')



