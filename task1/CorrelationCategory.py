import pandas as pd
import scipy.stats as scs

carData = pd.read_csv('../data/cleaned-outliers-auto-mpg.csv')


def chi_square_of_df_cols(df, col1, col2):
    return scs.chi2_contingency(pd.crosstab(df[col2], df[col1]))


print(chi_square_of_df_cols(carData, 'model year', 'cylinders'))



