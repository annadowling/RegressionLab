import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
'exec(%matplotlib inline)'
matplotlib.style.use('ggplot')
from scipy import stats
from sklearn.linear_model import LinearRegression


carData = pd.read_csv('../data/cleaned-auto-mpg.csv')