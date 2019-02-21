import numpy as np;
import pandas as pd;
import statsmodels.api as sm;
import matplotlib.pyplot as plt;
import scipy.stats;

df = pd.read_csv('../data/cleaned-outliers-auto-mpg.csv')

def reg_results(X, Y = df['mpg']):
    X = sm.add_constant(X);
    model = sm.OLS(Y,X);
    results = model.fit();
    print(results.summary());

def print_header(s):
    print("\n\n\t\t\t%s"%s);

print_header("\tPredictor only model");
reg_results(df['displacement']);

print_header("Predictor and Mediator model");
reg_results(np.column_stack((df['displacement'], df['USA'])));

print_header("\tMediator only model");
reg_results(df['USA']);

print_header("Predictor explains Mediator");
reg_results(df['displacement'], df['USA']);