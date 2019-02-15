import numpy as np;
import pandas as pd;
import statsmodels.api as sm;
import matplotlib.pyplot as plt;

df = pd.read_csv('../data/cleaned-auto-mpg.csv')

X = np.column_stack((df.horsepower,df.weight));
X = sm.add_constant(X);

print("Predictors");
print(X);

print("Outcome");
print(df.mpg);

model = sm.OLS(df.mpg,X);
results = model.fit();

print(results.summary());

plt.figure();
plt.scatter(df.mpg, results.resid);
plt.show();