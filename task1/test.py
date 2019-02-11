import pandas as pd
import numpy as np
'exec(%matplotlib inline)'
import matplotlib.pyplot as plt
import seaborn as sn
import scipy.stats as sts
import statsmodels.api as sm


carData = pd.read_csv('./regression_lab.csv')

print("---------- Basic Car Info ----------");

print(carData.info())

print("---------- Describe mpg, displacement, horsepower, weight, acceleration ----------");

print(carData[["mpg", "displacement","horsepower","weight","acceleration"]].describe())

print("---------- Horsepower Column Refactor ----------");
### horsepower is an object type. Which means it has some non numeric characters.
print("---------- Horsepower First Check ----------");
print(carData["horsepower"].isnull().values.any())

carData["horsepower"] = pd.to_numeric( carData["horsepower"], errors = 'corece' )

print("---------- Horsepower Second Check Post CleanUp ----------");
print(carData["horsepower"].isnull().values.any())

carData = carData.dropna()

print("---------- Re-Describe mpg, displacement, horsepower, weight, acceleration following Clean Up ----------");

print(carData[["mpg", "displacement","horsepower","weight","acceleration"]].describe())

# Average mpg by cylinders
mpg_cylinders_df = carData.groupby('cylinders')['mpg'].mean().reset_index()

sn.barplot( y = 'mpg',
            x = 'cylinders',
            data = mpg_cylinders_df )
plt.show()

sn.barplot( y = 'mpg',
            x = 'cylinders',
            data = mpg_cylinders_df,
            order = mpg_cylinders_df.sort_values('mpg')['cylinders'])

plt.show()

mpg_cylinders_origin_df = carData.groupby(['cylinders', 'origin'])['mpg'].mean().reset_index()

print("---------- Average MPG by Cylinders & Origin ----------");
print(mpg_cylinders_origin_df)

# Average mpg by cyinders and grouped by origin
sn.barplot( y = 'mpg',
            x = 'cylinders',
            data = mpg_cylinders_origin_df,
            hue = 'origin');
plt.show()

# Trend in average MPG by year for different origin cars
mpg_year_origin_df = carData.groupby(['model year', 'origin'])['mpg'].mean().reset_index()

sn.catplot( x = 'model year', y = 'mpg', hue = 'origin', kind = 'point', data = mpg_year_origin_df, height = 6 )
plt.show()

# Creating a histogram - Distribution of mpg (miles per gallon)
plt.hist( carData.mpg )
plt.hist( carData.mpg, bins = 50 );
plt.hist( carData.acceleration );
plt.hist( carData.weight );

#sn.distplot( carData[carData.mpg] )
#plt.show()

# Comparing mpg distributions of cars by different origins
sn.distplot( carData[carData.origin == 1].mpg, hist = False, label= 'American' )
sn.distplot( carData[carData.origin == 2].mpg, hist = False, label= 'European' )
sn.distplot( carData[carData.origin == 3].mpg, hist = False, label= 'Japaneese' )
plt.show()

# Calculating Statistics
mpg_desc = carData.mpg.describe()

print("---------- Mean of MPG ----------");
print(mpg_desc['mean'])

sn.set(rc={"figure.figsize": (8, 6)});

# Distribution of mpg for all cars¶
sn.boxplot( y = carData.mpg )
plt.show()

# MPG distribution for different number of cylinders
sn.boxplot( x = carData.cylinders,
            y = carData.mpg,
            order = carData.cylinders.unique().sort() )
plt.title( "Box plots for various cylinder counts")
plt.show()


# Finding Outliers for 6 Cylinder Cars

carData[carData.cylinders == 6].mpg.quantile( 0.75 )
print(sts.iqr( carData[carData.cylinders == 6].mpg ))

# Extreme outliers: 0.75 percetile + 3 * iqr

outlier = carData[carData.cylinders == 6].mpg.quantile( 0.75 ) + 3 * sts.iqr( carData[carData.cylinders == 6].mpg )

print(outlier)

print(carData[carData.cylinders == 6][ carData[carData.cylinders == 6].mpg > outlier ])

# Creating scatter plots - weight vs. mpg
plt.scatter( carData.weight, carData.mpg );
plt.scatter( carData.weight, carData.mpg )
plt.title("Autos Mpg Vs. Weight")
plt.xlabel('weight', fontsize=18)
plt.ylabel('mpg', fontsize=16)
plt.savefig('test.png')

# Joint Plots
sn.jointplot( carData.mpg, carData.weight, height = 6 );
plt.show()

# Multivariate distribution plot
sn.jointplot(x="mpg", y="acceleration", data=carData, kind="kde");
plt.show()

# Weight vs. mpg for different number of cylinders in cars¶
sn.lmplot(x="weight", y="mpg", data= carData, fit_reg = True, height = 6 )
plt.show()

# Visualizing the correlation between more than 2 variables, pair-wise at same time
car_stats = carData[['mpg', 'displacement',
                     'weight', 'acceleration']]
plt.figure( figsize = (6,6));
sn.pairplot( car_stats );
plt.show()

# Finding correlations
print(car_stats.corr())

# Creating a heatmap to depict correlations
sn.heatmap( car_stats.corr(), annot = True )
plt.show()

# Comparing distribution of variables together
autos_subset_df = carData[['mpg', 'displacement', 'weight', 'acceleration', 'origin']]
auto_melt_df = pd.melt(autos_subset_df, "origin", var_name="measures")
print(auto_melt_df.sample( 10 ))

sn.violinplot(x="measures", y="value", hue="origin", data=auto_melt_df)
plt.show()

autos_subset_df = carData[['mpg', 'displacement', 'weight', 'acceleration', 'origin']]
autos_subset_df = autos_subset_df.apply(lambda x:(x.astype(float) - min(x))/(max(x)-min(x)), axis = 0)
auto_melt_df = pd.melt(autos_subset_df, "origin", var_name="measures")
sn.swarmplot(x="measures", y="value", hue="origin", data=auto_melt_df);
plt.show()

#Does america cars have different mpg than japaneese cars?
#Hypothesis Test: 1
#Null Hypothesis: average mpg for american cars = average mpg for japaneese cars.
#Alternative Hypothesis: average mpg for amarican cars <> average mpg for japaneese cars.

print(stats.ttest_ind( carData[ carData.origin == 1]["mpg"],
                       carData[ carData.origin == 3 ]["mpg"],
                       equal_var=True))

# Conclusion: As p-value is less than 0.05. Yes, average mpg for american cars are different than average mpg for japaneese cars


















