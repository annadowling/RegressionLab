'exec(%matplotlib inline)'
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import scipy.stats as sts

# Which numeric variables could come from a normal distribution

# https://medium.com/@rrfd/testing-for-normality-applications-with-python-6bf06ed646a9

# https://www.google.com/search?q=shapiro+wilk+test+python&oq=shapiro+wilk+test+python&aqs=chrome..69i57j0.10685j0j7&sourceid=chrome&ie=UTF-8

# https://www.programcreek.com/python/example/100330/scipy.stats.shapiro

# https://community.periscopedata.com/t/18bzry/test-for-normal-distribution-of-data-with-python

carData = pd.read_csv('../data/regression_lab.csv')
# Check for nulls
print(carData.isnull().any())

def cleanUpHPData(cardata):
    print("Cleaning up horsepower data")
    cardata["horsepower"] = pd.to_numeric(cardata["horsepower"], errors='coerce')

    cardata.dropna()
    return cardata;

# Shapiro-Wilk test
def assert_normality_shapiro(data, title):
    print("Shapiro-Wilk Test for:  " + title)
    statistic, pvalue = sts.shapiro(data)
    print("Shapiro Statistic " + str(statistic) + " and p-value " + str(pvalue))
    if pvalue > 0.05:
        print("Normal")
        return True
    else:
        print("Not normal")
        return False

# Kolmogorov-Smirnov test
def assert_normality_KS(data, title):
    print("Kolmogorov-Smirnov Test for:  " + title)
    ks_results = scipy.stats.kstest(data, cdf='norm')
    print(ks_results)


# Anderson-Darling test
def assert_normality_AD(data):
    anderson_results = scipy.stats.anderson(data)
    print(anderson_results)


# Q-Q Plot Visual Test
def confirm_normality_qq(x):
    sts.probplot(x, dist='norm', plot=plt)
    plt.show()


# From the previous stage it was noticed that to work with horsepower the data needed to be cleaned up.
carData = cleanUpHPData(carData)

assert_normality_shapiro(carData["weight"], "weight")
assert_normality_shapiro(carData["acceleration"], "acceleration")
assert_normality_shapiro(carData["displacement"], "displacement")
assert_normality_shapiro(carData["mpg"], "mpg")
assert_normality_shapiro(carData["horsepower"], "horsepower")

assert_normality_KS(carData["weight"], "weight")
assert_normality_KS(carData["acceleration"], "acceleration")
assert_normality_KS(carData["displacement"], "displacement")
assert_normality_KS(carData["mpg"], "mpg")
assert_normality_KS(carData["horsepower"], "horsepower")

assert_normality_AD(carData["weight"])
assert_normality_AD(carData["acceleration"])
assert_normality_AD(carData["displacement"])
assert_normality_AD(carData["mpg"])
assert_normality_AD(carData["horsepower"])

confirm_normality_qq(carData["weight"])
confirm_normality_qq(carData["acceleration"])
confirm_normality_qq(carData["displacement"])
confirm_normality_qq(carData["mpg"])
confirm_normality_qq(carData["horsepower"])







