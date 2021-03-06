import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

def estimate_coefficients(x, y):
    # size of the dataset OR number of observations/points
    n = np.size(x)

    # mean of x and y
    # Since we are using numpy just calling mean on numpy is sufficient
    mean_x, mean_y = np.mean(x), np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x - n*mean_y*mean_x)
    SS_xx = np.sum(x*x - n*mean_x*mean_x)

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = mean_y - b_1*mean_x

    return(b_0, b_1)

    # x,y are the location of points on graph
    # color of the points change it to red blue orange play around



def plot_regression_line(x, y, b, xlabel, ylabel):
    # plotting the points as per dataset on a graph
    plt.scatter(x, y, color = "m",marker = "o", s = 30)

    # predicted response vector
    y_pred = b[0] + b[1]*x

    # plotting the regression line
    plt.plot(x, y_pred, color = "g")

    # putting labels for x and y axis
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # function to show plotted graph
    plt.show()


def predict_with_intercept(X, y):
    # Note the difference in argument order
    X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

    # Note the difference in argument order
    model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
    predictions = model.predict(X)
    print("Predictions: ")
    print(predictions[0:5])

    # Print out the statistics
    print(model.summary())



def main(X, Y):
    df = pd.read_csv('../data/cleaned-outliers-auto-mpg.csv')

    x = df[X]
    y = df[Y] # Y is the variable we are trying to predict

    # estimating coefficients
    b = estimate_coefficients(x, y)
    print("Estimated coefficients:\nb_0 = {} \nb_1 = {}".format(b[0], b[1]))

    predict_with_intercept(x, y)


if __name__ == "__main__":
    main(['displacement', 'USA'], 'mpg')
