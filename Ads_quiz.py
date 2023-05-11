# -*- coding: utf-8 -*-
"""
Created on Sun May  7 00:47:10 2023

@author: DELL
"""
#import modules
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from sklearn import cluster
import sklearn.metrics as skmet
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
import cluster_tools as ct

# define Funtions


def world_bank_data(csv_file):
    """
    Reads in data from a World Bank data file in CSV format.

    Args:
        csv_file (str): The name of the file to read.

    Returns:
        A pandas DataFrame containing the data from the file.
    """
    # Read in the CSV file using pandas
    df_data = pd.read_csv(csv_file, header=2)
    return df_data


def process_world_bank_data(df_data):
    """
    This function takes a pandas dataframe containing world bank data as\n" 
    "input and performs the following operations:
    1. Filters the dataframe to select only the\n" 
    "CO2 emissions (metric tons per capita)" indicator.
    2. Removes columns with all NaN values.
    3. Drops the "Country Code", "Indicator Name", and "Indicator Code" columns.
    4. Sets the "Country Name" column as the index.
    5. Transposes the dataframe.
    6. Selects data for the United Kingdom, Belgium, Brazil, China, India,\n"
    " United States, and World for the years 2000-2014.
    7. Transposes the resulting dataframe again.

    Args:
        df (pandas.DataFrame): Input dataframe containing world bank data.

    Returns:
        pandas.DataFrame: Processed dataframe containing the selected data\n"
        " for the specified countries and years.
    """
    # Filter the dataframe to select only the "CO2 emissions (metric tons per capita)" indicator.
    df_data = df_data[df_data["Indicator Name"] ==
                      "CO2 emissions (metric tons per capita)"]
    # Remove columns with all NaN values.
    df_data = df_data.dropna(axis=1, how='all')
    # Drop the "Country Code", "Indicator Name", and "Indicator Code" columns.
    df_data = df_data.drop(
        ["Country Code", "Indicator Name", "Indicator Code"], axis=1)
    # Set the "Country Name" column as the index.
    df_data = df_data.set_index("Country Name").T
    # Transpose the dataframe.
    df_new = df_data.transpose()
    # Select data for the specified countries and years.
    df_new = df_new.loc[['United Kingdom', 'Belgium',
                         'Brazil', 'China', 'India', 'United States', 'World']]
    # Transpose the resulting dataframe again.
    df_new = df_new.transpose()
    return df_new

def normalize_dataframe():
    """ Normalizes all columns in the dataframe to the 0-1 range.

    Args:
        df (pandas.DataFrame): Input dataframe to be normalized.

    Returns:
        pandas.DataFrame: Normalized dataframe.
        pandas.Series: Minimum values of the original dataframe.
        pandas.Series: Maximum values of the original dataframe.
    """
    df_min = df_new.min()
    df_max = df_new.max()
    df_norm = (df_new - df_min) / (df_max - df_min)
    return df_norm, df_min, df_max


# Read in the data file
df_data = world_bank_data('API_19_DS2_en_csv_v2_5361599.csv')

# Process the data
df_new = process_world_bank_data(df_data)

# Print the processed data
print(df_new)
print(df_new.describe())

# show the heatmap
print(ct.map_corr(df_new))

corr = df_new.corr()

# Find lowest correlation coefficient
min_corr = corr.min().min()

# Find the columns with the lowest correlation coefficient
low_corr_cols = corr[corr == min_corr].stack().index.tolist()

# Print the result
print('Columns with the lowest correlation coefficient:')
print(low_corr_cols)

# United Kingdom v indian has the lowest correlation coefficent
# copy the dataframe to prevent changes of the original dataframe(df_new)
df_fit = df_new[['United Kingdom', 'India']].copy()

df_fit, df_min, df_max = ct.scaler(df_fit)
print(df_fit.describe())

# Normalize the dataframe

data_norm = StandardScaler().fit_transform(df_fit)

silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=20)
    kmeans.fit(data_norm)
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_
    score = skmet.silhouette_score(data_norm, labels)
    silhouette_scores.append(score)
    print(f"Silhouette score for {k} clusters: {score}")


# Add the cluster labels as a new column to the dataframe
df_new['Cluster'] = kmeans.labels_

#data_orig = df_fit * (df_max - df_min) + df_min
print(df_new)
print(kmeans.cluster_centers_)


# Plot 3 clusters
Nc = 3  # number of cluster centres

kmeans = cluster.KMeans(n_clusters=Nc)
kmeans.fit(df_fit)

# extract labels and cluster centres
labels = kmeans.labels_
cen = kmeans.cluster_centers_

plt.figure(figsize=(6.0, 6.0))
# scatter plot with colours selected using the cluster numbers
plt.scatter(df_new["United Kingdom"], df_new["India"], c=labels, cmap="tab10")

# show cluster centres
center = ct.backscale(cen, df_min, df_max)
xc = center[:, 0]
yc = center[:, 1]
plt.scatter(xc, yc, c="b", marker="H", s=200)
# c = colour, s = size

plt.xlabel("United Kingdom")
plt.ylabel("India")
plt.title("Fig:2. 3 clusters")
plt.show()

# Creation of simple model(s) fitting data sets with curve_fit.

# Filter the rows with the specified country and indicators and drop irrelevant columns
df_t = df_data[(df_data["Country Name"] == "United Kingdom") & (
    df_data["Indicator Name"].isin(["Urban population", "Urban population growth (annual %)"]))]

# Convert "Unnamed: 66" column to float and drop irrelevant columns
df_t['Unnamed: 66'] = df_t['Unnamed: 66'].astype(float)
df_t.drop(['Country Name', 'Country Code', 'Indicator Name',
          'Indicator Code', 'Unnamed: 66'], axis=1, inplace=True)

# Transpose the filtered dataframe and rename the columns
df_curve = df_t.transpose()
df_curve = df_curve.rename(
    columns={6157: "Urban population", 6158: "Urban population growth (annual %)"})

# Reset the index of the transposed dataframe and rename the columns
df_curve = df_curve.reset_index()
df_curve = df_curve.dropna()
df_curve.columns = ['Date', 'Urban population',
                    'Urban population growth (annual %)']
df_curve['Urban population'] = df_curve['Urban population'].astype(int)

df_curve['Date'] = df_curve['Date'].astype(int)

# initial quess
p0 = [50000000, 0.6, 1960]

# define exponential function


def exp_func(x, a, b, c):
    """Computes exponential function with scale and growth as free parameters
    """
    return a * np.exp(b * (x - c))


xdata = df_curve['Date']
ydata = df_curve['Urban population']

popt, pcov = curve_fit(exp_func, xdata, ydata, p0=p0)

# Print the fit parameters
print('Fit parameters:')
print('a =', popt[0])
print('b =', popt[1])
print('c =', popt[2])

# Plot the data and the fit
plt.plot(xdata, ydata, '--', label='Data')
plt.plot(xdata, exp_func(xdata, *popt), 'r-', label='Fit')
plt.legend()
plt.xlabel('Year')
plt.ylabel('Urban Population (millions)')
plt.title('Fig:3. Exponential Fit of Urban Population')
plt.show()

y_pred = exp_func(2040, *popt)
print(y_pred)

# define err_ranges


def err_ranges():
    """
    Calculates the confidence intervals for the fit parameters.

    Parameters:
    
    func (function): The function used to fit the data.

    Returns:
    confs (array): The confidence intervals for the fit parameters.
    """

    perr = np.sqrt(np.diag(pcov))

    n = len(ydata)    # number of data points

    tval = 2.306      # t-value for 95% confidence interval with degrees of freedom
    confs = tval * perr * \
        np.sqrt(1/n + (x_pred - np.mean(xdata))**2 /
                np.sum((xdata - np.mean(xdata))**2))
    return confs


x_pred = np.array([2040])
y_pred = exp_func(x_pred, *popt)
confs = err_ranges()
print(
    f"Predicted urban population in 2040: {y_pred[0]:.0f} +/- {confs[0]:.0f}")
