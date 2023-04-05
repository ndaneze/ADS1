# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 07:17:26 2023

@author: DELL
"""
# import library functions
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# define functions


def world_bank_data(filename):
    """
    Reads in data from a World Bank data file in CSV format.

    Args:
        filename (str): The name of the file to read.

    Returns:
        A pandas DataFrame containing the data from the file.
    """
    # Read in the CSV file using pandas
    df_data = pd.read_csv(filename, header=2)
    # Drop unnecessary columns
    df_data.drop(['Country Code', 'Indicator Code',
                  'Unnamed: 66'], axis=1, inplace=True)
    df_data = df_data.dropna(axis=1, how='all')

    # Transpose the dataframe to create a dataframe with years as columns
    df_t = df_data.set_index(['Country Name', 'Indicator Name']).T

    # Transpose the cleaned dataframe to create a dataframe with countries as columns
    df_c = df_t.transpose()
    df_c = df_c.loc[['United States', 'China',
                     'India', 'Nigeria', 'World'], '2010':'2021']
    indicators = [
        'Population growth (annual %)',
        'Mortality rate, under-5 (per 1,000 live births)',
        'Population, total']
    df_c = df_c.loc[df_c.index.get_level_values(1).isin(indicators)]
    df_t = df_c.transpose()

    return df_t, df_c


# read world bank data csv file into pandas dataframe
df_t, df_c = world_bank_data('API_19_DS2_en_csv_v2_5346672.csv')

# print the transposed data
print(df_t)
# explore the data
print(df_t.head())
print(df_t.describe())

corr_matrix = df_t.corr()
print(corr_matrix)

# extract Mortality rate, under-5 (per 1,000 live births) data from df_t
motality = df_t.loc[:, (slice(
    None), 'Mortality rate, under-5 (per 1,000 live births)')]
rate = motality.transpose()
rate.reset_index(inplace=True)
rate.drop('Indicator Name', axis=1, inplace=True)
mot_rate = rate.set_index('Country Name').transpose()

# extract annual population growth data from df_t
pop = df_t.loc[:, (slice(None), 'Population growth (annual %)')]
growth = pop.transpose()
growth.reset_index(inplace=True)
growth.drop('Indicator Name', axis=1, inplace=True)
pop_growth = growth.set_index('Country Name').transpose()

# extract population total fro the dataframe
pop_t = df_t.loc[:, (slice(None), 'Population, total')]
t = pop_t.transpose()
t.reset_index(inplace=True)
t.drop('Indicator Name', axis=1, inplace=True)
pop_total = t.set_index('Country Name').transpose()

# print the extracted dataframes based on indicators
print(pop_growth)
print(mot_rate)
print(pop_total)

# Line plot of mrtality rate
fig, ax = plt.subplots(figsize=(10, 6))
for country in mot_rate.columns:
    ax.plot(mot_rate.index, mot_rate[country], label=('Country Name'))
# set labels
ax.set_xlabel('Year')
ax.set_ylabel('Mortality rate, under-5 (per 1,000 live births)')
ax.set_title('Under-5 Mortality Rate by Country and Year')
plt.legend()
plt.show()  # show the graph

# Bar plot Mortality rate, under-5 (per 1,000 live births)
fig, ax = plt.subplots(figsize=(10, 6))
mot_rate.plot(kind='bar', ax=ax)
ax.set_xlabel('Year')
ax.set_ylabel('Mortality rate, under-5 (per 1,000 live births)')
ax.set_title('Under-5 Mortality Rate by Country and Year')
plt.ylim(0, 1.4 * max(mot_rate.max()))  # set y-axis limit
# show the bar graph
plt.show()

# line plot of population growth
fig, ax = plt.subplots(figsize=(10, 6))
for country in pop_growth.columns:
    ax.plot(pop_growth.index, pop_growth[country], label=country)
# set appropraite labels
ax.set_title('Population growth (annual %)')
ax.set_xlabel('Year')
ax.set_ylabel('Annual pop growth')

# Add a legend and gridlines
ax.legend(loc='upper left', frameon=False, fontsize='small')
ax.grid(True, linestyle='--', alpha=0.5)

# Show the plot
plt.show()


# bar plot of Population growth (annual %)
fig, ax = plt.subplots(figsize=(10, 6))
pop_growth.plot(kind='bar', ax=ax, label=country)
ax.set_xlabel('Year')
ax.set_ylabel('Annual pop growth')
ax.set_title('Population growth (annual %')
plt.ylim(0, 1.4 * max(pop_growth.max()))  # set y-axis limit
plt.show()
# plot a scatter plot of population total for the countries
us_pop = pop_total['United States']
china_pop = pop_total['China']
india_pop = pop_total['India']
nigeria_pop = pop_total['Nigeria']

# Create a scatter plot of china, india and nigeria v United states
plt.scatter(us_pop, china_pop, label='China')
plt.scatter(us_pop, india_pop, label='India')
plt.scatter(us_pop, nigeria_pop, label='Nigeria')
plt.xlabel('Population of United States')
plt.ylabel('Population')
plt.title('Population of China, India, and Nigeria vs United States')
plt.legend()
plt.show()

# calculate the mean mortality rate and annual population growth
mean_mot_rate = np.mean(mot_rate)
mean_pop_growth = np.mean(pop_growth)
print(mean_mot_rate)
print(mean_pop_growth)

# calculate correlation matrix
corr_matrix = pop_growth.corr()

# create heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

# show plot
plt.show()
