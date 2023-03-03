
"""
Created on Tue Feb 28 21:29:29 2023

@author: DELL
"""

import pandas as pd
import matplotlib.pyplot as plt

# Data source: UK House Price Index summary: December 2022 - GOV.UK (www.gov.uk)
# load Annual-price-change-by-country into pandas dataframe

uk_countries = pd.read_csv('Annual-price-change-by-country-2022-12.txt')
print(uk_countries)


def plot_line_graph():
    """
    Plots a line graph using the given x and y data.

    Parameters:
    x_data ("Date")
    y_data ("United_Kingdom", "England", "Scotland", "Wales", "Northern_Ireland")
    x_label ("date")
    y_label (price change)
    title ("Annual price change by country")

    Returns:
    None
    """


# set the width and height in inches to 10 and 4 respectively
plt.figure(figsize=(10, 4))
# plot the changes in prices of the countries
plt.plot(uk_countries["Date"],
         uk_countries["United_Kingdom"], label="United Kingdom")
plt.plot(uk_countries["Date"], uk_countries["England"], label="England")
plt.plot(uk_countries["Date"], uk_countries["Scotland"], label="Scotland")
plt.plot(uk_countries["Date"], uk_countries["Wales"], label="Wales")
plt.plot(uk_countries["Date"], uk_countries["Northern_Ireland"],
         label="Northern Ireland")
# add lables and legends
plt.legend()
plt.xticks(rotation=90)  # set x-ticks to 90degrees for visibility
plt.xlabel('Date')
plt.ylabel('Price Change')
plt.title('Annual price change by country')
plt.show()


def plot_sales_volumes():
    """
    Reads sales volume data from a CSV file and plots a bar chart.

    Parameters: sales_volumes for UK and its constituent countries
    -----------
    file_path : str
        The path to the CSV file containing the sales volume data. The file should
        have a 'Date' column and a column containing the sales volumes for each date.

    Returns:
    --------
    None
        The function only plots the chart and does not return anything.
    """


# Read in the sales volumes data from the CSV file
sales_volumes = pd.read_csv("Sales-volumes-by-country-2022-12.csv.txt")
print(sales_volumes.head())
# Set the index of the data to the 'Date' column
sales_volumes = sales_volumes.set_index('Date')
print(sales_volumes)
# plot bar chat and add labels
ax = sales_volumes.plot(kind='bar', figsize=(10, 4))
ax.set_xlabel('Date')
plt.xticks(rotation=0)
ax.set_ylabel('Sales volumes')
ax.set_title('Sales volumes for the UK over the past 5 years')
plt.show()  # dispplay the Bar plot of Uk sales volume


# Pie plot of UK's average price change in property for 2022
# load data into Panda dataframe
properties = pd.read_csv(
    'Uk-average-price-change-by-property-type-2022-12.txt')
print(properties)

labels = ['Detached', 'Semi-detached', 'Terraced', 'Flat or maisonette']
sizes = [463108, 286413, 241147, 233400]


def divide_by_12(sizes):
    """
    Divides each element in a list by 12 and returns the new list.

    Parameters:
    sizes (list): A list of numbers to be divided by 12.
    Returns:
    list: A new list containing the result of dividing each element of `sizes` by 12.
    """
    sizes = [463108, 286413, 241147, 233400]
    return [size / 12 for size in sizes]
    divided_list = []
    for number in sizes:
        divided_list.append(number / 12)
    return divided_list


divided_list = divide_by_12(sizes)
print(divided_list)

colors = ['red', 'orange', 'yellow', 'green']

# Create the pie chart
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
# Add a title to the chart
ax.set_title('Average Monthly Price by Property Type')

# Display the chart
plt.show()
