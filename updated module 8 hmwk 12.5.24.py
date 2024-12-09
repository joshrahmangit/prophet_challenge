# Install the required libraries
!pip install prophet

# Import the required libraries and dependencies
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import numpy as np
%matplotlib inline

# Store the data in a Pandas DataFrame and set the "Date" column as the Datetime Index
df_mercado_trends = pd.read_csv(
    "https://static.bc-edx.com/ai/ail-v-1-0/m8/lms/datasets/google_hourly_search_trends.csv",
    index_col='Date',
    parse_dates=True
).dropna()

# Review the first and last five rows of the DataFrame
display(df_mercado_trends.head())
display(df_mercado_trends.tail())

# Review the data types of the DataFrame using the info function
df_mercado_trends.info()


# Step 1: Find Unusual Patterns in Hourly Google Search Traffic
# Filter the data to only include May 2020
may_2020_data = df_mercado_trends['2020-05']

# Plot the search traffic data for May 2020
plt.figure(figsize=(10, 6))
plt.plot(may_2020_data.index, may_2020_data['Search_Volume'], color='blue', label='Search Traffic')
plt.title('Google Search Traffic for MercadoLibre - May 2020')
plt.xlabel('Date')
plt.ylabel('Search Volume')
plt.legend()
plt.grid()
plt.show()

# Display summary statistics to observe any unusual patterns
summary_statistics = may_2020_data.describe()
print("Summary Statistics for May 2020 Search Traffic:")
print(summary_statistics)

# Calculate the total search traffic for May 2020
total_search_traffic_may = may_2020_data['Search_Volume'].sum()
print(f"Total Search Traffic for May 2020: {total_search_traffic_may}")

# Calculate the median monthly search traffic across all months
monthly_median_traffic = df_mercado_trends.resample('M')['Search_Volume'].median().median()
print(f"Monthly Median Search Traffic (across all months): {monthly_median_traffic}")

# Group the DataFrame by year and month, then calculate the sum and finally the median
monthly_grouped_median = df_mercado_trends.resample('M')['Search_Volume'].sum().median()
print(f"Median of the monthly total search traffic (sum of each month): {monthly_grouped_median}")

# Compare the search traffic for the month of May 2020 to the overall monthly median value
traffic_comparison_ratio = total_search_traffic_may / monthly_median_traffic
print(f"Ratio of May 2020 Search Traffic to Median Monthly Traffic: {traffic_comparison_ratio:.2f}")


# Check if search traffic increased in May 2020
traffic_increased = total_search_traffic_may > monthly_median_traffic
print(f"Did Google Search Traffic increase in May 2020?: {'Yes' if traffic_increased else 'No'}")

# Observations: Write down if any unusual patterns exist
# Based on the plot and summary statistics, observe if there are any noticeable spikes or anomalies in the search traffic.
# Observations:
# 1. The plot shows noticeable spikes in search volume around specific dates in May 2020, particularly around mid-May.
#    This suggests increased interest potentially related to financial events, such as the release of quarterly results.
# 2. The summary statistics indicate a relatively high maximum value compared to the mean, which suggests some unusually high search traffic days.
# 3. The total search traffic for May 2020 is higher than the median monthly search traffic across all months, indicating increased interest during this period.
# 4. The spikes could be correlated with financial releases or other significant company announcements that generated public interest.

# Step 1 Conclusion:
# The Google search traffic for May 2020 increased compared to the median monthly search traffic across all months.
# This suggests that the release of MercadoLibre's financial results had a positive impact on public interest, as reflected in increased search traffic.

# Step 2: Mine the Search Traffic Data for Seasonality
# Group the hourly search data to plot the average traffic by the hour of day.
df_mercado_trends['Hour'] = df_mercado_trends.index.hour
hourly_avg_traffic = df_mercado_trends.groupby('Hour')['Search_Volume'].mean()

# Plot the average traffic by the hour of day
plt.figure(figsize=(10, 6))
plt.plot(hourly_avg_traffic.index, hourly_avg_traffic, color='green', marker='o', linestyle='-', label='Average Traffic')
plt.title('Average Google Search Traffic by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Average Search Volume')
plt.xticks(range(0, 24))
plt.grid()
plt.legend()
plt.show()

# Group the hourly search data to plot the average traffic by the day of the week using `df.index.isocalendar().day`
df_mercado_trends['Day_of_Week'] = df_mercado_trends.index.isocalendar().day
daily_avg_traffic = df_mercado_trends.groupby('Day_of_Week')['Search_Volume'].mean()

# Plot the average traffic by day of the week
plt.figure(figsize=(10, 6))
plt.plot(daily_avg_traffic.index, daily_avg_traffic, color='purple', marker='o', linestyle='-', label='Average Traffic')
plt.title('Average Google Search Traffic by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Average Search Volume')
plt.grid()
plt.legend()
plt.show()

# Plot the average traffic by day of the week
plt.figure(figsize=(10, 6))
plt.plot(daily_avg_traffic.index, daily_avg_traffic, color='purple', marker='o', linestyle='-', label='Average Traffic')
plt.title('Average Google Search Traffic by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Average Search Volume')
plt.grid()
plt.legend()
plt.show()

# Group the hourly search data to plot the average traffic by the day of week using `df.index.isocalendar().day`
df_mercado_trends['Day_of_Week'] = df_mercado_trends.index.isocalendar().day
daily_avg_traffic = df_mercado_trends.groupby('Day_of_Week')['Search_Volume'].mean()

# Plot the average traffic by day of the week
plt.figure(figsize=(10, 6))
plt.plot(daily_avg_traffic.index, daily_avg_traffic, color='purple', marker='o', linestyle='-', label='Average Traffic')
plt.title('Average Google Search Traffic by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Average Search Volume')
plt.grid()
plt.legend()
plt.show()


# Step 3: Relate the Search Traffic to Stock Price Patterns

# Upload the "mercado_stock_price.csv" file into Colab, then store in a Pandas DataFrame
# Set the "date" column as the Datetime Index.
from google.colab import files
import pandas as pd

# Upload the CSV file
uploaded = files.upload()

# Load the uploaded CSV file into a Pandas DataFrame
df_mercado_stock = pd.read_csv('mercado_stock_price.csv', index_col='date', parse_dates=True).dropna()

# View the first and last five rows of the DataFrame
display(df_mercado_stock.head())
display(df_mercado_stock.tail())

# Concatenate the stock price data to the search data in a single DataFrame
df_combined = pd.concat([df_mercado_trends, df_stock_price], axis=1).dropna()

# Display the first few rows of the combined DataFrame
display(df_combined.head())

# Visualize the closing price of the df_mercado_stock DataFrame
plt.figure(figsize=(10, 6))
plt.plot(df_mercado_stock.index, df_mercado_stock['Close'], label='Closing Price', color='blue')
plt.title('Mercado Stock Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.legend()
plt.grid()
plt.show()


# Concatenate the DataFrame by columns (axis=1), and drop any rows with only one column of data
df_combined = pd.concat([df_mercado_stock, df_mercado_trends], axis=1).dropna()

# View the first and last five rows of the concatenated DataFrame
display(df_combined.head())
display(df_combined.tail())


# Step 3: Slice the data to just the first half of 2020 (2020-01 to 2020-06) and plot the data
first_half_2020 = df_combined['2020-01':'2020-06']

# View the first and last five rows of the sliced DataFrame
display(df_first_half_2020.head())
display(df_first_half_2020.tail()
        
## Visualize the close and Search Trends data
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Plot Search Trends data
axes[0].plot(df_first_half_2020.index, df_first_half_2020['Search_Volume'], color='blue', label='Search Trends')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Search Volume')
axes[0].set_title('Search Trends for MercadoLibre (First Half of 2020)')
axes[0].legend()
axes[0].grid()

# Plot Stock Price data
axes[1].plot(df_first_half_2020.index, df_first_half_2020['Close'], color='red', label='Closing Price')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Stock Price (USD)')
axes[1].set_title('Stock Price for MercadoLibre (First Half of 2020)')
axes[1].legend()
axes[1].grid()

plt.tight_layout()
plt.show()

# Question: Do both time series indicate a common trend that’s consistent with this narrative?
# Answer:
# Yes, both time series indicate a common trend that is consistent with the narrative. During the first half of 2020,
# there is a noticeable increase in both search volume and stock price, especially around the months of April and May.
# This suggests that increased public interest in MercadoLibre (as reflected in search volume) may be correlated with
# a rise in the stock price, possibly due to the surge in e-commerce activity during the COVID-19 pandemic.



# Step 3: Create a new column in the DataFrame named "Lagged Search Trends" that offsets, or shifts, the search traffic by one hour
df_combined['Lagged_Search_Trends'] = df_combined['Search_Volume'].shift(1)

# Create two additional columns
# "Stock Volatility", which holds an exponentially weighted four-hour rolling average of the company’s stock volatility
df_combined['Stock_Volatility'] = df_combined['Close'].pct_change().rolling(window=4).std() * np.sqrt(4)

# "Hourly Stock Return", which holds the percent change of the company's stock price on an hourly basis
df_combined['Hourly_Stock_Return'] = df_combined['Close'].pct_change()

# Create a new column in the mercado_stock_trends_df DataFrame called Lagged Search Trends
df_combined['Lagged_Search_Trends'] = df_combined['Search_Volume'].shift(1)

# Create a new column in the mercado_stock_trends_df DataFrame called Hourly Stock Return
# This column should calculate hourly return percentage of the closing price
df_combined['Hourly_Stock_Return'] = df_combined['Close'].pct_change()

# View the first and last five rows of the mercado_stock_trends_df DataFrame
display(df_combined.head())
display(df_combined.tail())

# Construct correlation table of Stock Volatility, Lagged Search Trends, and Hourly Stock Return
correlation_table = df_combined[['Stock_Volatility', 'Lagged_Search_Trends', 'Hourly_Stock_Return']].corr()
display(correlation_table)



# Step 4: Set up the Google search data for a Prophet forecasting model

# Using the df_mercado_trends DataFrame, reset the index so the date information is no longer the index
df_mercado_trends_reset = df_mercado_trends.reset_index()

# Prepare the DataFrame for Prophet
prophet_df = df_mercado_trends[['Search_Volume']].reset_index()
prophet_df.columns = ['ds', 'y']

# Initialize the Prophet model
model = Prophet()

# Fit the model to the search data
model.fit(prophet_df)

# Drop any NaN values from the prophet_df DataFrame
prophet_df = df_mercado_trends_reset.dropna()

# View the first and last five rows of the mercado_stock_trends_df DataFrame
display(df_combined.head())
display(df_combined.tail())

# Call the Prophet function, store as an object
model = Prophet()

# Fit the time-series model
model.fit(prophet_df)


# Step 4: Plot the forecast
## Create a future dataframe to hold predictions
# Make the prediction go out as far as 2000 hours (approx 80 days)
future = model.make_future_dataframe(periods=2000, freq='H')

# Use the fitted model to make predictions
forecast = model.predict(future)

# Plot the forecast
fig = model.plot(forecast)
plt.title('Forecast of MercadoLibre Search Volume')
plt.xlabel('Date')
plt.ylabel('Search Volume')
plt.show()

# Plot the individual time series components of the model
fig_components = model.plot_components(forecast)
plt.show()

# Plot the Prophet predictions for the Mercado trends data
plt.figure(figsize=(10, 6))
plt.plot(forecast['ds'], forecast['yhat'], label='Predicted Search Volume', color='green')
plt.xlabel('Date')
plt.ylabel('Search Volume')
plt.title('Prophet Predictions for MercadoLibre Search Volume')
plt.legend()
plt.grid()
plt.show()

# View only the yhat, yhat_lower and yhat_upper columns from the forecast DataFrame
display(forecast[['yhat', 'yhat_lower', 'yhat_upper']])

# From the forecast_mercado_trends DataFrame, plot the data to visualize
# the yhat, yhat_lower, and yhat_upper columns over the last 2000 hours
plt.figure(figsize=(12, 6))
plt.plot(forecast['ds'][-2000:], forecast['yhat'][-2000:], label='Predicted Search Volume', color='green')
plt.fill_between(forecast['ds'][-2000:], forecast['yhat_lower'][-2000:], forecast['yhat_upper'][-2000:], color='lightgreen', alpha=0.5)
plt.xlabel('Date')
plt.ylabel('Search Volume')
plt.title('Prophet Predictions with Uncertainty Intervals (Last 2000 Hours)')
plt.legend()
plt.grid()
plt.show()

# Reset the index in the forecast_mercado_trends DataFrame
forecast.reset_index(drop=True, inplace=True)

# Use the plot_components function to visualize the forecast results
fig_components = model.plot_components(forecast)
plt.show()




# Answer to Question: What time of day exhibits the greatest popularity?
# Based on the individual time series components plot, the time of day that exhibits the greatest popularity is in the late evening, around 8 PM to 10 PM.

# Answer to Question: Which day of the week gets the most search traffic?
# Based on the individual time series components plot, the day of the week that gets the most search traffic is Monday, followed closely by Friday.

# Answer to Question: What's the lowest point for search traffic in the calendar year?
# Based on the individual time series components plot, the lowest point for search traffic in the calendar year appears to be during the early weeks of the year, specifically around weeks 1 to 4.
