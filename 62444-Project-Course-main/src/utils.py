import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import numpy as np
from keplergl import KeplerGl

def create_scatterplot(df, x_col, y_col, title, xlabel, ylabel):
    """
    This function creates a scatter plot with a linear regression line from a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    x_col (str): The column in the DataFrame to use for the x-axis.
    y_col (str): The column in the DataFrame to use for the y-axis.
    title (str): The title of the plot.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    """

    # Create the plot
    plt.figure(figsize=(7, 7))
    sns.regplot(x=df[x_col], y=df[y_col], scatter_kws={"alpha": 0.3})

    # Add labels and title
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Show the plot
    plt.show()


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keplergl import KeplerGl

def get_a_random_chunk_property(data):
    """
    This function only serves as an example of fetching some of the properties
    from the data.
    Indeed, all the content in "data" may be useful for your project!
    """
    # Randomly select a chunk
    chunk_index = np.random.choice(len(data))

    # Get list of dates
    date_list = list(data[chunk_index]["near_earth_objects"].keys())

    # Randomly select a date
    date = np.random.choice(date_list)

    # Get data for the selected date
    objects_data = data[chunk_index]["near_earth_objects"][date]

    # Randomly select an object
    object_index = np.random.choice(len(objects_data))

    # Get the selected object
    object = objects_data[object_index]

    # Get list of properties
    properties = list(object.keys())

    # Randomly select a property
    property = np.random.choice(properties)

    # Print the selected date, object name, and property value
    print("date:", date)
    print("NEO name:", object["name"])
    print(f"{property}:", object[property])

def clean_zones(df_zones):
    """
    Remove duplicate entries in the zones data.
    """
    return df_zones.drop_duplicates(subset='LocationID', keep='first')

def filter_data(df, start_time, end_time, selected_columns):
    """
    Filter data based on the time range and select relevant columns.
    """
    return df[(df['tpep_pickup_datetime'] >= start_time) & (df['tpep_pickup_datetime'] <= end_time)][selected_columns]

def add_location_data(df, zones_cleaned):
    """
    Add pickup and dropoff latitude/longitude to the dataframe.
    """
    # Set LocationID as index and select lat/lng columns
    locations = zones_cleaned.set_index('LocationID')[['lat', 'lng']]
    
    # Join pickup locations
    df = df.join(locations, on='PULocationID', rsuffix='_pickup')
    
    # Join dropoff locations
    df = df.join(locations, on='DOLocationID', rsuffix='_dropoff')
    
    return df

def calculate_iqr(df, column):
    """
    Calculate the Interquartile Range (IQR) for a given column.
    """
    # First quartile (25th percentile)
    Q1 = df[column].quantile(0.25)
    
    # Third quartile (75th percentile)
    Q3 = df[column].quantile(0.75)
    
    # Interquartile range
    IQR = Q3 - Q1
    
    return Q1, Q3, IQR

def filter_outliers(df, column, lower_bound, upper_bound):
    """
    Filter out outliers based on the IQR.
    """
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def clean_data(df):
    """
    Clean data by removing outliers and negative fares.
    """
    # Calculate IQR for trip distance
    Q1_trip, Q3_trip, IQR_trip = calculate_iqr(df, 'trip_distance')
    
    # Calculate IQR for fare amount
    Q1_fare, Q3_fare, IQR_fare = calculate_iqr(df, 'fare_amount')
    
    # Define bounds for trip distance
    lower_trip, upper_trip = Q1_trip - 1.5 * IQR_trip, Q3_trip + 1.5 * IQR_trip
    
    # Define bounds for fare amount
    lower_fare, upper_fare = Q1_fare - 1.5 * IQR_fare, Q3_fare + 1.5 * IQR_fare
    
    # Filter outliers for trip distance
    df = filter_outliers(df, 'trip_distance', lower_trip, upper_trip)
    
    # Filter outliers for fare amount
    df = filter_outliers(df, 'fare_amount', lower_fare, upper_fare)
    
    # Remove negative fares
    df = df[df['fare_amount'] >= 0]
    
    return df

def plot_boxplot(df, column, title, xlabel, ylabel, color):
    """
    Plot a boxplot for a given column.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[column], color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_histogram(df, column, bins, title, xlabel, ylabel, color, kde=False):
    """
    Plot a histogram for a given column.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], bins=bins, kde=kde, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_scatter_with_regression(df, x_column, y_column, title, xlabel, ylabel, scatter_color, line_color):
    """
    Plot a scatter plot with a regression line.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x_column, y=y_column, data=df, alpha=0.5, color=scatter_color)
    sns.regplot(x=x_column, y=y_column, data=df, scatter=False, color=line_color, line_kws={'label': 'Linear Regression'})
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def plot_heatmap(df, columns, title):
    """
    Plot a heatmap for the correlation matrix of given columns.
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(df[columns].corr(), annot=True, cmap='Blues')
    plt.title(title)
    plt.show()

def load_and_concat_data(base_url_yellow, base_url_green, months, year):
    """
    Load and concatenate monthly data for yellow and green taxis.
    """
    df_yellow_list = []
    df_green_list = []
    
    # Loop over the months
    for month in months:
        # Load yellow taxi data
        dfy = pd.read_parquet(base_url_yellow.format(year=year, month=month))
        
        # Load green taxi data
        dfg = pd.read_parquet(base_url_green.format(year=year, month=month))
        
        # Append yellow taxi data to list
        df_yellow_list.append(dfy)
        
        # Append green taxi data to list
        df_green_list.append(dfg)
    
    # Concatenate yellow taxi data
    df_yellow = pd.concat(df_yellow_list)
    
    # Concatenate green taxi data
    df_green = pd.concat(df_green_list)
    
    return df_yellow, df_green

def clean_zones(df_zones):
    """
    Remove duplicate entries in the zones data.
    """
    df_zones.drop_duplicates(subset='LocationID', keep='first', inplace=True)
    return df_zones

def filter_data(df, start_time, end_time, selected_columns):
    """
    Filter data based on the time range and select relevant columns.
    """
    return df[(df['tpep_pickup_datetime'] >= start_time) & (df['tpep_pickup_datetime'] <= end_time)][selected_columns]

def convert_datetime_to_str(df, columns):
    """
    Convert datetime columns to string format.
    """
    for column in columns:
        df[column] = df[column].astype(str)
    return df

def get_location_coordinates(df, df_zones, location_id_col, lat_col, lng_col):
    """
    Get latitude and longitude for pickup and dropoff locations.
    """
    # Join the dataframe with zones data to get lat/lng
    locations = df.join(df_zones.set_index('LocationID'), on=location_id_col, how='left')
    
    # Rename columns for clarity
    locations = locations.rename({'lat': lat_col, 'lng': lng_col}, axis=1)[[lat_col, lng_col]]
    
    return locations

def add_coordinates_to_df(df, pickup_locations, dropoff_locations):
    """
    Add pickup and dropoff coordinates to the dataframe.
    """
    # Add pickup coordinates
    for col in pickup_locations.columns:
        df[col] = pickup_locations[col]
    
    # Add dropoff coordinates
    for col in dropoff_locations.columns:
        df[col] = dropoff_locations[col]
    
    return df

def create_kepler_map(df_yellow_selected, df_routes):
    """
    Create a Kepler.gl map with the given data.
    """
    map_1 = KeplerGl(height=600, data={'yellow_taxi': df_yellow_selected, 'routes': df_routes})
    return map_1

def load_data_from_google_drive(url):
    """
    Load data from a Google Drive URL into a pandas DataFrame.
    """
    # Process the Google Drive URL to get the file ID
    url_processed = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
    
    # Load the data into a DataFrame
    df = pd.read_csv(url_processed)
    
    return df

def load_and_preprocess_data(base_url, months, year, datetime_col):
    """
    Load and preprocess data for a given year and months.
    """
    df_list = []
    
    # Loop over the months
    for month in months:
        # Load data for the month
        df = pd.read_parquet(base_url.format(year=year, month=month))
        
        # Convert datetime column to datetime format
        df['datetime'] = pd.to_datetime(df[datetime_col])
        
        # Append data to list
        df_list.append(df)
    
    # Concatenate data
    df = pd.concat(df_list)
    
    # Extract date from datetime
    df['date'] = df['datetime'].dt.date
    
    return df

def plot_temporal_analysis(df, title_prefix):
    # Number of rides by hour
    rides_by_hour = df.groupby('hour').size()
    plt.figure(figsize=(12, 6))
    sns.barplot(x=rides_by_hour.index, y=rides_by_hour.values)
    plt.title(f'{title_prefix} - Number of Rides by Hour')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Rides')
    plt.show()

    # Number of rides by day of the week
    rides_by_day_of_week = df.groupby('day_of_week').size()
    plt.figure(figsize=(12, 6))
    sns.barplot(x=rides_by_day_of_week.index, y=rides_by_day_of_week.values)
    plt.title(f'{title_prefix} - Number of Rides by Day of the Week')
    plt.xlabel('Weekday (0 = Monday, 6 = Sunday)')
    plt.ylabel('Number of Rides')
    plt.show()

    # Number of rides by month
    rides_by_month = df.groupby('month').size()
    plt.figure(figsize=(12, 6))
    sns.barplot(x=rides_by_month.index, y=rides_by_month.values)
    plt.title(f'{title_prefix} - Number of Rides by Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Rides')
    plt.show()

    # Average trip distance by hour
    avg_distance_by_hour = df.groupby('hour')['trip_distance'].mean()
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=avg_distance_by_hour.index, y=avg_distance_by_hour.values)
    plt.title(f'{title_prefix} - Average Trip Distance by Hour')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Average Trip Distance')
    plt.show()

    # Average fare by hour
    avg_fare_by_hour = df.groupby('hour')['fare_amount'].mean()
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=avg_fare_by_hour.index, y=avg_fare_by_hour.values)
    plt.title(f'{title_prefix} - Average Fare by Hour')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Average Fare')
    plt.show()

def forecast_with_prophet(df, title_prefix):
    # Prepare data for Prophet
    df_daily = df.groupby(df['datetime'].dt.date).size().reset_index(name='rides')
    df_daily.columns = ['ds', 'y']

    # Initialize and train Prophet model
    model = Prophet(seasonality_mode='additive', yearly_seasonality=True, weekly_seasonality=True)
    model.fit(df_daily)

    # Create future dataframe for forecasting (365 days into the future)
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    # Filter the forecast to show only from the next year onward
    forecast_next_year = forecast[forecast['ds'] >= pd.Timestamp('2023-01-01')]

    # Calculate RMSE for the forecasts on the current year data
    y_true = df_daily['y'].values
    y_pred = forecast.loc[:len(df_daily) - 1, 'yhat'].values
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Calculate and display correlation
    correlation = df_daily['y'].corr(pd.Series(forecast.loc[:len(df_daily) - 1, 'yhat'].values))

    # Plot actual data and forecast
    plt.figure(figsize=(14, 7))
    actual_data = df_daily[df_daily['ds'] <= pd.Timestamp('2022-12-31').date()]
    plt.plot(actual_data['ds'], actual_data['y'], label=f"{title_prefix} Actual (2022)", color='blue', alpha=0.8)
    plt.plot(forecast_next_year['ds'], forecast_next_year['yhat'], label=f"{title_prefix} Forecast (2023)", linestyle='--', color='red', alpha=0.8)
    plt.fill_between(forecast_next_year['ds'], forecast_next_year['yhat_lower'], forecast_next_year['yhat_upper'], color='gray', alpha=0.2, label="Confidence Interval (Forecast)")
    plt.xlim([pd.Timestamp('2022-01-01'), pd.Timestamp('2023-12-31')])
    plt.title(f"{title_prefix} Rides: Actual 2022 vs Forecast 2023", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Number of Rides", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.show()

    # Output RMSE and correlation
    print(f"{title_prefix} RMSE: {rmse:.2f}")
    print(f"{title_prefix} Correlation: {correlation:.2f}")