import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import calendar
from datetime import datetime
import seaborn as sns
import matplotlib.dates as mdates

months_array = np.empty(12) 
months = [calendar.month_name[month] for month in range(1, 13)]
file_path = r'C:\Users\ASUS\Desktop\forecasting_data_phase2.xlsx'
file_path2 = r'C:\Users\ASUS\Desktop\total_ratio.xlsx'
file_path3 = r'C:\Users\ASUS\Desktop\finansal_hesaplama.xlsx'
data = pd.read_excel(file_path)
data_ratio = pd.read_excel(file_path2)
sheets_dict = pd.read_excel(file_path3, sheet_name = None)
df_financial_sheet1 = sheets_dict['Sheet1']
df_financial_sheet2 = sheets_dict['Sheet2']
data_ratio.drop('Unnamed: 0', axis = 1, inplace = True)
common_stores = set(data['Account']).intersection(set(data_ratio['Store Name']))
# Filter both DataFrames to only include these common store names
data = data[data['Account'].isin(common_stores)]
data_ratio = data_ratio[data_ratio['Store Name'].isin(common_stores)]
###############################################################################   
def prepare_dataframe(data):
    #Define the items to drop
    items_to_drop = ["ITEM2", "ITEM3", "ITEM4", "ITEM5", "ITEM6"]
    # Drop rows containing the specified items in the 'Ürün' column
    filtered_data = data[~data['Ürün'].isin(items_to_drop)]
    ###########Sum all the daily sales for each shop, grouping by 'Store Code' and 'Account'#########
    sales_columns = filtered_data.columns[3:] 
    summed_sales = filtered_data.groupby(['Store Code', 'Account'])[sales_columns].sum()
    date_columns = summed_sales.columns[0:]
    date_columns_datetime = pd.to_datetime(date_columns, errors='coerce')
    date_column_mapping = {original: formatted.strftime('%Y-%m-%d %A') for original, formatted in zip(date_columns, date_columns_datetime)}
    summed_sales.rename(columns=date_column_mapping, inplace=True)
    return filtered_data, summed_sales

def sum_of_daily_sale(summed_sales):
    col_sums = summed_sales.sum(axis=0)
    total_sales_df = pd.DataFrame(col_sums).transpose()
    total_sales_df.index = ['Total Sales']
    return total_sales_df

def sum_of_everyday(filtered_data, summed_sales, total_sales_df):
    weekdays = list(calendar.day_name)
    sum_of_days = pd.DataFrame(index=total_sales_df.index)
    for i in range(len(weekdays)):
        filtered_columns = [col for col in total_sales_df.columns if pd.to_datetime(col).day_name() == weekdays[i]]
        day_df = total_sales_df[filtered_columns]
        row_sums = day_df.sum(axis=1)
        sum_of_days[weekdays[i]] = row_sums 
    sum_of_days = sum_of_days
    return sum_of_days

def ratio_of_everyday(sum_of_days):
    total = sum_of_days.sum(axis= 1)   
    sum_of_days_ratio = pd.DataFrame(columns = sum_of_days.columns, index = ["Total Sales Percentage"])
    for i in range(len(sum_of_days.columns)):
        sum_of_days_ratio.iloc[:, i] = (sum_of_days.iloc[:,i]/total)
    return sum_of_days_ratio

def sum_of_everyday_per_store(summed_sales):
    weekdays = list(calendar.day_name)
    store_sum_of_days = pd.DataFrame(index=summed_sales.index)
    for i in range(len(weekdays)):
        filtered_columns = [col for col in summed_sales.columns if pd.to_datetime(col).day_name() == weekdays[i]]
        day_df = summed_sales[filtered_columns]
        store_sum_of_days[weekdays[i]] = day_df.sum(axis=1)
    return store_sum_of_days

def ratios_per_store(store_sum_of_days):
    total = store_sum_of_days.sum(axis=1)
    total = total.replace(0, np.nan)  # Replace 0 with NaN to prevent division issues
    store_sum_of_days_ratio = pd.DataFrame(columns=store_sum_of_days.columns, index=store_sum_of_days.index)
    for i in range(len(store_sum_of_days)):
        for j in range(len(store_sum_of_days.columns)):
            store_sum_of_days_ratio.iloc[i, j] = store_sum_of_days.iloc[i, j] / total.iloc[i]
    store_sum_of_days_ratio = store_sum_of_days_ratio.fillna(0)
    return store_sum_of_days_ratio

def monthly_sales_per_Store(filtered_data, store_and_code_pair, i, j):
    
    my_Data = filtered_data.copy()
    my_Data.rename(columns = {'Account' : 'Store', 'Ürün' : 'Product'}, inplace = True) 
    flowers = my_Data['Product'].unique().tolist()
    my_Data = my_Data.groupby(['Store Code', 'Store', 'Product']).sum()
    filtered_store = my_Data.loc[[(store_and_code_pair[i][0], store_and_code_pair[i][1], flowers[j])]]
    filtered_store.columns = pd.to_datetime(filtered_store.columns)
    monthly_group = filtered_store.columns.to_period('M')
    grouped = filtered_store.T.groupby(monthly_group)
    monthly_sums = grouped.sum().T
    monthly_sums
    #pd.set_option('display.max_columns', None)
    #pd.reset_option('display.max_columns')
    return monthly_sums

def choose_Store(prep_organized_data, filtered_data):
    unique_store_codes = filtered_data['Store Code'].unique()
    unique_flowers = filtered_data['Ürün'].unique()
    for i in range(len(unique_store_codes)):
        for j in range(len(unique_flowers)):
            prep_row = prep_organized_data.loc[(unique_store_codes[i], slice(None), unique_flowers[j])]
            
    return prep_row

def create_prep_row_and_prep_ratio_row(filtered_data, data_ratio, months, sum_of_days_ratio, sum_of_days, months_array ):
    prep_data = filtered_data.copy()
    prep_data = prep_data.rename(columns = {'Account' : 'Store Name', 'Ürün' : 'Product'})
    flowers = prep_data['Product'].unique().tolist()
    store_and_code_pair = list(set(zip(prep_data['Store Code'], prep_data['Store Name'])))
    prep_sales_columns = prep_data.columns[3:] 
    prep_organized_data = prep_data.groupby(['Store Code', 'Store Name', 'Product'])[prep_sales_columns].sum()
    prep_organized_data.columns = pd.to_datetime(prep_organized_data.columns, format='%Y-%m-%d')
    weekdays = [date.strftime('%A') for date in prep_organized_data.columns]
    ##################Total Monthly Ratio####################
    prep_data_ratio = data_ratio.copy()
    prep_data_ratio = prep_data_ratio.rename(columns = {'Store ID' : 'Store Code'})
    prep_data_ratio_columns = prep_data_ratio.columns[2:]
    prep_data_ratio = prep_data_ratio.groupby(['Store Code', 'Store Name'])[prep_data_ratio_columns].sum()
    period_range = pd.period_range(start="2023-01", end="2023-12", freq='M')
    prep_data_ratio.columns = period_range

 
    return  flowers, store_and_code_pair, prep_organized_data, prep_data_ratio

def week_based_interval_estimation_for_flowers(start_week, end_week, monthly_data, item):
    monthly_interval_data = monthly_data.copy()

# Iterate over the months in the dictionary
    for month, df in  monthly_interval_data.items():
        # Convert the column names (which are datetime) to actual datetime objects
        df.columns = pd.to_datetime(df.columns)
        
        # Extract the week numbers from the datetime headers
        week_numbers = df.columns.isocalendar().week
        
        # Filter the DataFrame to only include columns (dates) within the desired week range
        filtered_df = df.loc[:, (week_numbers >= start_week) & (week_numbers <= end_week)]
        
        # Create a full date range for the entire period of interest
        full_date_range = pd.date_range(start=df.columns.min(), end=df.columns.max(), freq='D')
        
        # Create a DataFrame for Predicted Sales, initialized to 0
        zero_filled_sales_df = pd.DataFrame(0, index=['Predicted Sales'], columns=full_date_range)
        
        # Copy the "Day" row from the original DataFrame
        days_df = df.loc[['Day'], full_date_range]
        
        # Update the zero-filled Predicted Sales DataFrame with the filtered data
        zero_filled_sales_df.update(filtered_df.loc[['Predicted Sales']])
        
        # Combine the "Day" row and the zero-filled Predicted Sales row
        combined_df = pd.concat([days_df, zero_filled_sales_df])
        
        # Replace the original DataFrame with the combined one in the dictionary
        monthly_interval_data[month] = combined_df
    return monthly_interval_data

def real_data_production(ratio_for_real_data, store, header_list, month,  selected_elements, excluded_months_sales_df, monthly_data):
    month_indices = {
    "January": 0, "February": 1,    "March": 2,    "April": 3,    "May": 4,    "June": 5,    "July": 6,
    "August": 7,    "September": 8,    "October": 9,    "November": 10,    "December": 11}
    ratio_for_data_generation = (ratio_for_real_data.loc[f'{store[0]}', 
                                header_list[month_indices[month]]]/(ratio_for_real_data.loc[f'{store[0]}', 
                                    selected_elements[:]].sum()))
    for i in range(weekly_sales_df.shape[1]):
        excluded_months_sales_df.iloc[:, i] = weekly_sales_sum_df.iloc[:, i]*ratio_for_data_generation
    excluded_months_sales_dict = excluded_months_sales_df.loc['Total Sales'].to_dict()
    for col in monthly_data[month].columns:
        day_of_week = monthly_data[month][col]['Day']
        monthly_data[month][col]['Predicted Sales'] = excluded_months_sales_dict.get(day_of_week, None)
    return monthly_data

def main(monthly_sales, filtered_data, data_ratio, months, sum_of_days_ratio, sum_of_days, prep_row, prep_ratio_row, months_array, item):
    ################Calculate the Coefficients for a single shop for every month################
    coefficient = pd.DataFrame(index = prep_ratio_row.index, columns = prep_ratio_row.columns)
    dataframes = []
    for i in range(prep_ratio_row.shape[1]):
        coefficient = pd.DataFrame(index = prep_ratio_row.index, columns = prep_ratio_row.columns)
        for j in range(prep_ratio_row.shape[1]):
            coefficient.iloc[:, j] = prep_ratio_row.iloc[:, i] / prep_ratio_row.iloc[:, j]
        dataframes.append(coefficient)   
    dataframes
    combined_coeffcients = pd.concat(dataframes, axis=0, ignore_index = True)
    combined_coeffcients#We have gotten all the coefficients for every month for our spesific store and spesific flower type    monthly_sales
    months_array = np.round(months_array).astype(int) 
    ##############################################################################################################
    for i in range(monthly_sales.shape[1]):
        months_array[i+ 2] = int(monthly_sales.iloc[:, i].iloc[0])

    months_array[2] = months_array[2] * 2
    months_array[5] = months_array[5] * 6/5
    months_array_copy = months_array.copy()

    for i in range(0, 2):
        months_array[i] = (months_array_copy[2]*combined_coeffcients.iloc[i, 2] + months_array_copy[3]*combined_coeffcients.iloc[i, 3] +
        months_array_copy[4]*combined_coeffcients.iloc[i, 4] + months_array_copy[5]*combined_coeffcients.iloc[i, 5])/4
        #print(monthly_sales.shape[1])
    for i in range(6, 12):
        months_array[i] = (months_array_copy[2]*combined_coeffcients.iloc[i, 2] + months_array_copy[3]*combined_coeffcients.iloc[i, 3] +
        months_array_copy[4]*combined_coeffcients.iloc[i, 4] + months_array_copy[5]*combined_coeffcients.iloc[i, 5])/4
        #print(monthly_sales.shape[1])

    # Create a date range for the headers
    date_range = pd.date_range(start='2024-01-01', periods=12, freq='M')
    index_values = ['Monthly Total Sales']
    # Convert the array to a DataFrame and set the formatted date headers
    monthly_total_sales_df = pd.DataFrame([months_array], index = index_values, columns=[date.strftime('%Y-%m') for date in date_range])  
    month_names = pd.Series(date_range).dt.month_name()  
    monthly_total_sales_df.loc['Month'] = month_names.values
    monthly_total_sales_df = monthly_total_sales_df.reindex(['Month'] + index_values)
    ############################################################################################
    ################Generate July################
    monthly_data = {}
    # Define the index for the DataFrame
    index_values = ['Predicted Sales']
    # Loop through each month of the year 2024
    for month in range(1, 13):
        # Generate date range for the current month
        year = 2024
        num_days = calendar.monthrange(year, month)[1]
        date_range = pd.date_range(start=f'{year}-{month:02d}-01', end=f'{year}-{month:02d}-{num_days}', freq='D')
        # Create a DataFrame with the specified index and date range as columns
        df = pd.DataFrame(index=index_values, columns=date_range)   
        # Extract day names from the date range
        day_names = pd.Series(date_range).dt.day_name() 
        # Add day names as a new row in the DataFrame
        df.loc['Day'] = day_names.values   
        # Reindex to place 'Day' at the top
        df = df.reindex(['Day'] + index_values)
        # Store the DataFrame in the dictionary with the month name as the key
        monthly_data[date_range[0].strftime('%B')] = df
    # Display the DataFrame for a specific month, e.g., July
    weekly_Sales_num = sum_of_days_ratio.copy()
    weekly_Sales_num = pd.concat([weekly_Sales_num]*12, ignore_index=False)
    for i in range(monthly_total_sales_df.shape[1]):
        for j in range(weekly_Sales_num.shape[1]):
            weekly_Sales_num.iloc[i, j] *= months_array[i] 
    weekly_Sales_num.rename(index = {'Total Sales per Day' : 'Day Count'}, inplace = True)
    weekly_Sales_num.index = months
    ##########################Count the number of Days in a Month###############
    monthly_day_counts = pd.DataFrame(np.nan, index=weekly_Sales_num.index, columns=weekly_Sales_num.columns)
    monthly_data#this is our empty dictionary where we will fill with our predited sales
    for month in months:
        day_row = monthly_data[month].loc['Day']
        day_counts = day_row.value_counts()
        day_counts = day_row.value_counts()
        day_counts = day_counts.to_frame(name=f'Day Count of month {months[i]}')
        day_counts = day_counts.T
        desired_order = list(calendar.day_name)   
        day_counts = day_counts.reindex(columns=desired_order)
        monthly_day_counts.loc[month] =  day_counts.values.flatten()
    #################Now we find the weight for every single month#################
    average_sales = weekly_Sales_num.copy()
    weight_df = pd.DataFrame(np.nan, index = ['Weights for months'], columns = months)
    for i in range(len(average_sales)):
        weight = 0
        for j in range(sum_of_days_ratio.shape[1]):
            weight += (monthly_day_counts.values[i, j]*sum_of_days_ratio.values[:,j])
        weight_df.loc['Weights for months', months[i]] = weight
    ################Give the values###############################################
    for i in range(len(average_sales)):
        for j in range(average_sales.shape[1]):
            average_sales.iloc[i, j] = (months_array[i]*(monthly_day_counts.values[i, j]*sum_of_days_ratio.values[:,j]))/weight_df.values[:, i]
            average_sales.iloc[i, j] = average_sales.values[i, j] / monthly_day_counts.values[i, j]
    ###############################################################################
    # Create a dictionary mapping days to average sales
    # Iterate over the June DataFrame and fill in the predicted sales
    for month in months:
        average_sales_dict = average_sales.to_dict('index')[month]
        for col in monthly_data[month].columns:
            day_of_week = monthly_data[month][col]['Day']
            monthly_data[month][col]['Predicted Sales'] = average_sales_dict.get(day_of_week, None)
    ###############################################################################################################################        
    # Initialize a DataFrame with days of the week as columns and a single row for total sales
    weekly_sales_df = pd.DataFrame(index=['Total Sales'], columns=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    
    # Fill the DataFrame with zeros initially
    weekly_sales_df.loc['Total Sales'] = 0
    
    
    
    
    ratio_for_real_data = data_ratio.copy()
    ratio_for_real_data = ratio_for_real_data.drop(columns=['Store ID'])
    ratio_for_real_data.set_index('Store Name', inplace = True)
    header_list = ratio_for_real_data.columns.tolist()
    selected_elements = [header_list[i] for i in [0, 1, 6, 7, 8, 9, 10, 11]]
    # List of months to exclude
    exclude_months = ['March', 'April', 'May', 'June']
    # Initialize a dictionary to hold the sum of sales for each day of the week
    weekly_sales_sum = {day: 0 for day in calendar.day_name}

    # Loop through all months in the monthly_data dictionary
    for month, df in monthly_data.items():
        if month not in exclude_months:
                     
            # Loop through the columns (dates) of the DataFrame
           for col in df.columns:
                day_of_week = df[col]['Day']  # Get the day of the week (e.g., 'Monday')
                sales_value = df[col]['Predicted Sales']  # Get the sales value for that day
                # Add the sales value to the corresponding day in weekly_sales_sum
                weekly_sales_sum[day_of_week] += sales_value
    
    # Convert the dictionary to a DataFrame for easier viewing if needed
    weekly_sales_sum_df = pd.DataFrame.from_dict(weekly_sales_sum, orient='index', columns=['Total Sales'])
    weekly_sales_sum_df = weekly_sales_sum_df.T
    for month, df in monthly_data.items():
        if month in exclude_months:
            # Only include months that are not in the exclude list
            # Define the days of the week
            days_of_week = list(calendar.day_name)
            
            # Create a DataFrame with the days of the week as the index and initialize all values to 0
            excluded_months_sales_df = pd.DataFrame(0, columns=days_of_week, index=['Total Sales'])
            store = monthly_sales.index.get_level_values('Store')
    
            # Check if Predicted Sales is 0 and call the function
            if df.loc['Predicted Sales'].sum() == 0:
                monthly_data = real_data_production(ratio_for_real_data, store, header_list, month, selected_elements, excluded_months_sales_df, monthly_data)
    ###############################################################################################
    # Create a dictionary mapping days to average sales
    # Iterate over the June DataFrame and fill in the predicted sales
    # Mapping of items to their respective week intervals
    item_intervals = {
        'QUEEN CAMPANULA': (3, 18),
        'QUEEN EUPHORBIA MILII': (2, 51),
        'QUEEN KALANCHOE Q21': (23, 35),
        'QUEEN POINSETTIA Q10,5': (42, 52),
    }
    
    # Special case for 'QUEEN MİNİ GÜL'
    if item == 'QUEEN MİNİ GÜL':
        dict1 = week_based_interval_estimation_for_flowers(42, 52, monthly_data, item)
        dict2 = week_based_interval_estimation_for_flowers(2, 18, monthly_data, item)
        
        # Merge dict2 into dict1
        for month, df in dict2.items():
            if month in dict1:
                dict1[month] = dict1[month].add(df)
            else:
                dict1[month] = df
        
        monthly_data = dict1
    else:
        # Handle other items using the mapping
        if item in item_intervals:
            start_week, end_week = item_intervals[item]
            monthly_data = week_based_interval_estimation_for_flowers(start_week, end_week, monthly_data, item)
    
    #print(monthly_data)
    #Selecting a Specific Row and a Specific Column:
    #average_sales.iloc[day_of_week, column_index]
    #Display the updated June DataFrame
    #print(monthly_data["June"].loc["Predicted Sales", :])
    
    return monthly_data

def plot_yearly_distribution(monthly_data, months, store_and_code_pair, flowers, i, j):
    yearly_distribution = pd.Series(dtype = 'float64')
    for month in months:
        if yearly_distribution.empty:
            yearly_distribution = monthly_data[month].loc["Predicted Sales", :]
        else:
            yearly_distribution = pd.concat([yearly_distribution, monthly_data[month].loc["Predicted Sales", :]])
            
    yearly_distribution.index = pd.to_datetime(yearly_distribution.index)
    # Create a color palette with 12 distinct colors for the months
    colors = sns.color_palette("tab20", 12)

    # Create a list of colors based on the month
    bar_colors = [colors[month-1] for month in yearly_distribution.index.month]

    # Set the plot size
    plt.figure(figsize=(12, 6))

    # Plot the bar plot with the assigned colors
    plt.bar(yearly_distribution.index, yearly_distribution.values, color=bar_colors)

    # Customize the plot
    plt.title(f'Yearly Distribution for {flowers[j]} at {store_and_code_pair[i][1]}')
    plt.xlabel('Time')
    plt.ylabel('Values')

    # Set x-ticks to the first day of each month
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.grid(True)
    plot = plt.show()
    return plot

def plot_yearly_distribution_per_flower(pivot_df):
        # Create a color palette with 12 distinct colors for the months
        # Convert the 'Date' column to datetime format
        pivot_df['Date'] = pd.to_datetime(pivot_df['Date'])
        colors = sns.color_palette("tab20", 12)
        
        # Create a list of colors based on the month in the 'Date' column
        bar_colors = [colors[date.month - 1] for date in pivot_df['Date']]
        
        # Set the plot size
        plt.figure(figsize=(12, 6))
        
        # Plot the bar plot with the assigned colors
        plt.bar(pivot_df['Date'], pivot_df['row_sum'], color=bar_colors)
        
        # Customize the plot
        plt.title(f'Yearly Distribution for {pivot_df["flower_names"].iloc[0]}')
        plt.xlabel('Time')
        plt.ylabel('Values')
        
        # Set x-ticks to the first day of each month
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        
        plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
        plt.grid(True)
        plot = plt.show()

        return plot

def yearly_preciton_per_flower_per_store(flowers):
    for j, item in enumerate(flowers):
        list_of_dicts = []
        for i in range(len(store_and_code_pair)):       
            monthly_sales = monthly_sales_per_Store(filtered_data, store_and_code_pair, i, j)
            prep_row = prep_organized_data.loc[[(store_and_code_pair[i][0], store_and_code_pair[i][1], item)]]
            prep_ratio_row = prep_data_ratio.loc[[(store_and_code_pair[i][0], store_and_code_pair[i][1])]] 
            monthly_data = main(monthly_sales, filtered_data, data_ratio, months, sum_of_days_ratio, sum_of_days, prep_row, prep_ratio_row, months_array, item)
            list_of_dicts.append(monthly_data)
            plot = plot_yearly_distribution(monthly_data, months, store_and_code_pair, flowers, i, j)
    return 0

def yearly_prediction_per_flower(flowers, filtered_data):
    unique_store_names = filtered_data['Account'].unique()
    for j, item in enumerate(flowers):
        all_data = []
        list_of_dicts = []
        for i in range(len(store_and_code_pair)):       
            monthly_sales = monthly_sales_per_Store(filtered_data, store_and_code_pair, i, j)
            prep_row = prep_organized_data.loc[[(store_and_code_pair[i][0], store_and_code_pair[i][1], item)]]
            prep_ratio_row = prep_data_ratio.loc[[(store_and_code_pair[i][0], store_and_code_pair[i][1])]] 
            monthly_data = main(monthly_sales, filtered_data, data_ratio, months, sum_of_days_ratio, sum_of_days, prep_row, prep_ratio_row, months_array, item)
            list_of_dicts.append(monthly_data)
            for month, data in monthly_data.items():
               dates = list(data.keys())
               days = data.loc['Day']
               store_names = store_and_code_pair[i][1]
               flower_names = flowers[j]
               predicted_sales = data.loc['Predicted Sales']
               
               value_as_list = [[date.strftime('%Y-%m-%d'), day, store_names, flower_names, sale] 
                                 for date, day, sale in zip(dates, days, predicted_sales)]        
               all_data.extend(value_as_list)
        df = pd.DataFrame(all_data, columns = ['Date', 'Day', 'store_names', 'flower_names', 'Predicted Sales'])
        df = df.sort_values(by='Date')
        #Create a unique identifier for each shop and prediction combination
        df['shop_prediction'] = df['store_names'] + '_' + df.groupby(['Date', 'Day', 'flower_names', 'store_names']).cumcount().astype(str)
        
        # Pivot the dataframe so that each date has its corresponding 'Predicted Sales' in separate columns named by shop
        pivot_df = df.pivot_table(index=['Date', 'Day', 'flower_names'],
                                  columns='shop_prediction',
                                  values='Predicted Sales').reset_index()
        # Flatten the multi-level column index if needed
        pivot_df.columns = ['_'.join(map(str, col)).strip() if isinstance(col, tuple) else col for col in pivot_df.columns.values]
        numeric_sums = pivot_df.select_dtypes(include=['number']).sum(axis=1)
        pivot_df['row_sum'] = numeric_sums
        pivot_df = pivot_df.drop(columns=[col for col in pivot_df.columns if col in unique_store_names])         
        plot_yearly_distribution_per_flower(pivot_df)
    return 0    

def calculate_profit_per_store(flowers):
    
    return 0

def export_sales_per_flower_to_excel(flowers):
    for j, item in enumerate(flowers):
        all_data = []
        list_of_dicts = []
        for i in range(len(store_and_code_pair)):       
            monthly_sales = monthly_sales_per_Store(filtered_data, store_and_code_pair, i, j)
            prep_row = prep_organized_data.loc[[(store_and_code_pair[i][0], store_and_code_pair[i][1], item)]]
            prep_ratio_row = prep_data_ratio.loc[[(store_and_code_pair[i][0], store_and_code_pair[i][1])]] 
            monthly_data = main(monthly_sales, filtered_data, data_ratio, months, sum_of_days_ratio, sum_of_days, prep_row, prep_ratio_row, months_array)
            list_of_dicts.append(monthly_data)
            for month, data in monthly_data.items():
               dates = list(data.keys())
               days = data.loc['Day']
               store_names = store_and_code_pair[i][1]
               flower_names = flowers[j]
               predicted_sales = data.loc['Predicted Sales']
               
               value_as_list = [[date.strftime('%Y-%m-%d'), day, store_names, flower_names, sale] 
                                 for date, day, sale in zip(dates, days, predicted_sales)]        
               all_data.extend(value_as_list)
        df = pd.DataFrame(all_data, columns = ['Date', 'Day', 'store_names', 'flower_names', 'Predicted Sales'])
        df = df.sort_values(by='Date')
        
        #Create a unique identifier for each shop and prediction combination
        df['shop_prediction'] = df['store_names'] + '_' + df.groupby(['Date', 'Day', 'flower_names', 'store_names']).cumcount().astype(str)
        
        # Pivot the dataframe so that each date has its corresponding 'Predicted Sales' in separate columns named by shop
        pivot_df = df.pivot_table(index=['Date', 'Day', 'flower_names'],
                                  columns='shop_prediction',
                                  values='Predicted Sales').reset_index()
        # Flatten the multi-level column index if needed
        pivot_df.columns = ['_'.join(map(str, col)).strip() if isinstance(col, tuple) else col for col in pivot_df.columns.values]
        pivot_df.to_excel(f'{item}.xlsx', index=False)       
    return 0


(filtered_data, summed_sales) = prepare_dataframe(data)
#############Sum of total daily sales###############
total_sales_df = sum_of_daily_sale(summed_sales)
#############Sum of every day###############
sum_of_days = sum_of_everyday(filtered_data, summed_sales, total_sales_df)
#########ratio of everyday###############
sum_of_days_ratio = ratio_of_everyday(sum_of_days)
############Sum of everyday per Store#############
store_sum_of_days = sum_of_everyday_per_store(summed_sales)
############ratios of stores per day############## 
store_sum_of_days_ratio = ratios_per_store(store_sum_of_days)
####################################################################################################
(flowers, store_and_code_pair, prep_organized_data, prep_data_ratio) = create_prep_row_and_prep_ratio_row(filtered_data, data_ratio, months, sum_of_days_ratio, sum_of_days, months_array )
#####################Yearly Prediction of Yearly Sales per Store and per Flwoer#####################
yearly_preciton_per_flower_per_store(flowers)
yearly_prediction_per_flower(flowers, filtered_data)
###############################################################################
df_financial_sheet2.drop(columns=['Mal No', 'Saksı Çapı (cm)'], inplace=True)
df_financial_sheet1 = df_financial_sheet1.rename(columns={'Fiyat (KDV Hariç)' : 'Fiyat', 'Ürün Adı' : 'Ürün'})
df_financial_sheet2 = df_financial_sheet2.rename(columns={'Mal' : 'Ürün'})
#df_price_sheet = pd.concat([df_financial_sheet1, df_financial_sheet2], axis = 0)
#df_price_sheet.set_index('Ürün', inplace = True)
#df_price_sheet.iloc[0, 0] = 18.75
#df_price_sheet.to_excel('price_sheet.xlsx', index=True)
for i in range(len(store_and_code_pair)):
    store_name = store_and_code_pair[i][1]
    list_for_total_sale_of_shops = []
    date_range = pd.date_range(start = '2024-01-01', end='2024-12-12')
    result = pd.DataFrame()
    for j, item in enumerate(flowers):
        monthly_sales = monthly_sales_per_Store(filtered_data, store_and_code_pair, i, j)
        prep_row = prep_organized_data.loc[[(store_and_code_pair[i][0], store_and_code_pair[i][1], item)]]
        prep_ratio_row = prep_data_ratio.loc[[(store_and_code_pair[i][0], store_and_code_pair[i][1])]] 
        monthly_data = main(monthly_sales, filtered_data, data_ratio, months, sum_of_days_ratio, sum_of_days, prep_row, prep_ratio_row, months_array, item)
        df = pd.concat(monthly_data, axis = 1)
        df.columns = df.columns.droplevel(0)
        df.drop('Day', inplace=True)
        multi_index = [(f'{store_name}', f'{item}')]
        multi_index = pd.MultiIndex.from_tuples(multi_index, names=['Store Name', 'Flower Name'])
        df.set_index(multi_index, inplace=True)
        list_for_total_sale_of_shops.append(df)
         
    result = pd.concat(list_for_total_sale_of_shops, axis=0)
    result = result.apply(pd.to_numeric, errors='coerce')
    row_sums = result.sum(axis=0)
    row_sums = row_sums.to_frame()
    row_sums = row_sums.rename(columns = {0 : 'Predictions'})
    print(row_sums.loc[('Predicted Sales', 'KORU SİTESİ ANKARA MMM MİGROS', 'QUEEN ECHEVERIA Q5,5')])
############total sales of mini kalanchoes in stores##############
#####################All the mini Kalanchoes###################
'''
YILA TAŞINMAMASI GEREKEN ÜRÜNLER
-Campanula(week3-week18)
-Euphorbia MİLİ(WEEK2-WEEK51)
-EUPHORBİA 21CM(WEEK23-WEEK35)
-Hibiscus 10.5(week9-week18)
-Hibiscus 21(week23-week35)
-Kalanchoe 21(week 23-35)
-POINSETTIA (Week 42-52)
-gül(week2-week18, week40-endofyear)
-schlumbergera(week43-week52)
'''

###############################################################################
for j, item in enumerate(flowers):
    list_of_dicts = []
    item = 'QUEEN MİNİ KALANCHOE'
    for i in range(len(store_and_code_pair)):       
        monthly_sales = monthly_sales_per_Store(filtered_data, store_and_code_pair, 3, 0)
        prep_row = prep_organized_data.loc[[(7100, 'BUSE ESKİŞEHİR MM MİGROS', item)]]
        prep_ratio_row = prep_data_ratio.loc[[(7100, 'BUSE ESKİŞEHİR MM MİGROS')]]
        ################Calculate the Coefficients for a single shop for every month################
        coefficient = pd.DataFrame(index = prep_ratio_row.index, columns = prep_ratio_row.columns)
        dataframes = []
        for i in range(prep_ratio_row.shape[1]):
            coefficient = pd.DataFrame(index = prep_ratio_row.index, columns = prep_ratio_row.columns)
            for j in range(prep_ratio_row.shape[1]):
                coefficient.iloc[:, j] = prep_ratio_row.iloc[:, i] / prep_ratio_row.iloc[:, j]
            dataframes.append(coefficient)   
        dataframes
        combined_coeffcients = pd.concat(dataframes, axis=0, ignore_index = True)
        combined_coeffcients#We have gotten all the coefficients for every month for our spesific store and spesific flower type    monthly_sales
        months_array = np.round(months_array).astype(int) 
        ##############################################################################################################
        for i in range(monthly_sales.shape[1]):
            months_array[i+ 2] = int(monthly_sales.iloc[:, i].iloc[0])

        months_array[2] = months_array[2] * 2
        months_array[5] = months_array[5] * 6/5
        months_array_copy = months_array.copy()

        for i in range(0, 2):
            months_array[i] = (months_array_copy[2]*combined_coeffcients.iloc[i, 2] + months_array_copy[3]*combined_coeffcients.iloc[i, 3] +
            months_array_copy[4]*combined_coeffcients.iloc[i, 4] + months_array_copy[5]*combined_coeffcients.iloc[i, 5])/4
            #print(monthly_sales.shape[1])
        for i in range(6, 12):
            months_array[i] = (months_array_copy[2]*combined_coeffcients.iloc[i, 2] + months_array_copy[3]*combined_coeffcients.iloc[i, 3] +
            months_array_copy[4]*combined_coeffcients.iloc[i, 4] + months_array_copy[5]*combined_coeffcients.iloc[i, 5])/4
            #print(monthly_sales.shape[1])

        # Create a date range for the headers
        date_range = pd.date_range(start='2024-01-01', periods=12, freq='M')
        index_values = ['Monthly Total Sales']
        # Convert the array to a DataFrame and set the formatted date headers
        monthly_total_sales_df = pd.DataFrame([months_array], index = index_values, columns=[date.strftime('%Y-%m') for date in date_range])  
        month_names = pd.Series(date_range).dt.month_name()  
        monthly_total_sales_df.loc['Month'] = month_names.values
        monthly_total_sales_df = monthly_total_sales_df.reindex(['Month'] + index_values)
        ############################################################################################
        ################Generate July################
        monthly_data = {}
        # Define the index for the DataFrame
        index_values = ['Predicted Sales']
        # Loop through each month of the year 2024
        for month in range(1, 13):
            # Generate date range for the current month
            year = 2024
            num_days = calendar.monthrange(year, month)[1]
            date_range = pd.date_range(start=f'{year}-{month:02d}-01', end=f'{year}-{month:02d}-{num_days}', freq='D')
            # Create a DataFrame with the specified index and date range as columns
            df = pd.DataFrame(index=index_values, columns=date_range)   
            # Extract day names from the date range
            day_names = pd.Series(date_range).dt.day_name() 
            # Add day names as a new row in the DataFrame
            df.loc['Day'] = day_names.values   
            # Reindex to place 'Day' at the top
            df = df.reindex(['Day'] + index_values)
            # Store the DataFrame in the dictionary with the month name as the key
            monthly_data[date_range[0].strftime('%B')] = df
        # Display the DataFrame for a specific month, e.g., July
        weekly_Sales_num = sum_of_days_ratio.copy()
        weekly_Sales_num = pd.concat([weekly_Sales_num]*12, ignore_index=False)
        for i in range(monthly_total_sales_df.shape[1]):
            for j in range(weekly_Sales_num.shape[1]):
                weekly_Sales_num.iloc[i, j] *= months_array[i] 
        weekly_Sales_num.rename(index = {'Total Sales per Day' : 'Day Count'}, inplace = True)
        weekly_Sales_num.index = months
        ##########################Count the number of Days in a Month###############
        monthly_day_counts = pd.DataFrame(np.nan, index=weekly_Sales_num.index, columns=weekly_Sales_num.columns)
        monthly_data#this is our empty dictionary where we will fill with our predited sales
        for month in months:
            day_row = monthly_data[month].loc['Day']
            day_counts = day_row.value_counts()
            day_counts = day_row.value_counts()
            day_counts = day_counts.to_frame(name=f'Day Count of month {months[i]}')
            day_counts = day_counts.T
            desired_order = list(calendar.day_name)   
            day_counts = day_counts.reindex(columns=desired_order)
            monthly_day_counts.loc[month] =  day_counts.values.flatten()
        #################Now we find the weight for every single month#################
        average_sales = weekly_Sales_num.copy()
        weight_df = pd.DataFrame(np.nan, index = ['Weights for months'], columns = months)
        for i in range(len(average_sales)):
            weight = 0
            for j in range(sum_of_days_ratio.shape[1]):
                weight += (monthly_day_counts.values[i, j]*sum_of_days_ratio.values[:,j])
            weight_df.loc['Weights for months', months[i]] = weight
        ################Give the values###############################################
        for i in range(len(average_sales)):
            for j in range(average_sales.shape[1]):
                average_sales.iloc[i, j] = (months_array[i]*(monthly_day_counts.values[i, j]*sum_of_days_ratio.values[:,j]))/weight_df.values[:, i]
                average_sales.iloc[i, j] = average_sales.values[i, j] / monthly_day_counts.values[i, j]
        ###############################################################################
        # Create a dictionary mapping days to average sales
        # Iterate over the June DataFrame and fill in the predicted sales
        for month in months:
            average_sales_dict = average_sales.to_dict('index')[month]
            for col in monthly_data[month].columns:
                day_of_week = monthly_data[month][col]['Day']
                monthly_data[month][col]['Predicted Sales'] = average_sales_dict.get(day_of_week, None)
        ###############################################################################################################################        
        # Initialize a DataFrame with days of the week as columns and a single row for total sales
        weekly_sales_df = pd.DataFrame(index=['Total Sales'], columns=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        
        # Fill the DataFrame with zeros initially
        weekly_sales_df.loc['Total Sales'] = 0
        
        
        
        
        ratio_for_real_data = data_ratio.copy()
        ratio_for_real_data = ratio_for_real_data.drop(columns=['Store ID'])
        ratio_for_real_data.set_index('Store Name', inplace = True)
        header_list = ratio_for_real_data.columns.tolist()
        selected_elements = [header_list[i] for i in [0, 1, 6, 7, 8, 9, 10, 11]]
        # List of months to exclude
        exclude_months = ['March', 'April', 'May', 'June']
        # Initialize a dictionary to hold the sum of sales for each day of the week
        weekly_sales_sum = {day: 0 for day in calendar.day_name}
    
        # Loop through all months in the monthly_data dictionary
        for month, df in monthly_data.items():
            if month not in exclude_months:
                         
                # Loop through the columns (dates) of the DataFrame
               for col in df.columns:
                    day_of_week = df[col]['Day']  # Get the day of the week (e.g., 'Monday')
                    sales_value = df[col]['Predicted Sales']  # Get the sales value for that day
                    # Add the sales value to the corresponding day in weekly_sales_sum
                    weekly_sales_sum[day_of_week] += sales_value
        
        # Convert the dictionary to a DataFrame for easier viewing if needed
        weekly_sales_sum_df = pd.DataFrame.from_dict(weekly_sales_sum, orient='index', columns=['Total Sales'])
        weekly_sales_sum_df = weekly_sales_sum_df.T
        for month, df in monthly_data.items():
            if month in exclude_months:
                # Only include months that are not in the exclude list
                # Define the days of the week
                days_of_week = list(calendar.day_name)
                
                # Create a DataFrame with the days of the week as the index and initialize all values to 0
                excluded_months_sales_df = pd.DataFrame(0, columns=days_of_week, index=['Total Sales'])
                store = monthly_sales.index.get_level_values('Store')
                # Check if Predicted Sales is 0 and call the function
                if df.loc['Predicted Sales'].sum() == 0:
                    monthly_data = real_data_production(ratio_for_real_data, store, header_list, month, selected_elements, excluded_months_sales_df, monthly_data)
        ###############################################################################################
        # Create a dictionary mapping days to average sales
        # Iterate over the June DataFrame and fill in the predicted sales
        # Mapping of items to their respective week intervals
        item_intervals = {
            'QUEEN CAMPANULA': (3, 18),
            'QUEEN EUPHORBIA MILII': (2, 51),
            'QUEEN KALANCHOE Q21': (23, 35),
            'QUEEN POINSETTIA Q10,5': (42, 52),
        }
        
        # Special case for 'QUEEN MİNİ GÜL'
        if item == 'QUEEN MİNİ GÜL':
            dict1 = week_based_interval_estimation_for_flowers(42, 52, monthly_data, item)
            dict2 = week_based_interval_estimation_for_flowers(2, 18, monthly_data, item)
            
            # Merge dict2 into dict1
            for month, df in dict2.items():
                if month in dict1:
                    dict1[month] = dict1[month].add(df)
                else:
                    dict1[month] = df
            
            monthly_data = dict1
        else:
            # Handle other items using the mapping
            if item in item_intervals:
                start_week, end_week = item_intervals[item]
                monthly_data = week_based_interval_estimation_for_flowers(start_week, end_week, monthly_data, item)
       
        for month, data in monthly_data.items():
            plt.figure(figsize=(10, 6))
            plt.plot(data.columns, data.loc['Predicted Sales'], marker='o')
            plt.title(f"Predicted Sales for {month}")
            plt.xlabel("Date")
            plt.xticks(rotation=45)  # Rotate the date labels for better readability
            plt.ylabel("Predicted Sales")
            plt.grid(True)
            plt.tight_layout()  # Adjust layout to prevent overlap
            plt.show()


 










