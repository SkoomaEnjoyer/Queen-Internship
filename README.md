Here's a detailed `README.md` for your project:

---

# Yearly Sales Prediction and Visualization for Flower Stores

## Overview
This project is a comprehensive Python-based tool designed for predicting and visualizing yearly sales data for flower stores. The code reads multiple Excel datasets, processes the data, calculates predictive sales metrics, and generates detailed visual outputs. The primary goal is to support data-driven decision-making for flower sales management.

## Features
- **Data Integration**: Combines data from various Excel files to create a unified dataset.
- **Data Processing**: Cleans, filters, and aggregates data to make it suitable for analysis.
- **Prediction Algorithms**: Implements custom algorithms to forecast sales based on historical data.
- **Weekly and Monthly Analysis**: Breaks down sales predictions by days of the week and months to capture seasonal patterns.
- **Visualization**: Uses `matplotlib` and `seaborn` to create charts for sales distribution and prediction analysis.
- **Exclusion Logic**: Identifies specific products that should not be forecasted for certain periods based on predefined rules.

## Data Requirements
- **`forecasting_data_phase2.xlsx`**: Contains initial sales data for processing.
- **`total_ratio.xlsx`**: Provides additional ratio data for calculating weighted sales predictions.
- **`finansal_hesaplama.xlsx`**: Includes financial information spread across multiple sheets for comprehensive analysis.

## Key Functions
### Data Preparation
- **`prepare_dataframe()`**: Cleans and aggregates data for initial processing.
- **`sum_of_daily_sale()`**: Calculates the sum of daily sales across stores.
- **`sum_of_everyday()` and `sum_of_everyday_per_store()`**: Groups daily sales by day of the week and by store, respectively.

### Prediction and Analysis
- **`yearly_preciton_per_flower_per_store()`**: Generates annual sales predictions for each flower type per store.
- **`week_based_interval_estimation_for_flowers()`**: Customizes sales predictions for specific time intervals based on product characteristics.
- **`real_data_production()`**: Integrates real sales data into predictions to refine accuracy.

### Visualization
- **`plot_yearly_distribution()`**: Creates bar plots showing the yearly sales distribution for each product and store.
- **`plot_yearly_distribution_per_flower()`**: Visualizes sales predictions for a specific flower across the entire year.

## Contact
**Kerem Mehmet Budanaz**  
[LinkedIn](https://www.linkedin.com/in/kerem-mehmet-budanaz-298297220)  
Email: kerembudanaz@sabanciuniv.edu  

---
