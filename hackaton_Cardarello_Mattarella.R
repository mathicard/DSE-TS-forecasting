# Time Series Forecasting: Machine Learning and Deep Learning with R & Python ----
# 12 March 2022


# Packages ----------------------------------------------------------------

setwd("/Users/mathiascardarellofierro/Documents/DSE/TS lab/tsforecasting-course-master")
source("R/utils.R")
source("R/packages.R") #error with catboost

library(tidymodels)
library(modeltime)
library(modeltime.ensemble)
library(quantmod)


# Data --------------------------------------------------------------------

dataset <- read_rds("data/hackathon_dataset.rds")
# we have 20 series per each period type


## Part 1: Manipulation, Transformation & Visualization -----------------

# Manipulation ------------------------------------------------------------

# daily
dataset_daily <- dataset$data %>%
  filter(period == 'Daily')


# Visualization ------------------------------------------------------

dataset$data %>%
  filter(period == 'Daily') %>%
  group_by(id) %>%
  plot_time_series(date, value, 
                   .facet_ncol = 5, .facet_scales = "free",
                   .interactive = TRUE)

# Log Transforms
dataset$data %>%
  filter(period == 'Daily') %>%
  group_by(id) %>%
  plot_time_series(date, log(value+1), 
                   .facet_ncol = 5, .facet_scales = "free",
                   .interactive = TRUE)


# * Autocorrelation Function (ACF) Plot -----------------------------------

dataset$data %>%
  filter(period == 'Daily') %>%
  group_by(id) %>%
  plot_acf_diagnostics(date, log(value + 1), .lags = 10, 
                       .facet_ncol = 3, .facet_scales = "free",
                       .show_white_noise_bars = TRUE)


# * Smoothing Plot --------------------------------------------------------

dataset$data %>%
  filter(period == 'Daily') %>%
  group_by(id) %>%
  plot_time_series(date, log(value+1), 
                   .smooth_period = "90 days",
                   .smooth_degree = 1,
                   .facet_ncol = 5, .facet_scales = "free",
                   .interactive = TRUE)


# * Seasonality Plot ------------------------------------------------------

dataset$data %>%
  filter(period == 'Daily') %>%
  group_by(id) %>%
  plot_seasonal_diagnostics(date, log(value + 1))


# * Decomposition Plot ----------------------------------------------------

dataset$data %>%
  filter(id == "D1017") %>%
  plot_stl_diagnostics(date, log(value + 1))


# * Anomaly Detection Plot ------------------------------------------------

dataset$data %>%
  filter(id == "D1017") %>%
  tk_anomaly_diagnostics(date, value, .alpha = .01, .max_anomalies = .01)


# * Time Series Regression Plot -------------------------------------------

dataset$data %>%
  filter(id == "D1017") %>%
  plot_time_series_regression(
    date,
    log(value + 1) ~
      as.numeric(date) + # linear trend
      lubridate::wday(date, label = TRUE) + # week day calendar features
      lubridate::month(date, label = TRUE), # month calendar features
    .show_summary = TRUE
  )


# * Pad by Time -----------------------------------------------------------

# - Filling in time series gaps

# fill daily gaps
dataset$data %>%
  filter(period == 'Daily') %>%
  group_by(id) %>%  
  pad_by_time(.date_var = date, .pad_value = 0)


# Recipes -----------------------------------------------------------------

# Splitting Data
(dataset$data %>%
  filter(period == 'Daily') %>%
  group_by(id) %>%  
  tk_summary_diagnostics())


#14 days to forecast

nested_daily_data <- dataset$data %>%
  filter(period == 'Daily', type == 'train') %>%
  select(id, date, value) %>%
  # Step 1: Extend
  extend_timeseries(
    .id_var        = id,
    .date_var      = date,
    .length_future = 14
  ) %>%
  # Step 2: Nest
  nest_timeseries(
    .id_var        = id,
    .length_future = 14,
    ) %>%
  
  # Step 3: Splitting
  split_nested_timeseries(
    .length_test = 14
  )
  
    nested_daily_data


## Part 2: Create Tidymodels Workflows -----------------

#First, we create tidymodels workflows for the various models 


# Prophet ------------------------------------------------------------
    
rec_prophet <- recipe(value ~ date, training(nested_daily_data$.splits[[1]])) 

wflw_prophet <- workflow() %>%
  add_model(
    prophet_reg("regression", seasonality_yearly = TRUE) %>% 
      set_engine("prophet")
  ) %>%
  add_recipe(rec_prophet)






