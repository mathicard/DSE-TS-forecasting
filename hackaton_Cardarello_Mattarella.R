# Time Series Forecasting: Machine Learning and Deep Learning with R & Python ----
# 12 March 2022


# Packages ----------------------------------------------------------------

source("https://raw.githubusercontent.com/mathicard/DSE-TS-forecasting/main/utils.R")
source("https://raw.githubusercontent.com/mathicard/DSE-TS-forecasting/main/packages.R")
library(padr)

# Data --------------------------------------------------------------------

dataset <- read_rds("https://github.com/mathicard/DSE-TS-forecasting/blob/main/hackathon_dataset.rds?raw=true")

# we have 20 series per each period of time


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
dataset_hourly <- dataset$data %>%
  filter(period == "Hourly") %>%
  group_by(id) %>%
  pad_by_time(.date_var = date, .by = "hour", .pad_value = 0)


dataset_daily <- dataset$data %>%
  filter(period == "Daily") %>%
  group_by(id) %>%
  pad_by_time(.date_var = date, .by = "day", .pad_value = 0)


dataset_weekly <- dataset$data %>%
  filter(period == "Weekly") %>%
  group_by(id) %>%
  pad_by_time(.date_var = date, .by = "7 days", .pad_value = 0)


dataset_monthly <- dataset$data %>%
  filter(period == "Monthly") %>%
  thicken('month')%>%
  group_by(id)%>%
  pad_by_time(.date_var = date_month, .by = "month", .pad_value = 0)


dataset_yearly <- dataset$data %>%
  filter(period == "Yearly") %>%
  group_by(id) %>%
  pad_by_time(.date_var = date, .by = "year", .pad_value = 0)



dataset_quarterly <- dataset$data %>%
  filter(period == "Quarterly")  %>%
  thicken('quarter') %>%
  group_by(id)%>%
  pad_by_time(.date_var = date_quarter, .by = "quarter", .pad_value = 0)



# Feature Engineering -----------------------------------------------------

year_time <- dataset_yearly %>%
  tk_augment_timeseries_signature() %>%
  select(
    -diff, -ends_with("iso"), -ends_with(".xts"), -contains("hour"),
    -contains("minute"), -contains("second"), -contains("am.pm")
  )
data_prep_signature_tbl %>% glimpse()


# linear trend
year_time %>%
  filter(id == "Y10216")%>%
  plot_time_series_regression(date, .formula = value ~ index.num)

# nonlinear trend - basis splines
year_time %>%
  filter(id == "Y10216")%>%
  plot_time_series_regression(
    date,
    value ~ splines::bs(index.num, df = 3),
    .show_summary = TRUE
  )

# nonlinear trend - natural splines
year_time %>%
  filter(id == "Y10216")%>%
  plot_time_series_regression(
    date,
    value ~ splines::ns(index.num, knots = quantile(index.num, probs = c(0.25, 0.5))),
    .show_summary = TRUE
  )


# weekly seasonality
dataset_weekly <- dataset_weekly %>%
  tk_augment_timeseries_signature() %>%
  select(
    -diff, -ends_with("iso"), -ends_with(".xts"), -contains("hour"),
    -contains("minute"), -contains("second"), -contains("am.pm")
  )

dataset_weekly %>%
  filter(id == "W118") %>%
  plot_time_series_regression(date, value ~ week, .show_summary = TRUE)



# * Interaction Features --------------------------------------------------

dataset_weekly %>%
  plot_time_series_regression(
    date,
    value ~ (as.factor(week2) * wday.lbl),
    .show_summary = TRUE
  )


# * Rolling Average Features ----------------------------------------------

# - tk_augment_slidify

data_prep_rolls_tbl <- dataset_daily %>%
  tk_augment_slidify(
    value, mean,
    .period = c(7, 14, 30, 90),
    .align = "center",
    .partial = TRUE
  )
data_prep_rolls_tbl %>% glimpse()



# * Lag Features ----------------------------------------------------------

# tk_augment_lags

dataset_daily %>%
  plot_acf_diagnostics(date, value, .lags = 100)

data_prep_lags_tbl <- dataset_daily %>%
  tk_augment_lags(value, .lags = c(1, 7, 14, 30, 90, 365)) %>%
  drop_na()
data_prep_lags_tbl %>% glimpse()


# * Fourier Series Features -----------------------------------------------

# - tk_augment_fourier

dataset_daily %>%
  plot_acf_diagnostics(optin_time, optins_trans, .lags = 100)

data_prep_fourier_tbl <- dataset_daily %>%
  tk_augment_fourier(date, .periods = c(1, 7, 14, 30, 90, 365), .K = 2)
data_prep_fourier_tbl %>% glimpse()

data_prep_fourier_tbl %>%
  filter(id == 'D1017')%>%
  select(-c(type,period))%>%
  plot_time_series_regression(date, value ~ ., .show_summary = TRUE)


# Recipes -----------------------------------------------------------------

# Splitting Data
(dataset$data %>%
   filter(period == 'Daily') %>%
   group_by(id) %>%
   tk_summary_diagnostics())


#14 days to forecast

nested_daily_data <- dataset_daily %>%
  select(id, date, value) %>%
  # Step 1 - Nest: We'll predict 14 days into the future
  nest_timeseries(
    .id_var        = id,
    .length_future = 14
  ) %>%

  # Step 2 - Splitting:
  split_nested_timeseries(
    .length_test = 14
  )

nested_daily_data


## Part 2.A: Create Tidymodels Workflows -----------------

#First, we create tidymodels workflows for the various models

# S-NAIVE & WINDOWS -------------------------------------------------------
# Compare the model against a predefined benchmark


# NAIVE: our forecast is just the most recent observation in time

rec_naive <- recipe(value ~ date, extract_nested_train_split(nested_daily_data)) %>%
  # Time Series Signature
  step_timeseries_signature(date) %>%
  step_rm(matches("(iso)|(xts)|(hour)|(minute)|(second)|(am.pm)")) %>%
  step_normalize(matches("(index.num)|(year)|(yday)")) %>%
  step_dummy(all_nominal(), one_hot = TRUE) %>%
  # Fourier
  step_fourier(date, period = c(7, 14, 30, 90, 365), K = 2)

rec_naive %>% prep() %>% juice() %>% glimpse()


# Spline Recipe
# - natural spline series on index.num
rcp_spec_spline <- rec_naive %>%
  step_ns(ends_with("index.num"), deg_free = 2) %>%
  step_rm(date) %>%
  step_rm(starts_with("lag_"))
rcp_spec_spline %>% prep() %>% juice() %>% glimpse()

# Lag Recipe
# - lags of optins_trans and rolls
rcp_spec_lag <- rec_naive %>%
  step_naomit(starts_with("lag_")) %>%
  step_rm(date)
rcp_spec_lag %>% prep() %>% juice() %>% glimpse()


# * Model Engine Specification --------------------------------------------

model_spec_lm <- linear_reg() %>%
  set_engine("lm")

# Workflow
wflw_naive <- workflow() %>%
  add_model(
    naive_reg() %>%
      set_engine("naive")
  ) %>%
  add_recipe(rec_naive)


# LM Spline Workflow
wrkfl_fit_lm_1_spline <- workflow() %>%
  add_model(model_spec_lm) %>%
  add_recipe(rcp_spec_spline) %>%
  fit(training(splits))
wrkfl_fit_lm_1_spline
wrkfl_fit_lm_1_spline %>%
  extract_fit_parsnip() %>%
  pluck("fit") %>%
  summary()

# LM Lag Workflow
wrkfl_fit_lm_2_lag <- workflow() %>%
  add_model(model_spec_lm) %>%
  add_recipe(rcp_spec_lag) %>%
  fit(training(splits))
wrkfl_fit_lm_2_lag
wrkfl_fit_lm_2_lag %>%
  extract_fit_parsnip() %>%
  pluck("fit") %>%
  summary()



# WINDOW - MEAN

rec_mean <- recipe(value ~ date, extract_nested_train_split(nested_daily_data))

wflw_mean <- workflow() %>%
  add_model(
    window_reg(window_size = 7) %>%
      set_engine("window_function",
        window_function = mean,
        na.rm = TRUE)
     )  %>%
    add_recipe(rec_mean)


# WINDOW - WEIGHTED MEAN (a moving average, based on the last 3 obs. for instance)

rec_wmean <- recipe(value ~ date, extract_nested_train_split(nested_daily_data))

wflw_wmean <- workflow() %>%
  add_model(
    window_reg(window_size = 7) %>%
      set_engine(
        "window_function",
        window_function = ~ sum(tail(.x, 3) * c(0.1, 0.3, 0.6))
      )
  )  %>%
  add_recipe(rec_mean)



# Prophet ------------------------------------------------------------
# A common modeling method is prophet, that can be created using prophet_reg().
# Note that we use the first nested_data_tbl$.splits[[1]]) to help us determine
# how to build features.

rec_prophet <- recipe(value ~ date, extract_nested_train_split(nested_daily_data))

wflw_prophet <- workflow() %>%
  add_model(
    prophet_reg("regression", seasonality_yearly = TRUE) %>%
      set_engine("prophet")
  ) %>%
  add_recipe(rec_prophet)


# XGBoost ------------------------------------------------------------
# Next, we can use a machine learning method that can get good results: XGBoost.
# We will add a few extra features in the recipe feature engineering step
# to generate features that tend to get better modeling results.

rec_xgb <- recipe(value ~ ., extract_nested_train_split(nested_daily_data)) %>%
  step_timeseries_signature(date) %>%
  step_rm(date) %>%
  step_zv(all_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE)

wflw_xgb <- workflow() %>%
  add_model(boost_tree("regression") %>% set_engine("xgboost")) %>%
  add_recipe(rec_xgb)



# * Modeltime -------------------------------------------------------------

# Calibration
calibration_tbl <- modeltime_table(
  wrkfl_fit_lm_1_spline,
  wrkfl_fit_lm_2_lag
) %>%
  update_model_description(1, "LM - Spline Recipe") %>%
  update_model_description(2, "LM - Lag Recipe") %>%
  modeltime_calibrate(new_data = testing(splits))
calibration_tbl


# Forecasting
calibration_tbl %>%
  modeltime_forecast(new_data = testing(splits), actual_data = dataset_daily) %>%
  plot_modeltime_forecast()

# Accuracy
calibration_tbl %>% modeltime_accuracy()

# Refitting
refit_tbl <- calibration_tbl %>%
  modeltime_refit(data = dataset_daily)

refit_tbl %>%
  modeltime_forecast(new_data = forecast_tbl, actual_data = data_prep_tbl) %>%
  mutate(
    across(
      .value:.conf_hi,
      .fns = ~ standardize_inv_vec(x = ., mean = std_mean, sd = std_sd)
    )
  ) %>%
  mutate(
    across(
      .value:.conf_hi,
      .fns = ~ log_interval_inv_vec(
        x = ., limit_lower = limit_lower, limit_upper = limit_upper, offset = offset
      )
    )
  ) %>%
  plot_modeltime_forecast()


## Part 2.B: Nested Modeltime Tables -----------------
# With a couple of modeling workflows in hand, we are now ready to test them
# on each of the time series.
# We start by using the modeltime_nested_fit() function,
# which iteratively fits each model to each of the nested time series
# train/test “.splits” column.


nested_modeltime_tbl <- modeltime_nested_fit(
  # Nested data
  nested_data = nested_daily_data,

  # Add workflows
  wflw_prophet,
  wflw_xgb
)

nested_modeltime_tbl



# Accuracy check ------------------------------------------------------------

tab_style_by_group <- function(object, ..., style) {

  subset_log <- object[["_boxhead"]][["type"]]=="row_group"
  grp_col    <- object[["_boxhead"]][["var"]][subset_log] %>% rlang::sym()

  object %>%
    tab_style(
      style = style,
      locations = cells_body(
        rows = .[["_data"]] %>%
          rowid_to_column("rowid") %>%
          group_by(!! grp_col) %>%
          filter(...) %>%
          ungroup() %>%
          pull(rowid)
      )
    )
}


# Now we can see which models are the winners,
# performing the best by group with the lowest RMSE (root mean squared error).

nested_modeltime_tbl %>%
  extract_nested_test_accuracy() %>%
  group_by(id) %>%
  table_modeltime_accuracy(.interactive = FALSE) %>%
  tab_style_by_group(
    rmse == min(rmse),
    style = cell_fill(color = "lightgreen")
  )


## Part 2.C: Make Ensembles -----------------

# Now that we’ve fitted submodels, our goal is to improve
# on the submodels by leveraging ensembles.

# Average Ensemble: We’ll give a go at an average ensemble using a simple mean
# with the ensemble_nested_average() function. We select type = "mean" for simple average.

nested_ensemble_1_tbl <- nested_modeltime_tbl %>%
  ensemble_nested_average(
    type           = "mean",
    keep_submodels = TRUE
    )

nested_ensemble_1_tbl

# We can check the accuracy again.

nested_ensemble_1_tbl %>%
  extract_nested_test_accuracy() %>%
  group_by(id) %>%
  table_modeltime_accuracy(.interactive = FALSE) %>%
  tab_style_by_group(
    rmse == min(rmse),
    style = cell_fill(color = "lightgreen")
  )



# Weighted Ensemble: Next, we can give a go at a weighted ensemble
# with the ensemble_nested_weighted() function.

nested_ensemble_2_tbl <- nested_ensemble_1_tbl %>%
  ensemble_nested_weighted(
    loadings        = c(2,1),
    metric          = "rmse",
    model_ids       = c(1,2),
    control         = control_nested_fit(allow_par = FALSE, verbose = TRUE)
  )

nested_ensemble_2_tbl

# Next, let’s check the accuracy on the new ensemble.

nested_ensemble_2_tbl %>%
  extract_nested_test_accuracy() %>%
  group_by(id) %>%
  table_modeltime_accuracy(.interactive = FALSE) %>%
  tab_style_by_group(
    rmse == min(rmse),
    style = cell_fill(color = "lightgreen")
  )


## Part 3: Best models -----------------

# Using the accuracy data, we can pick a metric and select the best model
# based on that metric.
# The available metrics are in the default_forecast_accuracy_metric_set().

best_nested_modeltime_tbl <- nested_ensemble_2_tbl %>%
  modeltime_nested_select_best(
    metric                = "rmse",
    minimize              = TRUE,
    filter_test_forecasts = TRUE
  )

# The best model selections can be accessed with extract_nested_best_model_report().

best_nested_modeltime_tbl %>%
  extract_nested_best_model_report() %>%
  table_modeltime_accuracy(.interactive = TRUE)


# Forecast plot: Once we’ve selected the best models,
# we can easily visualize the best forecasts by time series.

best_nested_modeltime_tbl %>%
  extract_nested_test_forecast() %>%
  group_by(id) %>%
  filter(id == "D1142") %>%
  plot_modeltime_forecast(
    .facet_ncol  = 1,
    .interactive = TRUE
  )




