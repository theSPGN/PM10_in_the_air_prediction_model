# %% load used libraries and data
library(tidymodels) # collection of packages for modeling
library(dplyr) # data manipulation
library(xgboost) # engine for boost_tree model
library(yardstick) # metrics for model
library(vip) # parameters importance

load("prepared_data.RData")

# %%
set.seed(213)
# %% Create model
# define xgboost/boost_tree model with one parameter tuned
# use all cores of cpu to train model faster
# set mode to regression problem
xgb_mod <-
    boost_tree(
        trees = tune(),
        mtry = tune(),
        min_n = tune(),
        learn_rate = tune()
    ) |>
    set_engine(
        engine = "xgboost",
        num.threads = parallel::detectCores() - 1
    ) |>
    set_mode("regression")
# %% Create recipe

# define which variables shouldn't be included
id_variables <- train_data |>
    select_if(is.numeric) |>
    cor(use = "complete.obs") |>
    as.data.frame() |>
    select(grimm_pm10) |>
    abs() |>
    filter(grimm_pm10 < 0.3) |>
    rownames()

# define roles and steps of recipe

# due to tmwr.org/pre-proc-table:
# step_dummy() xgboost require dummy unlike other tree based models
# works similar to one-hot-encoding
# step_zv() might be helpful because erase data that occurred once in set
# step_pca() might be helpful in xgboost model, because it de-correlate data
# threshold define % of total variance that should be covered

# due to intuition:
# month, day of week and hour may influence the pm10 level in the air
# step_rm(date) required to get rid off date in the model that is not used
# after step_date and step_time

# due to correlation value:
# all of id_variables (except ops_pm10) has too low correlation with grimm_pm10
# to include it as predictor, ops_pm10 has too high correlation and makes
#  regression problem too easy and useless

xgb_recipe <-
    recipe(grimm_pm10 ~ ., data = train_data) |>
    update_role(all_of(id_variables), new_role = "ID") |>
    step_naomit() |>
    step_time(date, features = c("hour")) |>
    step_rm(date) |>
    step_dummy(all_nominal_predictors()) |>
    step_zv(all_predictors()) |>
    step_pca(threshold = 0.9)

summary(xgb_recipe)

xgb_recipe |>
    prep() |>
    bake(train_data) |>
    glimpse()

# %% Create a workflow for xgboost model
xgb_workflow <-
    workflow() |>
    add_recipe(xgb_recipe) |>
    add_model(xgb_mod)

# %% Making a grid for xgboost tuning
min_n_values <- seq(1, 5, length.out = 5)
trees_values <- seq(50, 1000, length.out = 10)
mtry_values <- seq(5, 20, length.out = 10)
learn_rate_value <- seq(0.1, 0.5, length.out = 10)

xgb_grid <- crossing(
    min_n = min_n_values,
    trees = trees_values,
    mtry = mtry_values,
    learn_rate = learn_rate_value
)


# %% resample model
xgb_res <-
    xgb_workflow |>
    tune_grid(
        resamples = val_set,
        grid = xgb_grid,
        control = control_grid(save_pred = TRUE),
        metrics = metric_set(mae, rmse, rsq)
    )

xgb_top_models <-
    xgb_res |>
    show_best(metric = "mae", n = Inf) |>
    arrange(mean) |>
    mutate(mean = mean |> round(x = _, digits = 3))

xgb_top_models |> gt::gt()

xgb_best <-
    xgb_res |> select_best()

xgb_mae <- xgb_res |>
    show_best(metric = "mae", n = Inf) |>
    filter(.config == xgb_best$.config) |>
    select(
        min_n,
        .metric,
        mean
    )

xgb_mae
# %% Final model
xgb_best_mod <-
    xgb_workflow |>
    finalize_workflow(xgb_best)

# %% Last fit
xgb_fit <-
    xgb_best_mod |>
    last_fit(split = data_split)

# %% vip
xgb_fit |>
    extract_fit_parsnip() |>
    vip(num_features = 20) +
    scale_x_discrete(expand = c(0, 0)) +
    scale_y_continuous(expand = c(0, 0)) +
    geom_boxplot(color = "black", fill = "grey85")
# %% metrics
xgb_fit |>
    collect_metrics() |>
    select(-.config, -.estimator, ) |>
    add_row(.metric = c("mae", "rmse", "rsq"), .estimate = xgb_mae$mean)

# %% predictions
xgb_fit |> collect_predictions()
# %%
save(xgb_fit, file = "last_fit_xgboost.rdata")

# %% predictions for outer station data
xgb_workflow_ <- extract_workflow(xgb_fit)

predictions <- predict(xgb_workflow_, new_data = other_station_data)$.pred

results <- other_station_data |>
    mutate(.pred = predictions)

# results |> select(grimm_pm10, .pred)

metrics <- results |>
    metrics(truth = grimm_pm10, estimate = .pred)
print(metrics)

ggplot(results, aes(x = grimm_pm10, y = .pred)) +
    geom_point() +
    labs(
        title = "Wykres grimm_pm10 i .pred od daty",
        x = "real",
        y = "pred",
        color = "Legenda"
    ) +
    geom_abline(slope = 1, intercept = 0, color = "blue", linetype = "solid") +
    geom_abline(slope = 0.5, intercept = 0, color = "red", linetype = "solid") +
    geom_abline(slope = 2, intercept = 0, color = "green", linetype = "solid") +
    coord_fixed()

# %% Retraining the model
set.seed(213)
new_split <- initial_split(
    other_station_data,
    prop = 3 / 4,
    strata = grimm_pm10
)
new_train <- training(new_split)
new_test <- testing(new_split)

combined_train <- bind_rows(train_data, new_train)
xgb_new_fit <- xgb_workflow_ |>
    fit(data = combined_train)

new_predictions <- predict(xgb_new_fit, new_data = new_test)$.pred
new_results <- tibble(new_test, new_predictions)
new_metrics <-
    new_results |>
    metrics(
        truth = grimm_pm10,
        estimate = new_predictions
    )

print(new_metrics)
