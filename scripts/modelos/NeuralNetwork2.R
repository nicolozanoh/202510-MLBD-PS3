library(sf)
library(spatialsample)
library(recipes)
library(keras)
library(purrr)
library(dplyr)

# 1. Read and prepare data
train_sf <- st_read(dsn = "stores/work/Train", layer = "Train")

# 2. Create leave-location-out splits
cv_loc_splits <- spatial_leave_location_out_cv(
  data     = train_sf,
  group = loc_nmb
)

# 3. Hyperparameter grid
hyper_grid <- expand.grid(
  units1        = c(32, 64),
  units2        = c(16, 32),
  dropout_rate  = c(0.2, 0.4),
  lr            = c(1e-3, 1e-4)
)

# 4. Model builder factory
build_model <- function(input_shape, units1, units2, dropout_rate, lr) {
  model <- keras_model_sequential() %>%
    layer_dense(units = units1, activation = "relu",
                input_shape = input_shape) %>%
    layer_dropout(rate = dropout_rate) %>%
    layer_dense(units = units2, activation = "relu") %>%
    layer_dense(units = 1)
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = lr),
    loss      = "mean_absolute_error",
    metrics   = list("mean_absolute_error")
  )
  model
}

# 5. Evaluation function for one hyperparameter combo
evaluate_combo <- function(units1, units2, dropout_rate, lr) {
  # run through each fold, collect MAE
  maes <- map_dbl(cv_loc_splits$splits, function(split) {
    tr  <- analysis(split)  %>% st_set_geometry(NULL)
    te  <- assessment(split) %>% st_set_geometry(NULL)
    rec <- recipe(price ~ ., data = tr) %>%
      step_normalize(all_numeric_predictors()) %>%
      prep()
    x_tr <- bake(rec, new_data = tr) %>% select(-price) %>% as.matrix()
    y_tr <- bake(rec, new_data = tr) %>% pull(price)
    x_te <- bake(rec, new_data = te) %>% select(-price) %>% as.matrix()
    y_te <- bake(rec, new_data = te) %>% pull(price)
    
    m <- build_model(ncol(x_tr), units1, units2, dropout_rate, lr)
    m %>% fit(x_tr, y_tr, epochs = 30, batch_size = 32,
              validation_split = 0.2, verbose = 0)
    m %>% evaluate(x_te, y_te, verbose = 0) %>% `[[`("mean_absolute_error")
  })
  
  mean(maes)
}

# 6. Grid‐search: attach mean MAE to each row
results <- hyper_grid %>%
  mutate(mean_mae = pmap_dbl(
    list(units1, units2, dropout_rate, lr),
    evaluate_combo
  ))

# 7. Pick the best
best <- results %>% slice_min(mean_mae, n = 1)
print(best)

units1       <- best$units1
units2       <- best$units2
dropout_rate <- best$dropout_rate
lr           <- best$lr

# 2. Prepare full training data
train_sf <- st_read(dsn = "stores/work/Train", layer = "Train") %>%
  mutate(loc_id = as.factor(loc_nmb))
train_df <- train_sf %>% st_set_geometry(NULL)

# 3. Fit preprocessing recipe on full data
rec_full <- recipe(price ~ ., data = train_df) %>%
  step_normalize(all_numeric_predictors()) %>%
  prep()

x_full <- bake(rec_full, new_data = train_df) %>% select(-price) %>% as.matrix()
y_full <- bake(rec_full, new_data = train_df) %>% pull(price)

# 4. Build the final model with best hyperparameters
build_model <- function(input_shape, units1, units2, dropout_rate, lr) {
  model <- keras_model_sequential() %>%
    layer_dense(units = units1, activation = "relu",
                input_shape = input_shape) %>%
    layer_dropout(rate = dropout_rate) %>%
    layer_dense(units = units2, activation = "relu") %>%
    layer_dense(units = 1)
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = lr),
    loss      = "mean_absolute_error",
    metrics   = list("mean_absolute_error")
  )
  model
}

final_model <- build_model(
  input_shape  = ncol(x_full),
  units1       = units1,
  units2       = units2,
  dropout_rate = dropout_rate,
  lr           = lr
)

# 5. Train on the full dataset
final_model %>% fit(
  x_full, y_full,
  epochs     = 50,
  batch_size = 32,
  verbose    = 1
)

# 6. Load your new data (without `price`) and preprocess
new_sf <- st_read(dsn = "stores/work/Test", layer = "Test") %>%
  mutate(loc_id = as.factor(loc_nmb))
new_df <- new_sf %>% st_set_geometry(NULL)

x_new <- bake(rec_full, new_data = new_df) %>% as.matrix()

# 7. Make predictions
predictions <- final_model %>% predict(x_new)
new_df$predicted_price <- as.numeric(predictions)

# 8. (Optional) Combine back with spatial data
new_sf$predicted_price <- new_df$predicted_price

# View the first few predictions
head(new_sf)
