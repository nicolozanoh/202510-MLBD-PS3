require(pacman)
p_load(
  ggplot2,
  tidyverse,
  dplyr,
  visdat,
  sf,
  stargazer,
  leaflet,
  gridExtra,
  osmdata,
  FNN
)

# train <- read_sf("//stores//work_jcp//Train//Train.shp")
# test <- read_sf("//stores//work_jcp//Test//Test.shp")


## Preparacion

# 0. Librerías -------------------------------------------------------------
pacman::p_load(
  tidymodels, sf, blockCV,       # flujo general y CV espacial
  xgboost, nnet, brulee, keras,  # motores de modelos
  SuperLearner, doParallel,      # ensemble y paralelización
  janitor, readr, stringr
)

# 1. Rutas --------------------------------------------
path_train <- "//stores//work//Train//Train.shp"
path_test  <- "//stores//work//Test//Test.shp"
dir_out    <- "//stores//work_jcp//predicciones"

list.files("//stores/work_jcp/Train", pattern = "\\.shp$")
# Deberías ver "Train.shp" en el resultado

head(path_train)

head(path_test)



# 2. Lectura de datos ------------------------------------------------------
train_sf <- st_read(path_train, quiet = TRUE)
test_sf  <- st_read(path_test, quiet = TRUE)

# 3. Drop geometry para modelar -------------------------------------------
train <- train_sf |> st_drop_geometry()
test  <- test_sf  |> st_drop_geometry()

# 4. Semilla y cluster -----------------------------------------------------
set.seed(777)
cl <- makeCluster(parallel::detectCores() - 1)
registerDoParallel(cl)













train <- "//stores//work_jcp//Train//Train.shp"
test <- "//stores//work_jcp//Test//Test.shp"


## Validacion cruzada espacial

# 5. Dividir en folds ------------------------------------------------------

# genera grillas de 2 km; fold = bloque
folds_spatial <- spatialBlock(
  speciesData = train_sf,
  theRange    = 2000,         # 2 km
  k           = 5,            # 5 folds
  selection   = "random",
  iteration   = 250,
  biomod2Format = FALSE,
  showBlocks  = FALSE
)

# Convertimos a vfold_cv con índices
folds <- vfold_cv(
  data  = train,
  v     = length(folds_spatial$foldID),
  repeats = 1,
  strata  = NULL,
  id      = as.factor(folds_spatial$foldID)
)

## Receta base
receta_1 <- recipe(price ~ srf_ttl+surf_cv+rooms+bdrms+bathrm+prp_typ+PC1+PC2+
                     PC3+PC4+PC5+PC6+PC7+PC8+PC9+PC10+PC11+PC12+PC13+PC14+PC15+
                     PC16+PC17+PC18+PC19+PC20+PC21+PC22+PC23+PC24+PC25+PC26+
                     PC27+PC28+PC29+PC30+nm_prqd+estrato+dist_nv+dst_hsp+
                     dst_rst+dist_pl+dist_sc+dst_clt+dst_dsc+dst_prq+dst_ttr+dist_br+
                     dst_plt+dst_stc+dst_gym+dst_jrd+dst_prk+dist_jg+dst_vpr+
                     dst_vpd+dst_cyc+dst_mll+rati_cv+bdrm_pr, data = Train) %>%
  step_novel(all_nominal_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_interact(terms = ~ srf_ttl:matches("estrato")+dst_vpr:matches("estrato")) %>% 
  step_poly(dst_vpr, degree = 2) %>%
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors())
  #step_log(price, offset = 1) |> # (opcional) estabilizar varianza
  step_corr(all_numeric(), threshold = .9)



# CSV Para Kaggle
write_kaggle <- function(df_pred, nombre_archivo){
  df_pred |>
    select(property_id, price = .pred) |>
    write_csv(file.path(dir_out, nombre_archivo),
              na = "", append = FALSE)
}


#### MODELO 1 – XGBoost

# Especificación -----------------------------------------------------------
xgb_spec <- boost_tree(
    trees = tune(), tree_depth = tune(), learn_rate = tune(),
    loss_reduction = tune(), mtry = tune(), sample_size = tune()
  ) |>
  set_engine("xgboost") |>
  set_mode("regression")

xgb_wf <- workflow() |> add_model(xgb_spec) |> add_recipe(receta_1)

# Grid y ajuste -----------------------------------------------------------
xgb_grid <- grid_latin_hypercube(
  trees(), tree_depth(), learn_rate(), loss_reduction(),
  finalize(mtry(), train), sample_size = sample_prop(),
  size = 30
)

xgb_res <- tune_grid(
  xgb_wf, resamples = folds, grid = xgb_grid,
  metrics = metric_set(mae)
)

best_xgb <- select_best(xgb_res, "mae")
xgb_fit  <- finalize_workflow(xgb_wf, best_xgb) |> fit(data = train)

# Predicción y CSV --------------------------------------------------------
pred_xgb <- predict(xgb_fit, new_data = test) |> bind_cols(test)
write_kaggle(pred_xgb, "pred_xgb.csv")





### MODELO 2 – MLP-nnet

mlp_nnet_spec <- mlp(
    hidden_units = tune(), penalty = tune(), epochs = tune()
  ) |>
  set_engine("nnet", linout = TRUE, trace = FALSE, MaxNWts = 5000) |>
  set_mode("regression")

mlp_nnet_wf <- workflow() |> add_model(mlp_nnet_spec) |> add_recipe(receta_1)

grid_nnet <- grid_regular(
  hidden_units(range(5L, 50L)),
  penalty(range(-4, -0.5)),          # log10
  epochs(range(50, 300)),
  levels = 5
)

res_nnet <- tune_grid(
  mlp_nnet_wf, resamples = folds, grid = grid_nnet,
  metrics = metric_set(mae)
)

best_nnet <- select_best(res_nnet, "mae")
nnet_fit  <- finalize_workflow(mlp_nnet_wf, best_nnet) |> fit(train)

pred_nnet <- predict(nnet_fit, test) |> bind_cols(test)
write_kaggle(pred_nnet, "pred_mlp_nnet.csv")







### MODELO 3 – MLP-brulee

mlp_br_spec <- mlp(
    hidden_units = tune(), penalty = tune(), epochs = tune()
  ) |>
  set_engine("brulee", activation = "relu", batch_size = 512,
             learn_rate = tune(), dropout = 0.2) |>
  set_mode("regression")

mlp_br_wf <- workflow() |> add_model(mlp_br_spec) |> add_recipe(receta_1)

grid_br <- grid_latin_hypercube(
  hidden_units(range(32L, 256L)),
  penalty(range(-5, -1)),
  learn_rate(range(-4, -1)),
  epochs(range(50, 200)),
  size = 20
)

res_br <- tune_grid(
  mlp_br_wf, resamples = folds, grid = grid_br,
  metrics = metric_set(mae)
)

best_br <- select_best(res_br, "mae")
br_fit   <- finalize_workflow(mlp_br_wf, best_br) |> fit(train)

pred_br  <- predict(br_fit, test) |> bind_cols(test)
write_kaggle(pred_br, "pred_mlp_brulee.csv")





### MODELO 4 – Keras-DNN

keras_build <- function(input_shape, units1 = 256, units2 = 128,
                        dropout_rate = 0.3, lr = 1e-3) {
  model <- keras_model_sequential() |>
    layer_dense(units1, activation = "relu", input_shape = input_shape) |>
    layer_dropout(rate = dropout_rate) |>
    layer_dense(units2, activation = "relu") |>
    layer_dropout(rate = dropout_rate) |>
    layer_dense(1)

  model |> compile(
    loss = "mae",
    optimizer = optimizer_adam(learning_rate = lr),
    metrics = "mae"
  )
  model
}

keras_spec <- mlp() |>
  set_engine("keras", build_fn = keras_build,
             units1 = tune(), units2 = tune(),
             dropout_rate = tune(), lr = tune(),
             epochs = tune(), batch_size = 512) |>
  set_mode("regression")

keras_wf <- workflow() |> add_model(keras_spec) |> add_recipe(receta_1)

grid_keras <- grid_latin_hypercube(
  units1(range(128L, 512L)),
  units2(range(64L, 256L)),
  dropout_rate(range(0.1, 0.5)),
  lr(range(-4, -2)),
  epochs(range(50, 150)),
  size = 15
)

res_keras <- tune_grid(
  keras_wf, resamples = folds, grid = grid_keras,
  metrics = metric_set(mae)
)

best_keras <- select_best(res_keras, "mae")
keras_fit  <- finalize_workflow(keras_wf, best_keras) |> fit(train)

pred_keras <- predict(keras_fit, test) |> bind_cols(test)
write_kaggle(pred_keras, "pred_keras_dnn.csv")





### MODELO 5 – SuperLearner

# 8.1 Conversión a matriz -----------------------------------------------
rec <- receta_1 |> prep(training = train, verbose = FALSE)
x_tr <- bake(rec, train, all_predictors()) |> as.matrix()
x_te <- bake(rec, test , all_predictors()) |> as.matrix()
y_tr <- bake(rec, train, all_outcomes())$price

# 8.2 Folds espaciales para SL ------------------------------------------
fold_list <- split(seq_len(nrow(train)), folds$in_id)

# 8.3 Conjunto de learners ----------------------------------------------
sl_learners <- c(
  "SL.xgboost", "SL.glmnet", "SL.nnet", "SL.randomForest"
)

# 8.4 Entrenamiento ------------------------------------------------------
set.seed(777)
sl_fit <- SuperLearner(
  Y = y_tr, X = x_tr, family = gaussian(),
  SL.library       = sl_learners,
  method           = "method.NNLS",
  cvControl        = list(V = length(fold_list), validRows = fold_list),
  verbose          = TRUE, parallel = "multicore"
)

# 8.5 Predicción y CSV ---------------------------------------------------
sl_pred <- predict(sl_fit, newdata = x_te)$pred
pred_sl <- tibble(property_id = test$property_id, .pred = sl_pred)
write_kaggle(pred_sl, "pred_superlearner.csv")





### Cálculo del MAE interno

mae_train <- train |> 
  bind_cols(predict(xgb_fit, train)) |> 
  mae(price, .pred)

