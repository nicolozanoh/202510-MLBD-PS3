
## Preparacion

# 0. Librerías -------------------------------------------------------------
pacman::p_load(
  tidymodels, sf, blockCV,       # flujo general y CV espacial
  xgboost, nnet, brulee, keras,  # motores de modelos
  SuperLearner, doParallel,      # ensemble y paralelización
  janitor, readr, stringr
)

# 1. Rutas --------------------------------------------

# 1. Construir rutas de forma segura ---------------------------
base_dir <- "C://work"    # toda con /
path_train <- file.path(base_dir, "Train", "Train.shp")
path_test  <- file.path(base_dir, "Test", "Test.shp")
dir_out    <- file.path(base_dir, "predicciones_jcp")

print(path_train)

# Comprobar que existen
stopifnot(file.exists(path_train), file.exists(path_test))

# 2. Leer shapefiles -------------------------------------------
library(sf)
train_sf <- st_read(path_train, quiet = TRUE)
test_sf  <- st_read(path_test, quiet  = TRUE)

# 3. Drop geometry para modelar -------------------------------------------
train <- train_sf |> st_drop_geometry()
test  <- test_sf  |> st_drop_geometry()

# Semilla reproducible
set.seed(777)




# ------------------------------------------------------------------
# ⁂  PARALelización future + doFuture ••• SIN '\U' •••
# ------------------------------------------------------------------
library(future); library(doFuture)

workers <- max(1L, future::availableCores() - 1L)

## Ruta a Rscript con barras normales
rscript_ok <- gsub("\\\\", "/", file.path(R.home("bin"), "Rscript.exe"))

## ‼  Lanza multisession usando ESA ruta
plan(multisession,
     workers = workers,
     rscript = rscript_ok,     # ← la clave está aquí
     outfile  = ""             # mensajes de cada worker en consola
)

registerDoFuture()             # foreach / tidymodels ya ven el backend
cat("Backend future-multisession activo con", workers, "workers\n")





















# ──────────────────────────────────────────────────────────────────────
#  VALIDACIÓN CRUZADA ESPACIAL (misma lógica, sin cambios de nombre)
# ──────────────────────────────────────────────────────────────────────
folds_spatial <- blockCV::spatialBlock(
  speciesData    = train_sf,
  theRange       = 2000,  # 2 km
  k              = 5,
  selection      = "random",
  iteration      = 250,
  biomod2Format  = FALSE,
  showBlocks     = FALSE
)

train$foldID <- folds_spatial$foldID  # agrega columna

folds <- group_vfold_cv(
  data  = train,
  group = foldID,
  v     = length(unique(train$foldID))
)

# ──────────────────────────────────────────────────────────────────────
#  RECETA (idéntica, solo nombres en inglés para evitar errores)
# ──────────────────────────────────────────────────────────────────────
library(recipes)

receta_1 <- recipe(
  price ~ srf_ttl + surf_cv + rooms + bdrms + bathrm + prp_typ +
    PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 +
    PC11 + PC12 + PC13 + PC14 + PC15 + PC16 + PC17 + PC18 + PC19 +
    PC20 + PC21 + PC22 + PC23 + PC24 + PC25 + PC26 + PC27 + PC28 +
    PC29 + PC30 + nm_prqd + estrato + dist_nv + dst_hsp + dst_rst +
    dist_pl + dist_sc + dst_clt + dst_dsc + dst_prq + dst_ttr +
    dst_plt + dst_stc + dst_gym + dst_jrd + dst_prk + dist_jg +
    dst_vpr + dst_vpd + dst_cyc + dst_mll + rati_cv + bdrm_pr,
  data = train
) %>%
  step_novel(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_interact(terms = ~ srf_ttl:estrato + dst_vpr:estrato) %>%
  step_poly(dst_vpr, degree = 2) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_predictors()) %>%
  # step_log(price, offset = 1) %>%   # descomenta si quieres log-transform
  step_corr(all_numeric_predictors(), threshold = .9)

# ──────────────────────────────────────────────────────────────────────
#  HELPER PARA CSV KAGGLE
# ──────────────────────────────────────────────────────────────────────
write_kaggle <- function(df_pred, file_name){
  df_pred %>%
    select(property_id, price = .pred) %>%
    readr::write_csv(file.path(dir_out, file_name), na = "")
}

# ──────────────────────────────────────────────────────────────────────
#  1. XGBOOST
# ──────────────────────────────────────────────────────────────────────
xgb_spec <- boost_tree(
  trees          = tune(), tree_depth   = tune(), learn_rate = tune(),
  loss_reduction = tune(), mtry         = tune(), sample_size = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("regression")

xgb_wf   <- workflow() %>% add_recipe(receta_1) %>% add_model(xgb_spec)

xgb_grid <- grid_latin_hypercube(
  trees(), tree_depth(), learn_rate(), loss_reduction(),
  finalize(mtry(), train), sample_size = sample_prop(),
  size = 30
)

xgb_res <- tune_grid(
  xgb_wf,
  resamples = folds,
  grid      = xgb_grid,
  metrics   = metric_set(mae),
  control   = control_grid(parallel_over = "everything")   # <- usa workers
)

best_xgb <- select_best(xgb_res, "mae")
xgb_fit  <- finalize_workflow(xgb_wf, best_xgb) %>% fit(train)

pred_xgb <- predict(xgb_fit, test) %>% bind_cols(test)
write_kaggle(pred_xgb, "pred_xgb.csv")

# ──────────────────────────────────────────────────────────────────────
#  2. MLP – nnet
# ──────────────────────────────────────────────────────────────────────
mlp_nnet_spec <- mlp(
  hidden_units = tune(), penalty = tune(), epochs = tune()
) %>%
  set_engine("nnet", linout = TRUE, trace = FALSE, MaxNWts = 5e4) %>%
  set_mode("regression")

mlp_nnet_wf <- workflow() %>% add_recipe(receta_1) %>% add_model(mlp_nnet_spec)

grid_nnet <- grid_regular(
  hidden_units(range(5L, 50L)),
  penalty(range(-4, -0.5)),
  epochs(range(50, 300)),
  levels = 5
)

res_nnet <- tune_grid(
  mlp_nnet_wf,
  resamples = folds,
  grid      = grid_nnet,
  metrics   = metric_set(mae),
  control   = control_grid(parallel_over = "everything")
)

best_nnet <- select_best(res_nnet, "mae")
nnet_fit  <- finalize_workflow(mlp_nnet_wf, best_nnet) %>% fit(train)

pred_nnet <- predict(nnet_fit, test) %>% bind_cols(test)
write_kaggle(pred_nnet, "pred_mlp_nnet.csv")

# ──────────────────────────────────────────────────────────────────────
#  3. MLP – brulee
# ──────────────────────────────────────────────────────────────────────
mlp_br_spec <- mlp(
  hidden_units = tune(), penalty = tune(), epochs = tune()
) %>%
  set_engine("brulee",
             activation = "relu", batch_size = 512,
             learn_rate = tune(), dropout = 0.2) %>%
  set_mode("regression")

mlp_br_wf <- workflow() %>% add_recipe(receta_1) %>% add_model(mlp_br_spec)

grid_br <- grid_latin_hypercube(
  hidden_units(range(32L, 256L)),
  penalty(range(-5, -1)),
  learn_rate(range(-4, -1)),
  epochs(range(50, 200)),
  size = 20
)

res_br <- tune_grid(
  mlp_br_wf,
  resamples = folds,
  grid      = grid_br,
  metrics   = metric_set(mae),
  control   = control_grid(parallel_over = "everything")
)

best_br <- select_best(res_br, "mae")
br_fit  <- finalize_workflow(mlp_br_wf, best_br) %>% fit(train)

pred_br <- predict(br_fit, test) %>% bind_cols(test)
write_kaggle(pred_br, "pred_mlp_brulee.csv")

# ──────────────────────────────────────────────────────────────────────
#  4. Keras DNN
# ──────────────────────────────────────────────────────────────────────
keras_build <- function(input_shape,
                        units1 = 256, units2 = 128,
                        dropout_rate = 0.3, lr = 1e-3){
  keras_model_sequential() %>%
    layer_dense(units1, activation = "relu", input_shape = input_shape) %>%
    layer_dropout(rate = dropout_rate) %>%
    layer_dense(units2, activation = "relu") %>%
    layer_dropout(rate = dropout_rate) %>%
    layer_dense(1) %>%
    compile(
      loss      = "mae",
      optimizer = optimizer_adam(learning_rate = lr),
      metrics   = "mae"
    )
}

keras_spec <- mlp() %>%
  set_engine("keras",
             build_fn = keras_build,
             units1 = tune(), units2 = tune(),
             dropout_rate = tune(), lr = tune(),
             epochs = tune(), batch_size = 512) %>%
  set_mode("regression")

keras_wf <- workflow() %>% add_recipe(receta_1) %>% add_model(keras_spec)

grid_keras <- grid_latin_hypercube(
  units1(range(128L, 512L)),
  units2(range(64L, 256L)),
  dropout_rate(range(0.1, 0.5)),
  lr(range(-4, -2)),
  epochs(range(50, 150)),
  size = 15
)

res_keras <- tune_grid(
  keras_wf,
  resamples = folds,
  grid      = grid_keras,
  metrics   = metric_set(mae),
  control   = control_grid(parallel_over = "everything")
)

best_keras <- select_best(res_keras, "mae")
keras_fit  <- finalize_workflow(keras_wf, best_keras) %>% fit(train)

pred_keras <- predict(keras_fit, test) %>% bind_cols(test)
write_kaggle(pred_keras, "pred_keras_dnn.csv")

# ──────────────────────────────────────────────────────────────────────
#  5. SUPERLEARNER  (reutiliza los mismos workers)
# ──────────────────────────────────────────────────────────────────────
rec_prep <- prep(receta_1, training = train, verbose = FALSE)

x_tr <- bake(rec_prep, train, all_predictors()) %>% as.matrix()
x_te <- bake(rec_prep, test , all_predictors()) %>% as.matrix()
y_tr <- bake(rec_prep, train, all_outcomes())$price

fold_list <- split(seq_len(nrow(train)), folds$in_id)

sl_learners <- c("SL.xgboost", "SL.glmnet", "SL.nnet", "SL.randomForest")

cl <- future::getCluster()  # mismo PSOCK creado por plan()

sl_fit <- SuperLearner(
  Y           = y_tr,
  X           = x_tr,
  family      = gaussian(),
  SL.library  = sl_learners,
  method      = "method.NNLS",
  cvControl   = list(V = length(fold_list), validRows = fold_list),
  parallel    = "snow",
  cluster     = cl,        # usa los mismos workers
  verbose     = TRUE
)

sl_pred <- predict(sl_fit, newdata = x_te)$pred
pred_sl <- tibble(property_id = test$property_id, .pred = sl_pred)
write_kaggle(pred_sl, "pred_superlearner.csv")

# ──────────────────────────────────────────────────────────────────────
#  6. COMPARAR MAE DE VALIDACIÓN
# ──────────────────────────────────────────────────────────────────────
collect_mae <- function(res) {
  res %>%
    collect_metrics(summarize = TRUE) %>%
    filter(.metric == "mae") %>%
    select(mean) %>%
    pull()
}

mae_tbl <- tibble(
  model = c("XGBoost", "MLP-nnet", "MLP-brulee", "Keras-DNN"),
  mae   = c(collect_mae(xgb_res),
            collect_mae(res_nnet),
            collect_mae(res_br),
            collect_mae(res_keras))
) %>%
  arrange(mae)

print(mae_tbl)

# El MAE del SuperLearner lo tomamos del validation set interno
mae_sl <- sl_fit$cvRisk   # promedio CV interno
cat("SuperLearner MAE (CV interno):", round(mae_sl, 2), "\n")

# ──────────────────────────────────────────────────────────────────────
#  RESULTADOS
# ──────────────────────────────────────────────────────────────────────
best_single <- mae_tbl$model[1]
cat("\n► Modelo con menor MAE entre los individuales:", best_single,
    "(≈", round(mae_tbl$mae[1], 2), ")\n")

if (mae_sl < mae_tbl$mae[1]) {
  cat("► SuperLearner mejora todavía más (MAE ≈", round(mae_sl, 2), ").\n")
} else {
  cat("► SuperLearner NO supera al mejor individual.\n")
}

# Si deseas terminar la sesión paralela:
# plan(sequential)

