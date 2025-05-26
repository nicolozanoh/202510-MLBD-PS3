

## ───────────────────────── 0. PREPARACIÓN ─────────────────────────────
pacman::p_load(
  tidymodels, sf, blockCV, xgboost, nnet, brulee, keras,
  SuperLearner, future, doFuture, janitor, readr, stringr
)

# ── Rutas
base_dir   <- "C:/work"
path_train <- file.path(base_dir, "Train", "Train.shp")
path_test  <- file.path(base_dir, "Test",  "Test.shp")
dir_out    <- file.path(base_dir, "predicciones_jcp")
stopifnot(file.exists(path_train), file.exists(path_test))

# ── Datos
train_sf <- st_read(path_train, quiet = TRUE)
test_sf  <- st_read(path_test , quiet = TRUE)
train <- st_drop_geometry(train_sf)
test  <- st_drop_geometry(test_sf)

set.seed(777)




## ── BACKEND future
workers <- 2L
rscript_ok <- gsub("\\\\", "/", file.path(R.home("bin"), "Rscript.exe"))
plan(multisession, workers = workers, rscript = rscript_ok, outfile = "")
registerDoFuture()

## ─────────────────── 1. OPCIÓN A – subir límite de globals ────────────
options(future.globals.maxSize = 4 * 1024^3)   ### ← Lo fijo en este valor para no saturar memoria VRAM




## ─────────────────── 2. VALIDACIÓN CRUZADA ESPACIAL ───────────────────
folds_spatial <- blockCV::spatialBlock(
  speciesData = train_sf,
  theRange    = 2000,
  k           = 5,
  selection   = "random",
  iteration   = 250,
  biomod2Format = FALSE,
  showBlocks  = FALSE)

train$foldID <- folds_spatial$foldID

folds <- group_vfold_cv(train, group = foldID,
                        v = length(unique(train$foldID)))



## ───────────────────── 3. RECETA ─────────────────────────────

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
  # step_log(price, offset = 1) %>%   # linea para incluir log-transform
  step_corr(all_numeric_predictors(), threshold = .9)




## ── Helper CSV
write_kaggle <- function(df, name){
  df %>% select(property_id, price = .pred) %>%
    write_csv(file.path(dir_out, name), na = "")
}

ctrl <- control_grid(
  parallel_over = "resamples",
  verbose       = TRUE
)


## ───────────────────── 4. XGBOOST (GPU) ───────────────────────────────
xgb_spec <- boost_tree(
  trees = tune(), tree_depth = tune(), learn_rate = tune(),
  loss_reduction = tune(), mtry = tune(), sample_size = tune(),
  stop_iter = 50) %>%
  set_engine("xgboost",
             tree_method = "gpu_hist",
             predictor   = "gpu_predictor",
             nrounds     = 1000,
             eval_metric = "mae") %>%
  set_mode("regression")

xgb_wf   <- workflow() %>% add_recipe(receta_1) %>% add_model(xgb_spec)

xgb_grid <- grid_space_filling(
  trees(), tree_depth(), learn_rate(), loss_reduction(),
  finalize(mtry(), train), sample_size = sample_prop(),
  size = 5)

xgb_res <- tune_grid(
  xgb_wf,
  resamples = folds,
  grid      = xgb_grid,
  metrics   = metric_set(mae),
  control   = ctrl)                   

best_xgb <- select_best(xgb_res, "mae")
xgb_fit  <- finalize_workflow(xgb_wf, best_xgb) %>% fit(train)
write_kaggle(bind_cols(test, predict(xgb_fit, test)), "pred_xgb.csv")



## ───────────────────── 5. NNET (CPU, ligero) ──────────────────────────
mlp_nnet_spec <- mlp(
  hidden_units = tune(), penalty = tune(), epochs = tune()
) %>%
  set_engine("nnet", linout = TRUE, trace = FALSE, MaxNWts = 5e4) %>%
  set_mode("regression")

mlp_nnet_wf <- workflow() %>% add_recipe(receta_1) %>% add_model(mlp_nnet_spec)

grid_nnet <- grid_regular(
  hidden_units(range(10L, 40L)),
  penalty(range(-4, -1)),
  epochs(range(100, 200)),
  levels = 3
)

res_nnet <- tune_grid(
  mlp_nnet_wf,
  resamples = folds,        # ← objeto vuelve a existir
  grid      = grid_nnet,
  metrics   = metric_set(mae),
  control   = ctrl
)

best_nnet <- select_best(res_nnet, "mae")
nnet_fit  <- finalize_workflow(mlp_nnet_wf, best_nnet) %>% fit(train)
write_kaggle(bind_cols(test, predict(nnet_fit, test)), "pred_mlp_nnet.csv")


## ───────────────────── 6. BRULEE / torch (GPU) ───────────────────────

mlp_br_spec <- mlp(
  hidden_units = tune(),
  penalty      = tune(),
  epochs       = tune(),
  learn_rate   = tune()
) %>%
  set_engine("brulee",
             device = "cuda",          # GPU
             batch_size = 1024) %>%    # 'batch_size' sí lo acepta
  set_mode("regression")

## (grid_br, res_br, best_br iguales)

br_fit <- finalize_workflow(mlp_br_wf, best_br) %>% fit(train)

## ─────────────── 2.  Función CSV usa nombre real de ID ───────────────
names(test) %>% head()

write_kaggle <- function(df, file_name, id_col){
  df %>%
    select({{ id_col }}, price = .pred) %>%
    write_csv(file.path(dir_out, file_name), na = "")
}

# Llamada
write_kaggle(bind_cols(test, predict(br_fit, test)),
             "pred_mlp_brulee.csv",
             id_col = id_propiedad)




best_br   <- select_best(res_br,   metric = "mae")   # brulee
br_fit  <- finalize_workflow(mlp_br_wf, best_br) %>% fit(train)
write_kaggle(bind_cols(test, predict(br_fit, test)), "pred_mlp_brulee.csv")



## ───────────────────── 7. KERAS / TensorFlow–GPU ──────────────────────
library(dials)

keras_build <- function(input_shape,
                        units1 = 256, units2 = 128,
                        dropout_rate = 0.30, lr = 1e-3) {
  keras_model_sequential() %>%
    layer_dense(units1, activation = "relu", input_shape = input_shape) %>%
    layer_dropout(rate = dropout_rate) %>%
    layer_dense(units2, activation = "relu") %>%
    layer_dropout(rate = dropout_rate) %>%
    layer_dense(1) %>%
    compile(loss = "mae",
            optimizer = optimizer_adam(learning_rate = lr),
            metrics = "mae")
}

keras_spec <- mlp() %>%
  set_engine("keras",
             build_fn     = keras_build,
             units1       = tune(),   # IDs: units1, units2, lr…
             units2       = tune(),
             dropout_rate = tune(),
             lr           = tune(),
             epochs       = tune(),
             batch_size   = 1024) %>%
  set_mode("regression")

keras_wf <- workflow() %>% add_recipe(receta_1) %>% add_model(keras_spec)

## 1. Obtener y actualizar el conjunto de parámetros --------------------
param_keras <- keras_wf %>% parameters()

param_keras <- update(
  param_keras,
  units1       = hidden_units(range = c(128L, 512L)),
  units2       = hidden_units(range = c(64L, 256L)),
  dropout_rate = dropout(range = c(0.10, 0.40)),
  lr           = learn_rate(range = c(-4, -2)),
  epochs       = epochs(range = c(50L, 120L))
)

## 2. Grid space-filling -----------------------------------------------
grid_keras <- grid_space_filling(param_keras, size = 5)

## 3. Ajuste -------------------------------------------------------------
res_keras <- tune_grid(
  keras_wf,
  resamples = folds,
  grid      = grid_keras,
  metrics   = metric_set(mae),
  control   = ctrl
)

best_keras <- select_best(res_keras, "mae")
keras_fit  <- finalize_workflow(keras_wf, best_keras) %>% fit(train)

## 4. Predicciones CSV ---------------------------------------------------
write_kaggle(
  bind_cols(test, predict(keras_fit, test)),
  "pred_keras_dnn.csv"
)





## ───────────────────── 8. SUPERLEARNER (CPU) ─────────────────────────
rec_prep <- prep(receta_1, training = train, verbose = FALSE)
x_tr <- bake(rec_prep, train, all_predictors()) %>% as.matrix()
x_te <- bake(rec_prep, test , all_predictors()) %>% as.matrix()
y_tr <- bake(rec_prep, train, all_outcomes())$price

fold_list <- split(seq_len(nrow(train)), folds$in_id)

sl_learners <- c("SL.xgboost", "SL.glmnet", "SL.nnet", "SL.randomForest")
cl <- future::getCluster()         # ← cluster para paralelizar

sl_fit <- SuperLearner(
  Y = y_tr, X = x_tr, family = gaussian(),
  SL.library = sl_learners,
  method     = "method.NNLS",
  cvControl  = list(V = length(fold_list), validRows = fold_list),
  parallel   = "snow"",
  cluster    = cl,
  verbose    = TRUE
)

sl_pred <- predict(sl_fit, newdata = x_te)$pred
write_kaggle(tibble(property_id = test$property_id, .pred = sl_pred),
             "pred_superlearner.csv")


## ───────────────────── 9. COMPARAR MAE (validación) ──────────────────
grab_mae <- function(x) collect_metrics(x) %>%
  filter(.metric == "mae") %>% pull(mean)

mae_tbl <- tibble(
  model = c("XGB-GPU", "nnet", "brulee-GPU", "keras-GPU"),
  mae   = c(grab_mae(xgb_res),  grab_mae(res_nnet),
            grab_mae(res_br),   grab_mae(res_keras))
) %>% arrange(mae)

print(mae_tbl)
cat("SuperLearner MAE (CV interno):", round(sl_fit$cvRisk, 2), "\n")

## ───────────────────── 10. FIN ───────────────────────────────────────
