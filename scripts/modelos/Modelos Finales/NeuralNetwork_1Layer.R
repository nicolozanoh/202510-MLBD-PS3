## ---------------------------
##
## Script name: GradientBoosting.R
##
## Purpose of script:
##
## Author: Nicolas Lozano Huertas
##
## Date Created: 2025-04-08
##
## Email: n.lozanoh@uniandes.edu.co
##
## ---------------------------
##
## Notes:
##
## ---------------------------

rm(list = ls())

require(pacman)

p_load(
  caret,
  dplyr,
  gbm,
  sf,
  tidymodels,
  recipes,
  spatialsample,
  parsnip,
  rsample,
  workflows,
  tune,
  tidyverse,
  dials,
  yardstick
)


## ---- Cargue de datos ----

train <- st_read(dsn = "stores\\work\\Train", layer = "Train")
pred <- st_read(dsn = "stores\\work\\Test", layer = "Test")

## ---- Dividimos los datos de entrenamiento ----

test <- train %>% subset( loc_nmb == 'CHAPINERO')
train <- train %>% subset( loc_nmb != 'CHAPINERO')


## ---- Receta ----

receta_1 <- recipe(price ~ srf_ttl+surf_cv+rooms+bdrms+bathrm+prp_typ+PC1+PC2+
                     PC3+PC4+PC5+PC6+PC7+PC8+PC9+PC10+PC11+PC12+PC13+PC14+PC15+
                     PC16+PC17+PC18+PC19+PC20+PC21+PC22+PC23+PC24+PC25+PC26+
                     PC27+PC28+PC29+PC30+nm_prqd+estrato+dist_nv+dst_hsp+
                     dst_rst+dist_pl+dist_sc+dst_clt+dst_dsc+dst_prq+dst_ttr+dist_br+
                     dst_plt+dst_stc+dst_gym+dst_jrd+dst_prk+dist_jg+dst_vpr+
                     dst_vpd+dst_cyc+dst_mll+rati_cv+bdrm_pr, data = train) %>%
  step_novel(all_nominal_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_interact(terms = ~ srf_ttl:matches("estrato")+dst_vpr:matches("estrato")) %>% 
  step_poly(dst_vpr, degree = 2) %>%
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors())

## ---- Modelo ----

nnet <- parsnip::mlp(
  hidden_units = 6,
  epochs = 100,
  engine = 'nnet',
  validation_split = 0.2
) %>% 
  parsnip::set_mode("regression") %>% 
  parsnip::set_engine("nnet")

## ---- Workflow ----

work <- workflow() %>% 
  add_recipe(receta_1) %>%
  add_model(nnet)

## ---- Estimamos el modelo ----

## ---- Validacion Cruzada espacial ----
grid_values <- tidyr::crossing(
  # `crossing` nos permite generar una grilla 
  # rectangular con la combinación de todos los hiperparámetros. 
  hidden_units = c(1, 3, 5, 10, 20, 40, 60, 80, 100),
  epochs =  c(100, 300, 500, 800, 1000)
)

# train_sf <- st_as_sf(
#   train,
#   # "coords" is in x/y order -- so longitude goes first!
#   coords = c("lon", "lat"),
#   # Set our coordinate reference system to EPSG:4326,
#   # the standard WGS84 geodetic coordinate reference system
#   crs = 3116
# )

set.seed(123)

block_folds <-
  spatial_leave_location_out_cv(
    train,
    group = loc_nmb
  )

nnet_tune <- 
  parsnip::mlp(hidden_units =tune(), epochs = tune()) %>% 
  parsnip::set_mode("regression") %>% 
  parsnip::set_engine("nnet", trace = 0)

workflow_tune <- workflow() %>% 
  add_recipe(receta_1) %>%
  add_model(nnet_tune) 

# set.seed(86936)
# block_folds <- spatial_block_cv(train_sf, v = 5)


tune_res1 <- tune_grid(
  workflow_tune,         # El flujo de trabajo que contiene: receta y especificación del modelo
  resamples = block_folds,  # Folds de validación cruzada espacial
  grid = grid_values,        # Grilla de valores de penalización
  metrics = metric_set(yardstick::mae)  # metrica
)

best_tune <- select_best(tune_res1, metric = "mae")
best_tune

nnet_tuned_final <- finalize_workflow(workflow_tune, best_tune)

nnet_tuned_final_fit <- fit(nnet_tuned_final, data = train)

augment(nnet_tuned_final_fit, new_data = test) %>%
  yardstick::mae(truth = price, estimate = .pred)


preds <- predict(nnet_tuned_final_fit, new_data = pred)
results <- bind_cols(pred$prprty_, preds)

colnames(results) <- c("property_id","price")

write.csv(results, "stores\\sub\\nn_1_layer.csv", row.names = FALSE)
