## ─────────────────────────────────────────────────────────────────────────────
## Script name:    LinearRegression.R
##
## Purpose:        Estimamos modelo regresión para predicciones PS3.       
##                 
##
## Author:         Nicolas Lozano Huertas
## Email:          n.lozanoh2@gmail.com
## Affiliation:    Universidad de los Andes
##
## Created on:     2025-05-24
## Last updated:   2025-05-24
##
## ─────────────────────────────────────────────────────────────────────────────

## ---- Limpiamos entorno ----

rm(list = ls())

## ---- Cargamos paquetes ----

require(pacman)

p_load(
  dplyr,
  tidymodels,
  sf,
  spatialsample
)

## ---- Cargamos datos ----

train <- st_read(dsn = "stores\\work\\Train", layer = "Train")
pred <- st_read(dsn = "stores\\work\\Test", layer = "Test")

## ---- Dividimos datos ----

test <- train %>% subset( loc_nmb == 'CHAPINERO')
train <- train %>% subset( loc_nmb != 'CHAPINERO')

## ---- Definimos recipe ----

receta1 <- recipe(price ~ srf_ttl+surf_cv+rooms+bdrms+bathrm+prp_typ+PC1+PC2+
                     PC3+PC4+PC5+PC6+PC7+PC8+PC9+PC10+PC11+PC12+PC13+PC14+PC15+
                     PC16+PC17+PC18+PC19+PC20+PC21+PC22+PC23+PC24+PC25+PC26+
                     PC27+PC28+PC29+PC30+nm_prqd+estrato+dist_nv+dst_hsp+
                     dst_rst+dist_pl+dist_sc+dst_clt+dst_dsc+dst_prq+dst_ttr+dist_br+
                     dst_plt+dst_stc+dst_gym+dst_jrd+dst_prk+dist_jg+dst_vpr+
                     dst_vpd+dst_cyc+dst_mll+rati_cv+bdrm_pr, data = train) %>%
  #step_log(price, base = exp(1), skip = TRUE) %>% 
  step_novel(all_nominal_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_interact(terms = ~ srf_ttl:matches("estrato")+dst_vpr:matches("estrato")) %>% 
  step_poly(dst_vpr, degree = 2) %>%
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors())


## ---- Definimos modelo ----
lm <- linear_reg() %>% 
  set_engine("lm")

## ---- Definimos workflow ----

wf <- workflow() %>% 
  add_recipe(receta1) %>%
  add_model(lm)

## ---- Estimamos el modelo ----

fit <- wf %>% fit(data=train)

## ---- Cross Validation ----

set.seed(123)

location_folds_train <-
  spatial_leave_location_out_cv(
    train,
    group = loc_nmb
  )

autoplot(location_folds_train)


lm_cv <- wf %>%
  fit_resamples(resamples = location_folds_train,
                metrics   = metric_set(yardstick::mae),
                control   = control_resamples(save_pred = TRUE))

augment(fit, new_data = test) %>%
  yardstick::mae(truth = price, estimate = .pred)

preds <- predict(fit, new_data = pred)
results <- bind_cols(pred$prprty_, preds)

colnames(results) <- c("property_id","price")

write.csv(results, "stores\\sub\\LinearRegression.csv", row.names = FALSE)
