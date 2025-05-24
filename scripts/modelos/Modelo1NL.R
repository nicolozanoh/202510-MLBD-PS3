## ---------------------------
##
## Script name: NeuralNetwork.R
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
  Metrics,
  gbm,
  sf,
  tidymodels,
  recipes,
  spatialsample,
  parsnip
)


## ---- Cargue de datos ----

train <- st_read(dsn = "stores\\work\\Train", layer = "Train")

## ---- Dividimos los datos de entrenamiento ----

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

gb_model <- boost_tree(
  trees = 1000,
  tree_depth = 6,
  min_n = 10,
  loss_reduction = 0.01,  # equivalent to 'min_split_loss'
  sample_size = 0.8,
  mtry = 10,
  learn_rate = 0.05
) %>%
  parsnip::set_engine("gbm") %>%
  set_mode("regression")

## ---- Workflow ----

work <- workflow() %>% 
        add_recipe(receta_1) %>%
        add_model(gb_model)