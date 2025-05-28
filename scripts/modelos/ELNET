
# Modelo Elastic net:


library(tidymodels)
receta_1 <- recipe(price ~ surf_total + surf_cov + rooms + bdrms + bathrm + prop_typ +
                     PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 +
                     PC11 + PC12 + PC13 + PC14 + PC15 + PC16 + PC17 + PC18 + PC19 + PC20 +
                     PC21 + PC22 + PC23 + PC24 + PC25 + PC26 + PC27 + PC28 + PC29 + PC30 +
                     num_parqueaderos + estrato + dist_univ + dist_hosp + dist_rest + dist_pol +
                     dist_esc + dist_culto + dist_disco + dist_parq + dist_teatr + dist_bar +
                     dist_plat + dist_estac + dist_gym + dist_jard + dist_park + dist_juego +
                     dist_vprim + dist_vped + dist_cycl + dist_mall + ratio_cov + bdrm_perr,
                   data = Train_localizado) %>%
  step_novel(all_nominal_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_interact(terms = ~ surf_total:estrato) %>% 
  step_interact(terms = ~ dist_vprim:estrato) %>% 
  step_poly(dist_vprim, degree = 2) %>%
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors())


library(spatialsample)
set.seed(86936)
block_folds <- spatial_block_cv(Train_localizado, v = 5)
block_folds
autoplot(block_folds)

# especificacion del modelo:
elastic_net <- parsnip::linear_reg(
  penalty = tune(), 
  mixture = tune()) %>%
  set_engine("glmnet")

grid_values <- grid_regular(penalty(range = c(-2,1)), levels = 100) %>%
  expand_grid(mixture = c(0,0.0001, 0.0005, 0.001, 0.005, 0.01,0.05,0.10,0.15,0.20, 0.25, 0.35, 0.5, 0.65, 0.75,  1))

workflow_elastic_net <- workflow() %>% 
  add_recipe(receta_1) %>%
  add_model(elastic_net)

set.seed(86936)
tune_receta_1 <- tune_grid(
  workflow_elastic_net,         # El flujo de trabajo que contiene: receta y especificación del modelo
  resamples = block_folds,  # Folds de validación cruzada espacial
  grid = grid_values,        # Grilla de valores de penalización
  metrics = metric_set(yardstick::mae)  # metrica
)

collect_metrics(tune_receta_1)


# utilizo 'select_best' para seleccionar el mejor valor.
best_tune_receta_1 <- select_best(tune_receta_1, metric = "mae")
best_tune_receta_1

# Finalizar el flujo de trabajo 'workflow' con el mejor valor de parametros
receta_1_final <- finalize_workflow(workflow_elastic_net, best_tune_receta_1)


# Ajustar el modelo  utilizando los datos de entrenamiento no solo la validacion cruzada
ev_ELNET_final1_fit <- fit(receta_1_final, data = Train_localizado)

predicciones <- augment(ev_ELNET_final1_fit, new_data = Test_localizado)
predicciones %>% select(prop_id, .pred)

ELNET_penalty_0.01_mixture_0.001  <- predicciones %>%
  select(prop_id, .pred) %>%
  rename(property_id = prop_id, price = .pred) %>%
  as.data.frame()

# Descargo mis submission
write.csv(ELNET_penalty_0.01_mixture_0.001, "C:\\Users\\samue\\OneDrive\\Escritorio\\ELNET_penalty_0.01_mixture_0.001.csv", row.names = FALSE)
