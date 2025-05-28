
# modelo Random Forest con tidymodels:

library(tidymodels)
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

library(spatialsample)
set.seed(86936)
block_folds <- spatial_block_cv(Train, v = 5)
block_folds
autoplot(block_folds)

modelo_Random_forest <- parsnip::rand_forest(
  mtry = tune(),          # número de variables a considerar en cada división
  min_n = tune(),         # mínimo de observaciones por nodo
  trees = 500)%>%             # número de árboles
  set_engine("ranger")%>%
  set_mode("regression")

grid_values <- expand.grid(
  mtry = c(3, 6),
  min_n = c(5, 10))

workflow_random_forest <- workflow() %>% 
  add_recipe(receta_1) %>%
  add_model(modelo_Random_forest)

#library(doParallel)
# Detecta el número de núcleos disponibles en tu CPU
#n_cores <- parallel::detectCores() - 1
# Registra el clúster de procesamiento paralelo
#registerDoParallel(cores = n_cores)
# Puedes verificar con:
#print(paste("Usando", n_cores, "núcleos para procesamiento paralelo."))


set.seed(86936)
tune_receta_1 <- tune_grid(
  workflow_random_forest,         # El flujo de trabajo que contiene: receta y especificación del modelo
  resamples = block_folds,  # Folds de validación cruzada espacial
  grid = grid_values,        # Grilla de valores de penalización
  metrics = metric_set(mae),  # metrica
  control = control_grid(verbose = TRUE)
)

#stopImplicitCluster()

collect_metrics(tune_receta_1)

# utilizo 'select_best' para seleccionar el mejor valor.
best_tune_receta_1 <- select_best(tune_receta_1, metric = "mae")
best_tune_receta_1

# Finalizar el flujo de trabajo 'workflow' con el mejor valor de parametros
receta_1_final <- finalize_workflow(workflow_random_forest, best_tune_receta_1)


# Ajustar el modelo  utilizando los datos de entrenamiento no solo la validacion cruzada
ev_random_forest_final1_fit <- fit(receta_1_final, data = Train)

predicciones <- augment(ev_random_forest_final1_fit, new_data = Test)
predicciones %>% select(prprty_, .pred)

Random_forest_mtry_6_min_n_5  <- predicciones %>%
  select(prprty_, .pred) %>%
  rename(property_id = prprty_, price = .pred) %>%
  as.data.frame()

# Descargo mis submission
write.csv(Random_forest_mtry_6_min_n_5, "C:\\Users\\samue\\OneDrive\\Escritorio\\Random_forest_mtry_6_min_n_5.csv", row.names = FALSE)

