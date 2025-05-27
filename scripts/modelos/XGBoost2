# Validacion cruzada espacial y modelos:
Packages <- c("tidyverse", 
              "ggplot2", 
              "pacman", 
              "dplyr",
              "haven",
              "boot",
              "broom",
              "lmtest", 
              "fixest", 
              "gridExtra", 
              "writexl", 
              "readxl",
              "stargazer",
              "sf",
              "tmaptools",
              "osmdata",
              "caret",
              "FNN",
              "visdat",
              "leaflet",
              "purrr",
              "stringi", # Manipular cadenas de texto
              "rio", # Importar datos fácilmente
              "sf", # Leer/escribir/manipular datos espaciales
              "tidymodels", # entrenamiento de modelos
              "blockCV", # validacion cruzada espacial
              "spatialsample",# Muestreo espacial para modelos de aprendizaje automático)
              "ranger") 

invisible(lapply(Packages, function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)){ 
    install.packages(pkg)}
  library(pkg, character.only = TRUE)}))


# llamamos nuestras bases de datos:


setwd("C:\\Users\\claud\\OneDrive\\Documents\\OneDrive - Universidad de los andes\\Universidad Los andes\\Noveno semestre\\Big data\\taller 3\\")


Train<- st_read(dsn = "Train", layer = "Train")
names(Train)  
class(Train)    # sf + data.frame 
head(Train)     # primeras filas (tabla de atributos)
str(Train)      # estructura (geometría + atributos)

Test<- st_read(dsn = "Test", layer = "Test")
names(Test)  
class(Test)    # sf + data.frame 
head(Test)     # primeras filas (tabla de atributos)
str(Test)      # estructura (geometría + atributos)

library(dplyr)

Train <- Train %>%
  rename(property_id = prprty_)
Test <- Test %>%
  rename(property_id = prprty_)

# Utilizamos el metodo de Spatial LLOCV
set.seed(123)
location_folds <- spatial_leave_location_out_cv(
  Train,           # tu base de datos en formato sf
  group = loc_nmb         # columna que define las "localizaciones"
)


autoplot(location_folds)

folds_train <- list()
for(i in 1:length(location_folds$splits)){
  folds_train[[i]] <- location_folds$splits[[i]]$in_id
}


fitControl_spatial<-trainControl(method ="cv",
                                 index=folds_train)


library(tidymodels)
receta_1 <- recipe(price ~ srf_ttl+ surf_cv+rooms+bdrms+bathrm+prp_typ+PC1+PC2+
                         PC3+PC4+PC5+PC6+PC7+PC8+PC9+PC10+PC11+PC12+PC13+PC14+PC15+
                         PC16+PC17+PC18+PC19+PC20+nm_prqd+estrato+dist_nv+dst_hsp+
                         dst_rst+dist_pl+dist_sc+dst_clt+dst_dsc+dst_prq+dst_ttr+dist_br+
                         dst_plt+dst_stc+dst_gym+dst_jrd+dst_prk+dist_jg+dst_vpr+
                         dst_vpd+dst_cyc+dst_mll+rati_cv+bdrm_pr, data =Train) %>%
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

#XGBoost Model

xgboost_model <- boost_tree(
  trees = 200,            
  tree_depth = tune(),    
  min_n = tune(),         
  learn_rate = tune(),    
  loss_reduction = tune(), 
  sample_size = tune()     
) %>%
  set_engine("xgboost") %>%
  set_mode("regression")


# Workflow for XGBoost (train)
workflow_xgb <- workflow() %>%
  add_recipe(receta_1) %>%
  add_model(xgboost_model)


# Define hyperparameter grid for XGBoost
xgb_grid <- expand.grid(
  tree_depth = c(6, 8),    
  min_n = c(15, 30),          
  learn_rate = c(0.05, 0.1), 
  loss_reduction = 0.01,    
  sample_size = 0.8           
)


# Control settings for verbose output
control_verbose <- control_grid(
  verbose = TRUE,        
  save_pred = TRUE,     
  save_workflow = TRUE   
)


# Cross-validation tuning for XGBoost (train)
tune_results_xgb <- tune_grid(
  workflow_xgb,
  resamples = block_folds,       
  grid = xgb_grid,
  metrics = metric_set(mae),
  control = control_verbose
)

# Save OOF predictions for real_train (optional for superlearner)
oof_predictions_train_xgb <- collect_predictions(tune_results_xgb) %>%
  mutate(price_pred_oofl_train_xgb = .pred) %>%  
  select(price_pred_oofl_train_xgb)

# Print OOF predictions for inspection (optional)
print(head(oof_predictions_train_xgb))


# Select the best hyperparameters for XGBoost (real_train)
best_xgb <- select_best(tune_results_xgb, metric = "mae")
print(best_xgb) 

# Finalize workflow for XGBoost (real_train)
final_workflow_xgb <- finalize_workflow(workflow_xgb, best_xgb)

# Train the finalized workflow on the complete real_train dataset
final_fit_xgb <- fit(final_workflow_xgb, data = Train)

# Predict on the test set
test_predictions_xgb <- predict(final_fit_xgb, new_data = Test) %>%
  bind_cols(Test) %>%
  mutate(price_pred_xgb = .pred) 

# Save predictions for real_test
real_test_predictions_xgb <- test_predictions_xgb %>%
  select(property_id, price_pred_xgb)

# Submission file for Kaggle
submission_xgb <- test_predictions_xgb %>%
  mutate(price = round(price_pred_xgb, 5)) %>% 
  select(property_id, price)

write.csv(submission_xgb, "C:\\Users\\claud\\OneDrive\\Documents\\OneDrive - Universidad de los andes\\Universidad Los andes\\Noveno semestre\\Big data\\taller 3\\submissions\\1_XGB_ntrees_200_minn_30_treedepth_6.csv", row.names = FALSE)

