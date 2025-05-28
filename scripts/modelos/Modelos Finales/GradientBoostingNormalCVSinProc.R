## ---------------------------
##
## Script name: GradientBoostingCVNormal.R
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
## Notes: Se eliminó la dependencia de 'spatialsample' y se configuró 'caret'
##        para usar validación cruzada k-fold estándar.
##
## ---------------------------

# Limpiar el entorno de trabajo
rm(list = ls())

# Cargar paquetes requeridos
require(pacman)

p_load(
  caret,
  dplyr,
  Metrics,
  gbm,
  sf,
  tidymodels,
  recipes,
  # spatialsample, # Ya no es necesario para CV no espacial
  parsnip,
  rsample,
  workflows,
  tune,
  tidyverse,
  dials,
  yardstick,
  forcats,
  fastDummies
)


## ---- Cargue de datos ----

# Cargar datos de entrenamiento y predicción
train <- st_read(dsn = "stores\\work\\Train", layer = "Train")
pred <- st_read(dsn = "stores\\work\\Test", layer = "Test")

## ---- Dividimos los datos de entrenamiento ----

# Separar un conjunto de prueba (Chapinero) y uno de entrenamiento
test <- train %>% filter( loc_nmb == 'CHAPINERO')
train <- train %>% filter( loc_nmb != 'CHAPINERO')

## ---- Receta ----

## ---- Cross Validation (No Espacial) ----

set.seed(123) # Para reproducibilidad

# Configurar el control de entrenamiento para validación cruzada k-fold (ej. 10 folds)
fitControl <- trainControl(
  method = "cv", # Método de validación cruzada
  number = 10,   # Número de folds (k)
  verboseIter = TRUE # Muestra el progreso
)

## ---- Definición de la Grilla de Hiperparámetros ----

# Definir la grilla de hiperparámetros a probar
grid <-  expand.grid(
  n.trees = c(100, 150, 300),
  interaction.depth = c(1, 2, 5),
  shrinkage = c(0.01),
  n.minobsinnode = c(5, 10)
)

## ---- Entrenamiento del Modelo ----

# Entrenar el modelo Gradient Boosting Machine (GBM)
modelo <- train(
  price ~ srf_ttl+surf_cv+rooms+bdrms+bathrm+prp_typ+PC1+PC2+
    PC3+PC4+PC5+PC6+PC7+PC8+PC9+PC10+PC11+PC12+PC13+PC14+PC15+
    PC16+PC17+PC18+PC19+PC20+PC21+PC22+PC23+PC24+PC25+PC26+
    PC27+PC28+PC29+PC30+nm_prqd+estrato+dist_nv+dst_hsp+
    dst_rst+dist_pl+dist_sc+dst_clt+dst_dsc+dst_prq+dst_ttr+dist_br+
    dst_plt+dst_stc+dst_gym+dst_jrd+dst_prk+dist_jg+dst_vpr+
    dst_vpd+dst_cyc+dst_mll+rati_cv+bdrm_pr,
  data = train,
  method = "gbm",
  metric = "MAE", # Métrica a optimizar (Mean Absolute Error)
  trControl = fitControl, # Usar el control de entrenamiento no espacial
  tuneGrid = grid,
  verbose = FALSE # Para reducir la salida de gbm
)

## ---- Evaluación y Predicción ----

test$price_hat <- predict(modelo, newdata = test)


mae_test <- mean(abs(test$price - test$price_hat))
print(paste("MAE en el conjunto de prueba (Chapinero):", mae_test))

predic <- predict(modelo, newdata = pred)

## ---- Guardar Resultados ----


results <- bind_cols(pred$prprty_, predic)

colnames(results) <- c("property_id", "price")

# Guardar los resultados en un archivo CSV
write.csv(results, "stores\\sub\\gbm_no_espacial_sin_proc.csv", row.names = FALSE)
