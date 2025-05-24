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
  Metrics,
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

train <- train %>% filter( loc_nmb != 'CHAPINERO')
test <- train %>% filter( loc_nmb == 'CHAPINERO')

## ---- Receta ----



## ---- Function ----



## ---- Cross Validation ----

set.seed(123)

location_folds_train <- 
  spatial_leave_location_out_cv(
    train,
    group = loc_nmb
  )

autoplot(location_folds_train)

folds_train<-list()
for(i in 1:length(location_folds_train$splits)){
  folds_train[[i]]<- location_folds_train$splits[[i]]$in_id
}

fitControl_spatial<-trainControl(method ="cv",
                                 index=folds_train)
