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
  yardstick,
  forcats,
  fastDummies
)


## ---- Cargue de datos ----

train <- st_read(dsn = "stores\\work\\Train", layer = "Train")
pred <- st_read(dsn = "stores\\work\\Test", layer = "Test")

## ---- Dividimos los datos de entrenamiento ----

test <- train %>% filter( loc_nmb == 'CHAPINERO')
train <- train %>% filter( loc_nmb != 'CHAPINERO')

## ---- Receta ----


## ---- Obetenmos matrices ----



## ---- Definimos el modelo ----


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

grid <-  expand.grid(
  n.trees= c(100,150,300),
  interaction.depth=c(1,2,5),
  shrinkage=c(0.01),
  n.minobsinnode=c(5, 10))


modelo <- train(price ~ srf_ttl+surf_cv+rooms+bdrms+bathrm+prp_typ+PC1+PC2+
                  PC3+PC4+PC5+PC6+PC7+PC8+PC9+PC10+PC11+PC12+PC13+PC14+PC15+
                  PC16+PC17+PC18+PC19+PC20+PC21+PC22+PC23+PC24+PC25+PC26+
                  PC27+PC28+PC29+PC30+nm_prqd+estrato+dist_nv+dst_hsp+
                  dst_rst+dist_pl+dist_sc+dst_clt+dst_dsc+dst_prq+dst_ttr+dist_br+
                  dst_plt+dst_stc+dst_gym+dst_jrd+dst_prk+dist_jg+dst_vpr+
                  dst_vpd+dst_cyc+dst_mll+rati_cv+bdrm_pr, 
                data = train, 
                method = "gbm",        
                metric = "MAE",               
                trControl = fitControl_spatial,       
                tuneGrid = grid)

test$price_hat<-predict(modelo,newdata = test)

mean(abs(test$price-test$price_hat))

predic<-predict(modelo,newdata = pred)

results <- bind_cols(pred$prprty_, predic)
                  
colnames(results) <- c("property_id","price")

write.csv(results, "stores\\sub\\gbm.csv", row.names = FALSE)



