require(pacman)

p_load(
  keras,
  dplyr,
  sf,
  recipes,
  tidyverse
)

set.seed(123)

train_st <- st_read(dsn = "stores/work/Train", layer = "Train")
pred_st <- st_read(dsn = "stores/work/Test", layer = "Test")

train_df <- train_st %>% st_drop_geometry()
pred_df <- pred_st %>% st_drop_geometry()

## ---- Separamos entre datos de validacion y datos de train ----
test <- train_df %>% filter( loc_nmb == 'CHAPINERO')
train <- train_df %>% filter( loc_nmb != 'CHAPINERO')

## ---- Especificamos la receta y realizamos preprocesing ----
receta <- recipe(price ~ srf_ttl+surf_cv+rooms+bdrms+bathrm+prp_typ+PC1+PC2+
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

receta2 <- prep(receta, training = train)

## ---- Creamos matriz con predictores ----

x_train <- bake(receta2, new_data = train, all_predictors()) %>% as.matrix()
x_val <- bake(receta2, new_data = test, all_predictors()) %>% as.matrix()

## ---- Creamos matriz con variables dependientes ----

y_train <- bake(receta2, new_data = train, all_outcomes()) %>% as.matrix()
y_val <- bake(receta2, new_data = test, all_outcomes()) %>% as.matrix()

## ---- Verificamos dimensiones ----

dim(x_train)
dim(x_val)

dim(y_train)
dim(y_val)

## ---- Creamos df Para Guardar los minimos MAE de cada modelo ----
mae_results <- data.frame(
  name = character(),
  val_mae = numeric()
)

## ---- Creamos funcion que entrena modelos y guarda el MAE mínimo ----

train_nn <- function(model, name, epochs_num, batch, pat, optim, res_df){
  
  early_stop <- callback_early_stopping(
    monitor = "val_loss",
    patience = pat,
    mode = "min",
    restore_best_weights = TRUE
  )
  
  model %>% compile(
    optimizer = optim,
    loss = 'mean_squared_error',
    metrics = c('mean_absolute_error')
  )
  
  history <- model %>% fit(
    x = x_train,
    y = y_train,
    epochs = epochs_num,
    batch_size = batch,
    validation_data = list(x_val, y_val),
    callbacks = list(early_stop),
    verbose = 1
  )
  
  eval <- model %>% evaluate(x_val, y_val, verbose = 0)
  
  #val_mae <- eval$mean_absolute_error
  
  val_mae <- eval[["mean_absolute_error"]] 
  
  r <- data.frame(
    name = name,
    val_mae = val_mae
  )
  
  res_df <- rbind(res_df, r)
  
  return(res_df)
}

## ---- Definimos Redes ----

# Red con 5 capas ocultas
nn_5l_v1 <- keras_model_sequential() %>% 
  layer_dense(units = 512, activation = 'relu', input_shape = ncol(x_train)) %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 256, activation = 'relu') %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 64, activation = 'relu') %>% 
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units=1, activation = 'linear')

nn_5l_v1_T1 <- nn_5l_v1
nn_5l_v1_T2 <- nn_5l_v1
nn_5l_v1_T3 <- nn_5l_v1
nn_5l_v1_T4 <- nn_5l_v1

mae_results <- train_nn(
  nn_5l_v1_T1, "5 HL- 512 neur, 128 batch, adam", 700, 128, 50, "adam", mae_results
)

mae_results <- train_nn(
  nn_5l_v1_T2, "5 HL- 512 neur, 64 batch, adam", 700, 64, 50, "adam", mae_results
)

mae_results <- train_nn(
  nn_5l_v1_T3, "5 HL- 512 neur, 128 batch, sgd", 700, 128, 50, "sgd", mae_results
)

mae_results <- train_nn(
  nn_5l_v1_T4, "5 HL- 512 neur, 64 batch, sgd", 700, 64, 50, "sgd", mae_results
)

# Mas Nodos
nn_5l_v2 <- keras_model_sequential() %>% 
  layer_dense(units = 1024, activation = 'relu', input_shape = ncol(x_train)) %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 512, activation = 'relu') %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 256, activation = 'relu') %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units=1, activation = 'linear')

nn_5l_v2_T1 <- nn_5l_v2
nn_5l_v2_T2 <- nn_5l_v2
nn_5l_v2_T3 <- nn_5l_v2
nn_5l_v2_T4 <- nn_5l_v2

mae_results <- train_nn(
  nn_5l_v2_T1, "5 HL- 1024 neur, 128 batch, adam", 700, 128, 50, "adam", mae_results
)

mae_results <- train_nn(
  nn_5l_v2_T2, "5 HL- 1024 neur, 64 batch, adam", 700, 64, 50, "adam", mae_results
)

mae_results <- train_nn(
  nn_5l_v2_T3, "5 HL- 1024 neur, 128 batch, sgd", 700, 128, 50, "sgd", mae_results
)

mae_results <- train_nn(
  nn_5l_v2_T4, "5 HL- 1024 neur, 64 batch, sgd", 700, 64, 50, "sgd", mae_results
)


## ---- Red con 8 capas ocultas pero mas nodos ----

# Red con 5 capas ocultas
nn_8l <- keras_model_sequential() %>% 
  layer_dense(units = 1024, activation = 'relu', input_shape = ncol(x_train)) %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 512, activation = 'relu') %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 256, activation = 'relu') %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 16, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 8, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units=1, activation = 'linear')

nn_8l_T1 <- nn_8l
nn_8l_T2 <- nn_8l
nn_8l_T3 <- nn_8l
nn_8l_T4 <- nn_8l

mae_results <- train_nn(
  nn_8l_T1, "8 HL- 1024 neur, 128 batch, adam", 700, 128, 50, "adam", mae_results
)

mae_results <- train_nn(
  nn_8l_T2, "8 HL- 1024 neur, 64 batch, adam", 700, 64, 50, "adam", mae_results
)

mae_results <- train_nn(
  nn_8l_T3, "8 HL- 1024 neur, 128 batch, sgd", 700, 128, 50, "sgd", mae_results
)

mae_results <- train_nn(
  nn_8l_T4, "8 HL- 1024 neur, 64 batch, sgd", 700, 64, 50, "sgd", mae_results
)

## ---- Red con 3 capas ocultas ----

nn_3l <- keras_model_sequential() %>% 
  layer_dense(units = 1024, activation = 'relu', input_shape = ncol(x_train)) %>% 
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 512, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 256, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units=1, activation = 'linear')

nn_3l_T1 <- nn_3l
nn_3l_T2 <- nn_3l
nn_3l_T3 <- nn_3l
nn_3l_T4 <- nn_3l

mae_results <- train_nn(
  nn_3l_T1, "3 HL- 1024 neur, 128 batch, adam", 700, 128, 50, "adam", mae_results
)

mae_results <- train_nn(
  nn_3l_T2, "3 HL- 1024 neur, 64 batch, adam", 700, 64, 50, "adam", mae_results
)

mae_results <- train_nn(
  nn_3l_T3, "3 HL- 1024 neur, 128 batch, sgd", 700, 128, 50, "sgd", mae_results
)

mae_results <- train_nn(
  nn_3l_T4, "3 HL- 1024 neur, 64 batch, sgd", 700, 64, 50, "sgd", mae_results
)

## ---- probanmos con 5 capas otra vez pero menos nodos ----

nn_5l_v3 <- keras_model_sequential() %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = ncol(x_train)) %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 64, activation = 'relu') %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 32, activation = 'relu') %>% 
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 8, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units=1, activation = 'linear')

nn_5l_v3_T1 <- nn_5l_v3
nn_5l_v3_T2 <- nn_5l_v3
nn_5l_v3_T3 <- nn_5l_v3
nn_5l_v3_T4 <- nn_5l_v3

mae_results <- train_nn(
  nn_5l_v3_T1, "5 HL- 256 neur, 128 batch, adam", 700, 128, 50, "adam", mae_results
)

mae_results <- train_nn(
  nn_5l_v3_T2, "5 HL- 256 neur, 64 batch, adam", 700, 64, 50, "adam", mae_results
)

mae_results <- train_nn(
  nn_5l_v3_T3, "5 HL- 256 neur, 128 batch, sgd", 700, 128, 50, "sgd", mae_results
)

mae_results <- train_nn(
  nn_5l_v3_T4, "5 HL- 256 neur, 64 batch, sgd", 700, 64, 50, "sgd", mae_results
)



## ---- Volvemos a probar con 5 capas pero mas nodos ----

nn_5l_v4 <- keras_model_sequential() %>% 
  layer_dense(units = 4096, activation = 'relu', input_shape = ncol(x_train)) %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 1024, activation = 'relu') %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 512, activation = 'relu') %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 256, activation = 'relu') %>% 
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units=1, activation = 'linear')

nn_5l_v4_T1 <- nn_5l_v4
nn_5l_v4_T2 <- nn_5l_v4
nn_5l_v4_T3 <- nn_5l_v4
nn_5l_v4_T4 <- nn_5l_v4


mae_results <- train_nn(
  nn_5l_v4_T1, "5 HL- 4096 neur, 128 batch, adam", 700, 128, 50, "adam", mae_results
)

mae_results <- train_nn(
  nn_5l_v4_T2, "5 HL- 4096 neur, 64 batch, adam", 700, 64, 50, "adam", mae_results
)


## ---- Miramos los resultados ----
mae_results <- na.omit(mae_results)

mae_results <- mae_results[order(mae_results$val_mae, decreasing= F), ]

head(mae_results)

# El mejor modelo fue el de 5 capas con batch de 64 (nn_5l_v4) y 4096 

## ---- Nuevo parametros mejor modelo ----

nn_5l_v4_T5 <- nn_5l_v4
nn_5l_v4_T6 <- nn_5l_v4

mae_results <- train_nn(
  nn_5l_v4_T5, "5 HL- 4096 neur, 256 batch, adam", 700, 256, 50, "adam", mae_results
)

mae_results <- train_nn(
  nn_5l_v4_T6, "5 HL- 4096 neur, 32 batch, adam", 700, 64, 50, "adam", mae_results
)

nn_5l_v5 <- keras_model_sequential() %>% 
  layer_dense(units = 4096, activation = 'relu', input_shape = ncol(x_train)) %>% 
  layer_dense(units = 1024, activation = 'relu') %>% 
  layer_dense(units = 512, activation = 'relu') %>% 
  layer_dense(units = 256, activation = 'relu') %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units=1, activation = 'linear')

nn_5l_v5_T1 <- nn_5l_v5
nn_5l_v5_T2 <- nn_5l_v5

mae_results <- train_nn(
  nn_5l_v5_T1, "5 HL- 4096 neur, 128 batch, adam, no drop", 700, 128, 50, "adam", mae_results
)

mae_results <- train_nn(
  nn_5l_v5_T2, "5 HL- 4096 neur, 256 batch, adam, no drop", 700, 256, 50, "adam", mae_results
)

nn_5l_v6 <- keras_model_sequential() %>% 
  layer_dense(units = 4096, activation = 'relu', input_shape = ncol(x_train)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 1024, activation = 'relu') %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 512, activation = 'relu') %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 256, activation = 'relu') %>% 
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units=1, activation = 'linear')

nn_5l_v6_T1 <- nn_5l_v6
nn_5l_v6_T2 <- nn_5l_v6

mae_results <- train_nn(
  nn_5l_v6_T1, "5 HL- 4096 neur, 128 batch, adam, alt drop", 700, 128, 50, "adam", mae_results
)

mae_results <- train_nn(
  nn_5l_v6_T2, "5 HL- 4096 neur, 256 batch, adam, alt drop", 700, 256, 50, "adam", mae_results
)

## ---- Predecimos ----

pred_baked <- bake(receta2, new_data = pred_df, all_predictors())

x_pred <- pred_baked %>% as.matrix()
predictions <- nn_5l_v4_T2 %>% predict(x_pred)

results <- bind_cols(pred_df$prprty_, predictions)

colnames(results) <- c("property_id","price")

write.csv(results, "stores\\sub\\nn_keras_final.csv", row.names = FALSE)

