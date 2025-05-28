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

mae_results <- train_nn(
  nn_5l_v4, "5 HL- 4096 neur, 64 batch, adam", 700, 64, 50, "adam", mae_results
)

pred_baked <- bake(receta2, new_data = pred_df, all_predictors())

x_pred <- pred_baked %>% as.matrix()
predictions <- nn_5l_v4_T2 %>% predict(x_pred)

results <- bind_cols(pred_df$prprty_, predictions)

colnames(results) <- c("property_id","price")

write.csv(results, "stores\\sub\\nn_5layers.csv", row.names = FALSE)