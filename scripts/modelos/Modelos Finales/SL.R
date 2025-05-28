# Superlearners:
listWrappers()




ySL <- Train$price
XSL <- Train  %>% select(srf_ttl, surf_cv, rooms, bdrms, bathrm, prp_typ, PC1, PC2,
                         PC3, PC4, PC5, PC6, PC7, PC8, PC9, PC10, PC11, PC12, PC13, PC14, PC15,
                         PC16, PC17, PC18, PC19, PC20, PC21, PC22, PC23, PC24, PC25, PC26,
                         PC27, PC28, PC29, PC30, nm_prqd, estrato, dist_nv, dst_hsp,
                         dst_rst, dist_pl, dist_sc, dst_clt, dst_dsc, dst_prq, dst_ttr, dist_br,
                         dst_plt, dst_stc, dst_gym, dst_jrd, dst_prk, dist_jg, dst_vpr,
                         dst_vpd, dst_cyc, dst_mll, rati_cv, bdrm_pr) %>% st_drop_geometry()

head(XSL)
XSL <- XSL %>%
  mutate(estrato = factor(estrato))
# Separo numéricas y categóricas
XSL_num <- XSL %>% select(where(is.numeric))
XSL_cat <- XSL %>% select(where(negate(is.numeric)))

dummies <- dummyVars(" ~ .", data = XSL_cat)
XSL_cat_dummies <- predict(dummies, newdata = XSL_cat)

XSL_final <- cbind(scale(XSL_num), XSL_cat_dummies) %>% as.data.frame()


library(spatialsample)
set.seed(123)
block_folds <- spatial_block_cv(Train, v = 5)   # 5 bloques espaciales

validRows <- lapply(block_folds$splits, function(x) {
  match(rownames(assessment(x)), rownames(Train))
})



length(validRows)  # Debe dar 5 folds
length(validRows[[1]])  # Número de observaciones en validación del primer fold

library(SuperLearner)
# Neural Net (nnet) con size y decay
nnet_grid <- create.Learner("SL.nnet",
                            tune = list(size  = c(5, 8),
                                        decay = c(0, 0.1)),
                            detailed_names = TRUE)

# XGBoost con eta y max_depth
xgb_grid <- create.Learner("SL.xgboost",
                           tune = list(eta       = c(0.1, 0.3),
                                       max_depth = c(3, 6),
                                       nrounds   = 50),
                           detailed_names = TRUE)

# Regresión lineal simple (baseline)
sl.lib <- c(nnet_grid$names, xgb_grid$names, "SL.lm")

anyNA(XSL_final)
any(!is.finite(as.matrix(XSL_final)))


fitY <- SuperLearner(Y = ySL,
                     X = XSL_final,
                     SL.library = sl.lib,
                     method = "method.NNLS",
                     cvControl = list(V = 5, validRows = validRows),
                     verbose = TRUE)



