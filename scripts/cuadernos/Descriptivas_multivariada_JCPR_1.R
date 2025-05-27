#======================================================================
#  DESCRIPTIVOS GRÁFICOS Y TABULARES  –  Train_localizado.shp
#======================================================================
#  1) Box-plots facetados de todas las variables numéricas
#  2) Tabla con media, mediana, sd y curtosis
#  3) Matriz de correlaciones + heat-map
#  4) Colinealidad respecto a ‘price’ (VIF)
#----------------------------------------------------------------------
#  •  Directorio de salida:  stores/work_jcp/metadata_descriptivas
#  •  Parámetros de imagen configurables al inicio
#======================================================================

if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
  sf, dplyr, tidyr, ggplot2, patchwork,
  gt, webshot2, moments,
  ggcorrplot, car, readr
)

# ---------- PARÁMETROS DEL USUARIO -----------------------------------
shp_path  <- "stores/work_jcp/Datasets/4/Train/Train_localizado.shp"
out_dir   <- "stores/work_jcp/metadata_descriptivas"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Tamaños de las imágenes (pulgadas)
img_w  <- 9
img_h  <- 6
dpi_png <- 300

# Fuente para cajas
ggplot2::theme_set(theme_bw(base_size = 9))




#======================================================================
#  A.  Lectura y preparación ------------------------------------------
#======================================================================
train_sf  <- sf::st_read(shp_path, quiet = TRUE)
train_df  <- st_drop_geometry(train_sf)

# seleccionar numéricas (excluimos 'price' para algunos análisis)
is_num    <- sapply(train_df, is.numeric)
num_df    <- train_df[, is_num]



#======================================================================
#  1. Box-plots facetados ---------------------------------------------
#======================================================================
#  1-bis. Box-plots en bloques de 16 variables ‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒---------
#======================================================================




# filtrar variables numéricas SIN PCs  -----------------
is_num   <- sapply(train_df, is.numeric)
num_df   <- train_df[, is_num]

# filtrado de otras variables
vars_skip <- c("shp_ar2", "shp_len", "shp_l", "shp_are", "obj_id", "cod_zon", "lat", "lon")
num_df <- num_df[ , !names(num_df) %in% vars_skip ]  # las elimina


# patrón PCxx
pcs_regex <- "^PC[0-9]+$"
num_df    <- num_df[ , !grepl(pcs_regex, names(num_df)) ]

# long format para ggplot
num_long <- num_df %>%
  tidyr::pivot_longer(everything(),
                      names_to  = "variable",
                      values_to = "value") %>%
  filter(!is.na(value))

# ----------------  parámetros de bloques -----------------------------
chunk_size <- 16         # ⇒ 4 × 4
n_chunks   <- ceiling(ncol(num_df) / chunk_size)

# helper para guardar cada bloque
guardar_box_chunk <- function(idx){
  vars <- names(num_df)[ ((idx-1)*chunk_size + 1) :
                         min(idx*chunk_size, ncol(num_df)) ]
  
  if (length(vars) == 0) return()   # por si sobra
  
  gg <- ggplot(num_long %>% filter(variable %in% vars),
               aes(x = "", y = value)) +
        geom_boxplot(fill = "#5592c9", outlier.size = .7, na.rm = TRUE) +
        facet_wrap(~ variable, scales = "free", ncol = 4) +
        labs(title = paste("Distribución de variables numéricas (bloque",
                           sprintf("%02d", idx), ")"),
             x = NULL, y = NULL) +
        theme(axis.text.x  = element_blank(),
              axis.ticks.x = element_blank())
  
  ggsave(
    filename = file.path(out_dir,
              paste0("boxplot_", sprintf("%02d", idx), ".png")),
    plot   = gg,
    width  = img_w, height = img_h, dpi = dpi_png
  )
}

purrr::walk(seq_len(n_chunks), guardar_box_chunk)

cat("✔ Box-plots generados (sin PCs) en", normalizePath(out_dir), "\n")

# Megapanel
# ggsave(box_png, gg_box, width = img_w, height = img_h, dpi = dpi_png)





#======================================================================
#  2. Estadísticos básicos --------------------------------------------
#======================================================================

# Version 4 - Solo 4 estadisticos y Variables 100% numericas y sin NA
library(dplyr)
library(tidyr)
library(moments)
library(gt)
library(webshot2)
library(readr)

#──────────────────────────────────────────────────────────────────────────
# 1.  Variables numéricas con al menos un valor finito
#──────────────────────────────────────────────────────────────────────────
num_df_valid <- num_df %>%                # <- num_df se generó en el paso previo
  select(where(~ any(is.finite(.x))))

#──────────────────────────────────────────────────────────────────────────
# 2.  Estadísticos y filtrado
#──────────────────────────────────────────────────────────────────────────
stats_tbl <- num_df_valid %>% 
  summarise(
    across(
      everything(),
      .fns = list(
        mean   = ~ mean(.x,  na.rm = TRUE),
        median = ~ median(.x, na.rm = TRUE),
        sd     = ~ sd(.x,    na.rm = TRUE),
        kurt   = ~ moments::kurtosis(.x, na.rm = TRUE),
        min    = ~ min(.x,   na.rm = TRUE),
        max    = ~ max(.x,   na.rm = TRUE),
        range  = ~ max(.x,   na.rm = TRUE) - min(.x, na.rm = TRUE)
      ),
      .names = "{.col}_{.fn}"
    )
  ) %>% 
  pivot_longer(
    everything(),
    names_to  = c("variable", ".value"),
    names_sep = "_"
  ) %>% 
  # Conservar solo filas con media, mediana y sd finitas
  filter(across(c(mean, median, sd), is.finite)) %>% 
  # Conservar y ordenar columnas requeridas
  select(variable, mean, median, sd, kurt, min, max, range)

#──────────────────────────────────────────────────────────────────────────
# 3.  Exportar CSV y PNG
#──────────────────────────────────────────────────────────────────────────
write_csv(stats_tbl, file.path(out_dir, "stats_basic_reduced.csv"))

gt(stats_tbl) %>% 
  fmt_number(columns = where(is.numeric), decimals = 2) %>% 
  tab_header(title = "Estadísticos básicos (cuadro reducido)") %>% 
  gtsave(
    file.path(out_dir, "stats_basic_reduced.png"),
    vheight = 120 + nrow(stats_tbl) * 18,
    zoom    = 2,
    expand  = 5
  )

cat(
  "✔ Cuadro reducido guardado en:\n  -",
  file.path(out_dir, "stats_basic_reduced.csv"), "\n  -",
  file.path(out_dir, "stats_basic_reduced.png"), "\n"
)






# Version 3- Variables 100% numericas, sin NA.  -------------------
# -----------------------------------------------------------------------------
library(dplyr); library(tidyr); library(moments); library(gt); library(webshot2); library(readr)
# 1.  Seleccionar variables numéricas con al menos un valor finito
#----------------------------------------------------------------------
num_df_valid <- num_df %>%               # num_df viene del paso previo
  dplyr::select(where(~ any(is.finite(.x))))

# 2.  Resumir y descartar filas con mean/median/sd = NA
#----------------------------------------------------------------------
stats_tbl <- num_df_valid %>%
  summarise(across(everything(),
          list(mean   = ~mean(.x,  na.rm = TRUE),
               median = ~median(.x, na.rm = TRUE),
               sd     = ~sd(.x,    na.rm = TRUE),
               kurt   = ~moments::kurtosis(.x, na.rm = TRUE)),
               min = ~min(.x, na.rm = TRUE),
               max = ~max(.x, na.rm = TRUE),
               range = ~max(.x, na.rm = TRUE) - min(.x, na.rm = TRUE)),
          .names = "{.col}_{.fn}")) %>%
  pivot_longer(everything(),
               names_to  = c("variable", ".value"),
               names_sep = "_") %>%
  # ── filtrar sólo filas donde mean, median y sd son numéricos finitos ──
  filter(across(c(mean, median, sd), ~ is.finite(.x)))


# 3.  Exportar CSV y PNG
#----------------------------------------------------------------------
write_csv(stats_tbl, file.path(out_dir, "stats_basic_clean.csv"))

stats_png <- file.path(out_dir, "stats_basic_clean.png")

gt(stats_tbl) %>%
  fmt_number(columns = where(is.numeric), decimals = 2) %>%
  tab_header(title = "Estadísticos básicos — columnas con datos válidos") %>%
  gtsave(stats_png,
         vheight = 120 + nrow(stats_tbl) * 18,   # tamaño dinámico
         zoom    = 2,
         expand  = 5)

cat("✔ Estadísticos básicos generados para",
    nrow(stats_tbl), "variables.\n",
    "Archivos:\n  -",
    file.path(out_dir, "stats_basic_clean.csv"), "\n  -",
    stats_png, "\n")






# Version 2 - Variables con al menos un valor numerico

# — 2.1  Mantener solo variables con al menos un valor finito ----------
num_df_clean <- num_df %>%
  dplyr::select(where(~ any(is.finite(.x))))   # descarta columnas vacías

# — 2.2  Cálculo de media, mediana, sd, curtosis -----------------------
stats_tbl <- num_df_clean %>%
  summarise(across(everything(),
          list(mean   = ~mean(.x,  na.rm = TRUE),
               median = ~median(.x, na.rm = TRUE),
               sd     = ~sd(.x,    na.rm = TRUE),
               min  = ~min(.x,    na.rm = TRUE),
               max  = ~max(.x,    na.rm = TRUE),
               range = ~max(.x, na.rm = TRUE) - min(.x, na.rm = TRUE),
               kurt   = ~moments::kurtosis(.x, na.rm = TRUE)),
          .names = "{.col}_{.fn}")) %>%
  tidyr::pivot_longer(everything(),
                      names_to  = c("variable", ".value"),
                      names_sep = "_")

# — 2.3  Exportar CSV ---------------------------------------------------
readr::write_csv(stats_tbl,
                 file.path(out_dir, "stats_basic.csv"))

# — 2.4  Exportar imagen PNG vía {gt} ----------------------------------
stats_png <- file.path(out_dir, "stats_basic.png")

gt::gt(stats_tbl) %>%
  fmt_number(columns = where(is.numeric), decimals = 2) %>%
  tab_header(title = "Estadísticos básicos — Train") %>%
  gtsave(stats_png,
         vheight = 120 + nrow(stats_tbl) * 18,   # alto dinámico
         zoom    = 2,
         expand  = 5)

cat("✔ Tabla de estadísticos generada sin columnas todo-NA.\n")


# Version 1 - Todas las varaibles

stats_tbl <- num_df %>%
  summarise(across(everything(),
          list(mean = ~mean(.x, na.rm = TRUE),
               median = ~median(.x, na.rm = TRUE),
               sd = ~sd(.x, na.rm = TRUE),
               kurt = ~moments::kurtosis(.x, na.rm = TRUE)),
          .names = "{.col}_{.fn}")) %>%
  pivot_longer(everything(),
               names_to  = c("variable", ".value"),
               names_sep = "_")

# CSV
write_csv(stats_tbl, file.path(out_dir, "stats_basic.csv"))

# PNG vía gt
stats_png <- file.path(out_dir, "stats_basic.png")
gt(stats_tbl) %>%
  fmt_number(columns = where(is.numeric), decimals = 2) %>%
  tab_header(title = "Estadísticos básicos — Train") %>%
  gtsave(stats_png, vheight = 120 + nrow(stats_tbl)*18,
         zoom = 2, expand = 5)









#======================================================================
#  3. Matriz de correlaciones  (filtrando varianza 0)
#======================================================================

# --- 3.1  quitar columnas con varianza cero o <2 valores finitos -------
nzv <- sapply(num_df, function(x){
          vals <- x[is.finite(x)]
          length(vals) > 1 && sd(vals) > 0
        })
num_corr <- num_df[ , nzv]

if (ncol(num_corr) < 2){
  stop("❌ No quedan suficientes variables para calcular correlaciones.")
}

# --- 3.2  correlación ---------------------------------------------------
corr_mat <- cor(num_corr, use = "pairwise.complete.obs")

# --- 3.3  exportar CSV --------------------------------------------------
write_csv(as.data.frame(corr_mat),
          file.path(out_dir, "corr_matrix.csv"))

# --- 3.4  preparar para plot (NA → 0 sólo visual) ----------------------
corr_plot <- corr_mat
corr_plot[is.na(corr_plot)] <- 0     # evita NA en hclust()
corr_png <- file.path(out_dir, "corr_matrix.png")

# ── parámetros de tamaño ────────────────────────────────────────────
lab_sz  <- 1.6   # números dentro de cada celda
tl_sz   <- 6     # etiquetas de filas/columnas
w_plot  <- 12    # pulgadas   ← ajusta a tu gusto
h_plot  <- 14    # pulgadas

p_corr <- ggcorrplot(
           corr_plot,
           type     = "lower",
           hc.order = TRUE,
           lab      = TRUE,
           lab_size = lab_sz,
           tl.cex   = tl_sz,
           tl.srt   = 45         # etiquetas en diagonal
         ) +
         scale_fill_gradient2(
           low  = "steelblue4",
           mid  = "white",
           high = "firebrick3",
           midpoint = 0,
           limits = c(-1, 1),
           na.value = "grey90"     # ← aquí sí se acepta
         ) +
         ggtitle("Matriz de correlaciones — Train")


ggsave(corr_png, p_corr, width = img_w, height = img_h, dpi = dpi_png)

cat("✔ Matriz de correlaciones creada con",
    ncol(num_corr), "variables (se omitieron",
    sum(!nzv), "sin varianza).\n")





#======================================================================
#  Mensaje final -------------------------------------------------------
#======================================================================
cat("✔ Resultados almacenados en:", normalizePath(out_dir), "\n",
    "- boxplot_matrix.png\n",
    "- stats_basic.(csv | png)\n",
    "- corr_matrix.(csv | png)\n")
