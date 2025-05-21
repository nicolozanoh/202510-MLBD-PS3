## ---------------------------
##
## Script name: 01_Limpieza_Datos.R
##
## Purpose of script: Organizar datos para el problem set 3.
##
## Author: Nicolas Lozano, Jhan Pulido, Gerardo Rendon, Samuel Narvaez
##
## Date Created: 2025-05-06
##
## ---------------------------
##
## Notes:
##   
##
## ---------------------------`

## ---- 01 Limpiamos Ambiente y cargamos paquetes ----
rm(list = ls())

require(pacman)

p_load(
  "ggplot2",
  "tidyverse",
  "dplyr",
  "visdat"
)

## ---- 02 Cargamos datos ----

test <- read.csv(
  "stores\\test.csv"
)

train <- read.csv(
  "stores\\train.csv"
)

## ---- 03 Organizamos tipos de las variables ----

train <- train %>%
  mutate(
    year = as.factor(year),
    month = as.factor(month))

test <- test %>%
  mutate(
    year = as.factor(year),
    month = as.factor(month))

## ---- 04 Balance Tipo de Propiedad ----

train %>%
  count(property_type)

test%>%
  count(property_type)

## ---- 05 Analizamos Missing Values ----

train <- train %>%
  mutate(title = na_if(title, "")) %>%
  mutate( description= na_if(description, ""))

vis_miss(train)

#80% de los datos missing en surface_total
#78% de los datos missing en surface_covered


test <- test %>%
  mutate(title = na_if(title, "")) %>%
  mutate( description= na_if(description, ""))

vis_miss(test)


## ---- 06 Imputamos Valores para Variables con Missing Values ----

train %>% count(rooms) %>% head()


## ---- 03 Creación de variables a partir de las existentes ----
crear_variables <- function(data, train = TRUE){
  if (train){
    data <- data %>% 
      mutate(
        price_per_surface <- price/surface_total
      )
  }
  
  data <- data %>% 
    mutate(
      bathroom_per_bedroom <- bathrooms/bedrooms,
      bathroom_per_room <- bathrooms/room,
      ratio_covered <- surface_covered/surface_total,
      bedroom_per_room <- bedrooms/room
    )
  return(data)
}

train <- crear_variables(train, TRUE)
test <- crear_variables(train, TRUE)

## ---- 05 Predictores texto ----

## ---- 06 Estadísticas Descriptivas ----

## ---- 07 Mapas ----

## ---- 08 Guardamos datos ----