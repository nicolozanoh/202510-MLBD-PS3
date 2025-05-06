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
  "osmdata",
  ""
)

## ---- 02 Cargamos datos ----

test <- read.csv(
  "stores\\test.csv"
)

train <- read.csv(
  "stores\\train.csv"
)

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


## ---- 04 Variables externas ----

# Miramos que features podemos usar para conseguir data de OSM
available_features()

# Para los features que aplican para el taller miramos los tags disponibles
print(available_tags("aeroway"), n=1000)
print(available_tags("agricultural"), n=1000)
print(available_tags("amenity"), n=1000)
print(available_tags("bicycle_road"), n=1000)
print(available_tags("building:levels"), n=1000)

print(available_tags("access"), n=1000)
print(available_tags("bus"), n=1000)

print(available_tags("bus:lanes"), n=1000)
print(available_tags("bus_bay"), n=1000)
print(available_tags("busway"), n=1000)

print(available_tags("building:flats"), n=1000)
print(available_tags("building:material"), n=1000)
print(available_tags("landuse"), n=1000)
print(available_tags("leisure"), n=1000)
print(available_tags("construction"), n=1000)
print(available_tags("construction_date"), n=1000)
print(available_tags("cycleway"), n=1000)
print(available_tags("drinking_water"), n=1000)


## ---- 05 Predictores texto ----

## ---- 06 Estadísticas Descriptivas ----

## ---- 07 Mapas ----

## ---- 08 Guardamos datos ----