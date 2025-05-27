# Taller 3 Preprocesamiento de datos:

# install and load required packages

require(pacman)
p_load(
  ggplot2,
  tidyverse,
  dplyr,
  visdat,
  sf,
  stargazer,
  leaflet,
  gridExtra,
  osmdata,
  FNN
)

# Recolección de los datos:
Train <- read.csv("stores\\raw\\train.csv")
Test <- read.csv("stores\\raw\\test.csv")
template <- read.csv("stores\\raw\\submission_template.csv")

#Cargamos también las bases de datos extraidas de texto

#ruta <- "C:\\Users\\samue\\OneDrive\\Escritorio\\Economia\\Big Data y Machine Learning\\Taller 3"

# Importar las bases .rds
data_texto_train <- readRDS("stores\\provisionales\\data_texto_train.rds")
data_texto_test  <- readRDS("stores\\provisionales\\data_texto_test.rds")

#Unificamos las bases de datos previo a la limpieza que se va a realizar

library(dplyr)

# 1. Identificar columnas nuevas (excluyendo 'property_id')
nuevas_vars_train <- setdiff(names(data_texto_train), names(Train))
nuevas_vars_test  <- setdiff(names(data_texto_test), names(Test))

View(Train)
View(data_texto_train)

View(Test)
View(data_texto_test)

# 2. Unir solo las variables que no están repetidas
Train_completo <- Train %>%
  left_join(data_texto_train[, c("property_id", nuevas_vars_train)], by = "property_id")

Test_completo <- Test %>%
  left_join(data_texto_test[, c("property_id", nuevas_vars_test)], by = "property_id")


##-----------------------------------------------------------------------------##

# organización y Pre procesamiento:

Train_completo %>%
  count(property_type)
Test_completo %>%
  count(property_type)

Train_completo <- Train_completo %>%
  mutate(
    year = as.factor(year),
    month = as.factor(month))

Test_completo <- Test_completo %>%
  mutate(
    year = as.factor(year),
    month = as.factor(month))

Train_completo <- Train_completo %>%
  dplyr::select(property_id, price, month, year,
                surface_total, surface_covered, rooms, bedrooms, bathrooms,
                property_type, lat, lon, title, description,
                PC1:PC30, num_parqueaderos)

Train_completo <- Train_completo %>%
  mutate(title = na_if(title, ""),
         description = na_if(description, ""))

vis_dat(Train_completo, warn_large_data = FALSE)

Test_completo <- Test_completo %>%
  dplyr::select(property_id, price, month, year,
                surface_total, surface_covered, rooms, bedrooms, bathrooms,
                property_type, lat, lon, title, description,
                PC1:PC30, num_parqueaderos)

Test_completo <- Test_completo %>%
  mutate(title = na_if(title, ""),
         description = na_if(description, ""))

vis_dat(Test_completo, warn_large_data = FALSE)

# Imputaciones
Train_completo  <- Train_completo %>% 
  mutate(
    rooms = ifelse(is.na(rooms), as.numeric(names(sort(table(rooms))[1])), rooms),
    bathrooms = ifelse(is.na(bathrooms), as.numeric(names(sort(table(bathrooms))[1])), bathrooms),
    bedrooms = ifelse(is.na(bedrooms), as.numeric(names(sort(table(bedrooms))[1])), bedrooms)
  )

Test_completo  <- Test_completo %>% 
  mutate(
    rooms = ifelse(is.na(rooms), as.numeric(names(sort(table(rooms))[1])), rooms),
    bathrooms = ifelse(is.na(bathrooms), as.numeric(names(sort(table(bathrooms))[1])), bathrooms),
    bedrooms = ifelse(is.na(bedrooms), as.numeric(names(sort(table(bedrooms))[1])), bedrooms)
  )

Train_completo <- Train_completo %>%
  mutate(across(c(rooms, bathrooms, bedrooms), ~ as.integer(as.character(.))))

Test_completo <- Test_completo %>%
  mutate(across(c(rooms, bathrooms, bedrooms), ~ as.integer(as.character(.))))

# Imputación lineal
linear_imput_model_covered  <- lm(surface_covered ~ property_type + rooms + bathrooms + bedrooms, data = Train_completo, na.action = na.exclude)   
linear_imput_model_total    <- lm(surface_total ~ property_type + rooms + bathrooms + bedrooms, data = Train_completo, na.action = na.exclude)

Train_completo$pred_covered <- predict(linear_imput_model_covered, newdata = Train_completo)
Train_completo$pred_total   <- predict(linear_imput_model_total, newdata = Train_completo)

# Para Test
linear_imput_model_covered  <- lm(surface_covered ~ property_type + rooms + bathrooms + bedrooms, data = Test_completo, na.action = na.exclude)   
linear_imput_model_total    <- lm(surface_total ~ property_type + rooms + bathrooms + bedrooms, data = Test_completo, na.action = na.exclude)

Test_completo$pred_covered <- predict(linear_imput_model_covered, newdata = Test_completo)
Test_completo$pred_total   <- predict(linear_imput_model_total, newdata = Test_completo)

# Sustituir valores NA por predichos
Train_completo <- Train_completo %>% 
  mutate(
    surface_covered = ifelse(is.na(surface_covered), pred_covered, surface_covered),
    surface_total   = ifelse(is.na(surface_total), pred_total, surface_total)
  )

Test_completo <- Test_completo %>% 
  mutate(
    surface_covered = ifelse(is.na(surface_covered), pred_covered, surface_covered),
    surface_total   = ifelse(is.na(surface_total), pred_total, surface_total)
  )

vis_dat(Train_completo, warn_large_data = FALSE)
vis_dat(Test_completo, warn_large_data = FALSE)

Train_completo <- Train_completo %>% select(-pred_covered, -pred_total)
Test_completo  <- Test_completo  %>% select(-pred_covered, -pred_total)

# Validaciones
stargazer(Train_completo, type = "text")
stargazer(Test_completo, type = "text")
# Hay una incosistencia valores negativos en area de superficie se vuelven 
# positivo dado que se presume un error de digitacion :
Test_completo$surface_total <- abs(Test_completo$surface_total)


# Calcular precio por metro cuadrado
Train_completo <- Train_completo %>%
  mutate(precio_por_mt2 = round(price / surface_total, 0) / 1e6)

# Outliers
low   <- round(mean(Train_completo$precio_por_mt2) - 2 * sd(Train_completo$precio_por_mt2))
up    <- round(mean(Train_completo$precio_por_mt2) + 2 * sd(Train_completo$precio_por_mt2))
perc1 <- unname(round(quantile(Train_completo$precio_por_mt2, probs = c(.01)), 2))

Graph_1 <- Train_completo %>%
  ggplot(aes(y = precio_por_mt2)) +
  geom_boxplot(fill = "darkblue", alpha = 0.4) +
  labs(title = "Muestra con valores atipicos", y = "Precio por metro cuadrado (millones)", x = "") +
  theme_bw()

Graph_2 <- Train_completo %>%
  filter(between(precio_por_mt2, perc1, up)) %>% 
  ggplot(aes(y = precio_por_mt2)) +
  geom_boxplot(fill = "darkblue", alpha = 0.4) +
  labs(title = "Muestra sin los valores atipicos", y = "Precio por metro cuadrado (millones)", x = "") +
  theme_bw()

grid.arrange(Graph_1, Graph_2, ncol = 2)

Train_completo <- Train_completo %>% filter(between(precio_por_mt2, perc1, up))

# Validación espacial
Train_completo <- Train_completo %>% filter(!is.na(lat) & !is.na(lon))
Test_completo  <- Test_completo  %>% filter(!is.na(lat) & !is.na(lon))

leaflet() %>%
  addTiles() %>%
  addCircles(lng = Train_completo$lon, lat = Train_completo$lat)

limes_Bog <- getbb("Bogota Colombia")

Train_completo <- Train_completo %>%
  filter(between(lon, limes_Bog[1, "min"], limes_Bog[1, "max"]),
         between(lat, limes_Bog[2, "min"], limes_Bog[2, "max"]))

Test_completo <- Test_completo %>%
  filter(between(lon, limes_Bog[1, "min"], limes_Bog[1, "max"]),
         between(lat, limes_Bog[2, "min"], limes_Bog[2, "max"]))

Train_completo <- Train_completo %>% filter(surface_covered > 15)
Test_completo  <- Test_completo  %>% filter(surface_covered > 15)

Train_completo <- Train_completo %>%
  mutate(precio_por_mt2_sc = (precio_por_mt2 - min(precio_por_mt2, na.rm = TRUE)) / 
           (max(precio_por_mt2, na.rm = TRUE) - min(precio_por_mt2, na.rm = TRUE)))

Train_completo <- Train_completo %>%
  mutate(color = case_when(
    property_type == "Apartamento" ~ "#EE6363",
    property_type == "Casa" ~ "#7AC5CD",
    TRUE ~ "gray"
  ))

Train_completo <- Train_completo %>%
  mutate(popup_html = paste0("<b>Precio:</b> ", scales::dollar(price),
                             "<br> <b>Área:</b> ", as.integer(surface_total), " mt2",
                             "<br> <b>Tipo de inmueble:</b> ", property_type,
                             "<br> <b>Alcobas:</b> ", as.integer(rooms),
                             "<br> <b>Baños:</b> ", as.integer(bathrooms)))

latitud_central <- mean(Train_completo$lat, na.rm = TRUE)
longitud_central <- mean(Train_completo$lon, na.rm = TRUE)

leaflet() %>%
  addTiles() %>%
  setView(lng = longitud_central, lat = latitud_central, zoom = 12) %>%
  addCircles(lng = Train_completo$lon, 
             lat = Train_completo$lat, 
             col = Train_completo$color,
             fillOpacity = 1,
             opacity = 1,
             radius = Train_completo$precio_por_mt2_sc * 10,
             popup = Train_completo$popup_html)

# Lectura de capas espaciales
localidades <- st_read(dsn = "stores\\datos espaciales\\localidades_bog", layer = "Loca")
Manzanas <- st_read(dsn = "stores\\datos espaciales\\Manzanas", layer = "ManzanaEstratificacion")

localidades_urbanas <- localidades %>%
  filter(LocNombre %in% c("USAQUEN", "CHAPINERO", "SANTA FE", "SAN CRISTOBAL",
                          "USME", "TUNJUELITO", "BOSA", "KENNEDY", "FONTIBON", 
                          "ENGATIVA", "SUBA", "BARRIOS UNIDOS", "TEUSAQUILLO", 
                          "LOS MARTIRES", "ANTONIO NARIÑO", "PUENTE ARANDA", 
                          "CANDELARIA", "RAFAEL URIBE URIBE", "CIUDAD BOLIVAR"))

ggplot() +
  geom_sf(data = localidades_urbanas, fill = "red", color = "black", alpha = 0.4) +
  geom_sf(data = Manzanas, fill = NA, color = "blue", size = 0.3) +
  labs(title = "Localidades Urbanas y Sectores Urbanos Coincidentes") +
  theme_minimal()

# Convertir a objetos sf
sf_Train <- st_as_sf(Train_completo, coords = c("lon", "lat"), crs = 4626) %>% st_transform(crs = 3116)
sf_Test  <- st_as_sf(Test_completo, coords = c("lon", "lat"), crs = 4626) %>% st_transform(crs = 3116)

localidades_urbanas <- st_transform(localidades_urbanas, crs = 3116)
Manzanas <- st_transform(Manzanas, crs = 3116)

# Definimos limites de bogota en Magnas sirgas proyectado 
pry_lim_x = c(999000, 1015000)
pry_lim_y = c(1007000, 1019000)

ggplot()+
  geom_sf(data=localidades_urbanas, color = "red") + 
  geom_sf(data=sf_Train,aes(color = precio_por_mt2) ,shape=15, size=0.3)+
  theme_bw()


# Unión espacial: agregar nombre de localidad a cada propiedad
Train_localizado <- st_join(sf_Train, localidades_urbanas)
Train_localizado <- st_join(Train_localizado, Manzanas)
#lo mismo para Test:
Test_localizado<- st_join(sf_Test, localidades_urbanas)
Test_localizado<- st_join(sf_Test, Manzanas)

# Reconozco mis key's de osmadata más relevantes para improtar de Openstreetmap:
cat(available_features(), sep = "\n")
print(available_tags("amenity"), n = Inf)
print(available_tags("public_transport"), n = Inf)
print(available_tags("highway"), n = Inf)
print(available_tags("leisure"), n = Inf)
print(available_tags("building"), n = Inf)
print(available_tags("shop"), n = Inf)

#Defino mi espacio:
bogota<-opq(bbox = getbb("Bogotá Colombia"))
bogota

# Recupero mis features de interes para mi espacio:
amenities <- bogota %>%
  add_osm_feature(key = "amenity", value = c("university", "bar", 
                                             "school", "hospital", 
                                             "restaurant", "parking", "place_of_worship", 
                                             "police","theatre","nightclub")) %>%
  osmdata_sf()

public_transport <- bogota %>%
  add_osm_feature(key = "public_transport", value = c("platform", "station")) %>%
  osmdata_sf()

highway <- bogota %>%
  add_osm_feature(key = "highway", value = c("pedestrian", "primary",
                                             "secondary","cycleway")) %>%
  osmdata_sf()

leisure <- bogota %>%
  add_osm_feature(key = "leisure", value = c("fitness_centre", "garden", "",
                                             "park", "playground")) %>%
  osmdata_sf()

building<- bogota %>%
  add_osm_feature(key = "building", value = c("supermarket")) %>%
  osmdata_sf()

shop  <- bogota %>%
  add_osm_feature(key = "shop", value = c("mall")) %>%
  osmdata_sf()


#De las features del parque nos interesa su geometría y donde estan ubicados 
amenities_geometria <- amenities$osm_polygons %>%
  dplyr::select(osm_id, name, amenity)

public_transport_geometria <- public_transport$osm_polygons %>%
  dplyr::select(osm_id, name, public_transport)

highway_geometria <- highway$osm_lines %>% 
  dplyr::select(osm_id, name,highway) 

leisure_geometria <- st_as_sf(leisure$osm_polygons) %>%
  dplyr::select(osm_id, name, leisure)

building_geometria <- st_as_sf(building$osm_polygons) %>%
  dplyr::select(osm_id, name, building)

shop_geometria <- st_as_sf(shop$osm_polygons) %>%
  dplyr::select(osm_id, name, shop)


# El area urbana de bogota es lo suficientemente plana para guiarse fielmente por los centroides:

calculo_centroides <- function(data) {
  # Calculamos los centroides
  data_centroides <- st_centroid(data, byid = TRUE)
  # Extraemos coordenadas X e Y y agregarlas como columnas
  coords <- st_coordinates(data_centroides)
  data_centroides <- data_centroides %>%
    mutate(x = coords[, "X"],
           y = coords[, "Y"])
  return(data_centroides)
}


amenities_centroides <- calculo_centroides(amenities_geometria)
public_transport_centroides <- calculo_centroides(public_transport_geometria)
building_centroides <- calculo_centroides(building_geometria)

# ARREGLAMOS GEOMETRIAS INVALIDAS 
# se tuvo un error por geometrias invalidas, inspeccionamos cuantas son:
validas <- st_is_valid(leisure_geometria)
table(validas)
# solo son 2 se decide precindir de ellas y volvemos a correr la funcion calculo centroides:
leisure_geometria <- leisure_geometria %>% filter(st_is_valid(.))
leisure_centroides <- calculo_centroides(leisure_geometria)

# se tuvo un error por geometrias invalidas, inspeccionamos cuantas son:
validas <- st_is_valid(shop_geometria)
table(validas)
# solo son 1 se decide precindir de ellas y volvemos a correr la funcion calculo centroides:
shop_geometria <- shop_geometria %>% filter(st_is_valid(.))
shop_centroides <- calculo_centroides(shop_geometria)

# Pasamos las geometrias de los centroides y lineas a Magnas sirgas proyectado:
amenities_centroides <- st_transform(amenities_centroides, crs = 3116)
public_transport_centroides <- st_transform(public_transport_centroides, crs = 3116)
leisure_centroides <- st_transform(leisure_centroides, crs = 3116)
highway_geometria <- st_transform(highway_geometria, crs = 3116)
shop_centroides <- st_transform(shop_centroides, crs = 3116)
building_centroides <- st_transform(building_centroides, crs = 3116)


# dividimos por features los vectores que agrupan mas de uno
dividir_por_categoria <- function(df) {
  # Detecta la columna categórica (excluyendo 'osm_id', 'name', 'x', 'y', 'geometry')
  cat_col <- setdiff(names(df), c("osm_id", "name", "x", "y", "geometry"))[1]
  
  if (is.null(cat_col)) stop("No se encontró columna categórica válida.")
  
  # Divide el dataframe en una lista usando esa columna
  lista <- split(df, df[[cat_col]])
  
  return(lista)
}

lista_amenities <- dividir_por_categoria(amenities_centroides)
names(lista_amenities)  
list2env(lista_amenities, envir = .GlobalEnv)

lista_public_transport <- dividir_por_categoria(public_transport_centroides)
names(lista_public_transport)  
list2env(lista_public_transport, envir = .GlobalEnv)

lista_leisure <- dividir_por_categoria(leisure_centroides)
names(lista_leisure)  
list2env(lista_leisure, envir = .GlobalEnv)

lista_highway <- dividir_por_categoria(highway_geometria)
names(lista_highway)  
list2env(lista_highway, envir = .GlobalEnv)

lista_shop <- dividir_por_categoria(shop_centroides)
names(lista_shop)  
list2env(lista_shop, envir = .GlobalEnv)

building <- dividir_por_categoria(building_centroides)
names(building)  
list2env(building, envir = .GlobalEnv)

# Creamos el mapa de Bogota con las ammenity seleccionadas:
leaflet() %>%
  addTiles() %>%
  setView(lng = longitud_central, lat = latitud_central, zoom = 12) %>%
  addPolygons(data = amenities_geometria, col = "red",weight = 10,
              opacity = 0.8, popup = amenities_geometria$name) %>%
  addCircles(lng = amenities_centroides$x, 
             lat = amenities_centroides$y, 
             col = "darkblue", opacity = 0.5, radius = 1)
# Creamos el mapa de Bogota con las public_transport seleccionadas:
leaflet() %>%
  addTiles() %>%
  setView(lng = longitud_central, lat = latitud_central, zoom = 12) %>%
  addPolygons(data = public_transport_geometria, col = "red",weight = 10,
              opacity = 0.8, popup = public_transport_geometria$name) %>%
  addCircles(lng = public_transport_centroides$x, 
             lat = public_transport_centroides$y, 
             col = "darkblue", opacity = 0.5, radius = 1)
# Creamos el mapa de Bogota con las leisure seleccionadas:
leaflet() %>%
  addTiles() %>%
  setView(lng = longitud_central, lat = latitud_central, zoom = 12) %>%
  addPolygons(data = leisure_geometria, col = "red",weight = 10,
              opacity = 0.8, popup = leisure_geometria$name) %>%
  addCircles(lng = leisure_centroides$x, 
             lat = leisure_centroides$y, 
             col = "darkblue", opacity = 0.5, radius = 1)

# Creamos el mapa de Bogota con las leisure seleccionadas:
leaflet() %>%
  addTiles() %>%
  setView(lng = longitud_central, lat = latitud_central, zoom = 12) %>%
  addPolygons(data = leisure_geometria, col = "red",weight = 10,
              opacity = 0.8, popup = leisure_geometria$name) %>%
  addCircles(lng = leisure_centroides$x, 
             lat = leisure_centroides$y, 
             col = "darkblue", opacity = 0.5, radius = 1)

ggplot() +
  geom_sf(data = localidades_urbanas,color = "black")+
  geom_sf(data = highway_geometria, color = "blue", size = 0.4) +
  labs(title = "Infraestructura de Calles y ciclorutas significativas") +
  theme_minimal()



# Creamos el mapa de Bogota con los mall:
leaflet() %>%
  addTiles() %>%
  setView(lng = longitud_central, lat = latitud_central, zoom = 12) %>%
  addPolygons(data = shop_geometria, col = "red",weight = 10,
              opacity = 0.8, popup = shop_geometria$name) %>%
  addCircles(lng = shop_centroides$x, 
             lat = shop_centroides$y, 
             col = "darkblue", opacity = 0.5, radius = 1)



# Calculamos distancias entre viviendas y features:
# Como calcularemos distancias tendremos que usar un Magna sirgas proyectado (ya todo esta en esta proyeccion).

#calculemos las distancias de cada vivienda al feuture más cercano primero
#calculando la matriz de distancias de cada propiedad a cada centroide de los features:

calculo_distancias_por_feature <- function(destino, nombre_columna = "distancia_data") {
  dist_matrix <- st_distance(x = Train_localizado, y = destino)
  dist_min <- apply(dist_matrix, 1, min)
  
  Train_localizado <- Train_localizado %>% mutate(!!nombre_columna := dist_min)
  return(Train_localizado)
}
Train_localizado <- calculo_distancias_por_feature(university, "dist_universidad")
Train_localizado <- calculo_distancias_por_feature(hospital, "dist_hospital")
Train_localizado <- calculo_distancias_por_feature(restaurant, "dist_restaurante")
Train_localizado <- calculo_distancias_por_feature(police, "dist_policia")
Train_localizado <- calculo_distancias_por_feature(school, "dist_escuela")
Train_localizado <- calculo_distancias_por_feature(place_of_worship, "dist_lugares_de_culto")
Train_localizado <- calculo_distancias_por_feature(nightclub, "dist_discoteca")
Train_localizado <- calculo_distancias_por_feature(parking, "dist_parqueadero")
Train_localizado <- calculo_distancias_por_feature(theatre, "dist_teatro")
Train_localizado <- calculo_distancias_por_feature(bar, "dist_bar")
Train_localizado <- calculo_distancias_por_feature(platform, "dist_plataformas_del_transp_publico")
Train_localizado <- calculo_distancias_por_feature(station, "dist_estaciones_del_transp_publico")
Train_localizado <- calculo_distancias_por_feature(fitness_centre, "dist_gimnasios")
Train_localizado <- calculo_distancias_por_feature(garden, "dist_jardines")
Train_localizado <- calculo_distancias_por_feature(park, "dist_parque")
Train_localizado <- calculo_distancias_por_feature(playground, "dist_zona_juegos_inft")
Train_localizado <- calculo_distancias_por_feature(primary, "dist_via_primary")
Train_localizado <- calculo_distancias_por_feature(secondary, "dist_via_secondary")
Train_localizado <- calculo_distancias_por_feature(pedestrian, "dist_via_pedestrian")
Train_localizado <- calculo_distancias_por_feature(cycleway, "dist_cycleway")
Train_localizado <- calculo_distancias_por_feature(mall, "dist_mall")


# LO MISMO PARA TEST
calculo_distancias_Test_localizado_por_feature_ <- function(destino, nombre_columna = "distancia_data") {
  dist_matrix <- st_distance(x = Test_localizado, y = destino)
  dist_min <- apply(dist_matrix, 1, min)
  
  Test_localizado <- Test_localizado %>% mutate(!!nombre_columna := dist_min)
  return(Test_localizado)
}
Test_localizado <- calculo_distancias_Test_localizado_por_feature_(university, "dist_universidad")
Test_localizado <- calculo_distancias_Test_localizado_por_feature_(hospital, "dist_hospital")
Test_localizado <- calculo_distancias_Test_localizado_por_feature_(restaurant, "dist_restaurante")
Test_localizado <- calculo_distancias_Test_localizado_por_feature_(police, "dist_policia")
Test_localizado <- calculo_distancias_Test_localizado_por_feature_(school, "dist_escuela")
Test_localizado <- calculo_distancias_Test_localizado_por_feature_(place_of_worship, "dist_lugares_de_culto")
Test_localizado <- calculo_distancias_Test_localizado_por_feature_(nightclub, "dist_discoteca")
Test_localizado <- calculo_distancias_Test_localizado_por_feature_(parking, "dist_parqueadero")
Test_localizado <- calculo_distancias_Test_localizado_por_feature_(theatre, "dist_teatro")
Test_localizado <- calculo_distancias_Test_localizado_por_feature_(bar, "dist_bar")
Test_localizado <- calculo_distancias_Test_localizado_por_feature_(platform, "dist_plataformas_del_transp_publico")
Test_localizado <- calculo_distancias_Test_localizado_por_feature_(station, "dist_estaciones_del_transp_publico")
Test_localizado <- calculo_distancias_Test_localizado_por_feature_(fitness_centre, "dist_gimnasios")
Test_localizado <- calculo_distancias_Test_localizado_por_feature_(garden, "dist_jardines")
Test_localizado <- calculo_distancias_Test_localizado_por_feature_(park, "dist_parque")
Test_localizado <- calculo_distancias_Test_localizado_por_feature_(playground, "dist_zona_juegos_inft")
Test_localizado <- calculo_distancias_Test_localizado_por_feature_(primary, "dist_via_primary")
Test_localizado <- calculo_distancias_Test_localizado_por_feature_(secondary, "dist_via_secondary")
Test_localizado <- calculo_distancias_Test_localizado_por_feature_(pedestrian, "dist_via_pedestrian")
Test_localizado <- calculo_distancias_Test_localizado_por_feature_(cycleway, "dist_cycleway")
Test_localizado <- calculo_distancias_Test_localizado_por_feature_(mall, "dist_mall")


# Procedemos a crear variables a partir de la informacion de la vivienda:
crear_variables_base_vivienda <- function(data){
  data <- data %>% 
    mutate(
      bathroom_per_bedroom = bathrooms/bedrooms,
      bathroom_per_room=bathrooms/rooms,
      ratio_covered=surface_covered/surface_total,
      bedroom_per_room=bedrooms/rooms
    )
  return(data)
}

Train_localizado <- crear_variables_base_vivienda(Train_localizado)
Test_localizado <- crear_variables_base_vivienda(Test_localizado)


names(Train_localizado
)

names(Test_localizado)

names(Train_localizado) <- c(
  "property_id", "price", "month", "year", "surf_total", "surf_cov", "rooms", "bdrms", "bathrm", "prop_typ",
  "title", "descript","PC1","PC2","PC3","PC4","PC5","PC6",
  "PC7","PC8","PC9","PC10","PC11","PC12","PC13","PC14","PC15","PC16","PC17","PC18","PC19","PC20","PC21","PC22","PC23",
  "PC24","PC25","PC26","PC27","PC28","PC29","PC30","num_parqueaderos","prec_mt2", "prec_mt2sc", "color", "popuphtml", "loc_nomb", "loc_admin", "loc_area",
  "loc_cod", "shp_leng", "shp_area", "obj_id", "cod_man", "estrato", "cod_zon", "cod_cri", "normatv",
  "acto_admin", "num_act", "fecha_act", "escala_cp", "fecha_cap", "responsab", "shp_area2", "shp_len", 
  "geometry", "dist_univ", "dist_hosp", "dist_rest", "dist_pol", "dist_esc", "dist_culto", "dist_disco", 
  "dist_parq", "dist_teatr", "dist_bar", "dist_plat", "dist_estac", "dist_gym", "dist_jard", "dist_park", 
  "dist_juego", "dist_vprim", "dist_vsec", "dist_vped", "dist_cycl", "dist_mall", 
  "bath_perb", "bath_perr", "ratio_cov", "bdrm_perr"
)

# Renombrar columnas de Test_localizado
names(Test_localizado) <- c(
  "property_id", "price", "month", "year", "surf_total", "surf_cov", "rooms", "bdrms", "bathrm", "prop_typ", 
  "title", "descript","PC1","PC2","PC3","PC4","PC5","PC6",
  "PC7","PC8","PC9","PC10","PC11","PC12","PC13","PC14","PC15","PC16","PC17","PC18","PC19","PC20","PC21","PC22","PC23",
  "PC24","PC25","PC26","PC27","PC28","PC29","PC30","num_parqueaderos","obj_id", "cod_man", "estrato", "cod_zon", "cod_cri", "normatv", "acto_admin", "num_act", 
  "fecha_act", "escala_cp", "fecha_cap", "responsab", "shp_area", "shp_len", "geometry", "dist_univ", "dist_hosp", 
  "dist_rest", "dist_pol", "dist_esc", "dist_culto", "dist_disco", "dist_parq", "dist_teatr", "dist_bar", 
  "dist_plat", "dist_estac", "dist_gym", "dist_jard", "dist_park", "dist_juego", "dist_vprim", "dist_vsec", "dist_vped", 
  "dist_cycl", "dist_mall", "bath_perb", "bath_perr", "ratio_cov", "bdrm_perr")



Train_localizado <- Train_localizado %>%
  left_join(Train %>% select(property_id, lat, lon), by = "property_id")

Test_localizado <- Test_localizado %>%
  left_join(Test %>% select(property_id, lat, lon), by = "property_id")


# Hacemos particion de Chapinero 
#Train_localizado <- Train_localizado %>% subset( loc_nomb != 'CHAPINERO' | is.na(loc_nomb)==TRUE )

# Imputamos bth_prb valores inf por el minimo de bath_perb
# Encontrar el mínimo valor finito en bath_perb
min_finite <- min(Train_localizado$bath_perb[is.finite(Train_localizado$bath_perb)], na.rm = TRUE)

# Reemplazar valores Inf con el mínimo finito
Train_localizado <- Train_localizado %>%
  mutate(bth_prb = ifelse(is.infinite(bath_perb), min_finite, bath_perb))




########## Imputacion de variables clave #################################################

# Función para calcular la moda
moda <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}
########## Imputacion estrato #################################################
# Separar los puntos con y sin estrato
con_estrato <- Train_localizado %>% filter(!is.na(estrato))
sin_estrato <- Train_localizado %>% filter(is.na(estrato))

con_estrato_TEST <- Test_localizado %>% filter(!is.na(estrato))
sin_estrato_TEST <- Test_localizado %>% filter(is.na(estrato))

# Extraer coordenadas en formato matriz para kNN
coords_con <- st_coordinates(st_centroid(con_estrato))
coords_sin <- st_coordinates(st_centroid(sin_estrato))

coords_con_TEST <- st_coordinates(st_centroid(con_estrato_TEST))
coords_sin_TEST <- st_coordinates(st_centroid(sin_estrato_TEST))

# Buscar los k vecinos más cercanos (aquí usamos k = 5)
knn_result_TRAIN <- get.knnx(coords_con, coords_sin, k = 5)
knn_result_TEST <- get.knnx(coords_con_TEST, coords_sin_TEST, k = 5)

# Imputar estrato usando moda de los vecinos
estrato_imputado_TRAIN <- apply(knn_result_TRAIN$nn.index, 1, function(indices) {
  vecinos <- con_estrato$estrato[indices]
  moda(vecinos)
})

estrato_imputado_TEST <- apply(knn_result_TEST$nn.index, 1, function(indices) {
  vecinos <- con_estrato_TEST$estrato[indices]
  moda(vecinos)
})
# Asignar los estratos imputados
sin_estrato$estrato <- estrato_imputado_TRAIN
sin_estrato_TEST$estrato <- estrato_imputado_TEST
# Combinar todo
Train_localizado <- bind_rows(con_estrato, sin_estrato)
Test_localizado <- bind_rows(con_estrato_TEST, sin_estrato_TEST)

########## Imputacion Numero de parqueaderos #################################################
# Separar los puntos con y sin estrato
con_nm_prqd  <- Train_localizado %>% filter(!is.na(num_parqueaderos ))
sin_nm_prqd  <- Train_localizado %>% filter(is.na(num_parqueaderos ))

con_nm_prqd_TEST  <- Test_localizado %>% filter(!is.na(num_parqueaderos ))
sin_nm_prqd_TEST  <- Test_localizado %>% filter(is.na(num_parqueaderos ))
# Extraer coordenadas en formato matriz para kNN
coords_con_nm_prqd <- st_coordinates(st_centroid(con_nm_prqd))
coords_sin_nm_prqd <- st_coordinates(st_centroid(sin_nm_prqd))

coords_con_nm_prqd_TEST <- st_coordinates(st_centroid(con_nm_prqd_TEST))
coords_sin_nm_prqd_TEST <- st_coordinates(st_centroid(sin_nm_prqd_TEST))
# Buscar los k vecinos más cercanos (aquí usamos k = 5)
knn_result <- get.knnx(coords_con_nm_prqd, coords_sin_nm_prqd, k = 5)
knn_result_Test <- get.knnx(coords_con_nm_prqd_TEST, coords_sin_nm_prqd_TEST, k = 5)

# Imputar estrato usando moda de los vecinos
nm_prqd_imputado <- apply(knn_result$nn.index, 1, function(indices) {
  vecinos <- con_nm_prqd$num_parqueaderos [indices]
  moda(vecinos)
})

nm_prqd_imputado_Test <- apply(knn_result_Test$nn.index, 1, function(indices) {
  vecinos <- con_nm_prqd_TEST$num_parqueaderos [indices]
  moda(vecinos)
})
# Asignar los nm_prqd imputados
sin_nm_prqd$num_parqueaderos  <- nm_prqd_imputado
sin_nm_prqd_TEST$num_parqueaderos  <- nm_prqd_imputado_Test
# Combinar todo
Train_localizado <- bind_rows(con_nm_prqd, sin_nm_prqd)
Test_localizado <- bind_rows(con_nm_prqd_TEST, sin_nm_prqd_TEST)

########## Imputacion Numero de codigo de manzana #################################################
# Separar los puntos con y sin estrato
con_cod_man    <- Train_localizado %>% filter(!is.na(cod_man))
sin_cod_man    <- Train_localizado %>% filter(is.na(cod_man))

con_cod_man_TEST  <- Test_localizado %>% filter(!is.na(cod_man))
sin_cod_man_TEST  <- Test_localizado %>% filter(is.na(cod_man ))
# Extraer coordenadas en formato matriz para kNN
coords_con_cod_man <- st_coordinates(st_centroid(con_cod_man))
coords_sin_cod_man <- st_coordinates(st_centroid(sin_cod_man))

coords_con_cod_man_TEST <- st_coordinates(st_centroid(con_cod_man_TEST))
coords_sin_cod_man_TEST <- st_coordinates(st_centroid(sin_cod_man_TEST))
# Buscar los k vecinos más cercanos (aquí usamos k = 5)
knn_result <- get.knnx(coords_con_cod_man, coords_sin_cod_man, k = 5)
knn_result_Test <- get.knnx(coords_con_cod_man_TEST, coords_sin_cod_man_TEST, k = 5)

# Imputar estrato usando moda de los vecinos
cod_man_imputado <- apply(knn_result$nn.index, 1, function(indices) {
  vecinos <- con_cod_man$cod_man [indices]
  moda(vecinos)
})

cod_man_imputado_Test <- apply(knn_result_Test$nn.index, 1, function(indices) {
  vecinos <- con_cod_man_TEST$cod_man [indices]
  moda(vecinos)
})
# Asignar los nm_prqd imputados
sin_cod_man$cod_man  <- cod_man_imputado
sin_cod_man_TEST$cod_man  <- cod_man_imputado_Test
# Combinar todo
Train_localizado <- bind_rows(con_cod_man, sin_cod_man)
Test_localizado <- bind_rows(con_cod_man_TEST, sin_cod_man_TEST)



# Guardamos en un archivo shapefield:
# para el training set:
st_write(Train_localizado, "stores\\work\\Train\\Train.shp")

#para el testing set:
st_write(Test_localizado, "stores\\work\\Test\\Test.shp")



############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################


## RUTAS
"stores/work_jcp/Datasets/0/train.csv"
"stores/work_jcp/Datasets/0/test.csv"

"stores/work_jcp/Datasets/1/Train/Train.shp"
"stores/work_jcp/Datasets/1/Test/Test.shp"

"stores/work_jcp/Datasets/1/Train/Train_localizado.shp"
"stores/work_jcp/Datasets/1/Test/Test_localizado.shp"


#======================================================================
#  ANALÍTICA BÁSICA DE LOS CSV ORIGINALES
#  ▸ Compara esquemas   (Train.csv vs Test.csv)
#  ▸ Tabla de descriptivos por variable (x2)
#  ▸ Resumen global de cada dataset     (x2)
#  ▸ Exporta   ▸ CSV   ▸ PNG   para insertar en LaTeX
#----------------------------------------------------------------------
#  Ejecutar en VS Code — requiere PhantomJS o Chrome para webshot2
#======================================================================

# 0. Paquetes ----------------------------------------------------------
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
  readr, dplyr, purrr, tibble, tidyr,
  skimr,                         # descriptivos
  gt, webshot2,                  # tablas → PNG
  janitor, stringr
)

# 1. Rutas -------------------------------------------------------------
csv_train <- "stores/work_jcp/Datasets/0/train.csv"
csv_test  <- "stores/work_jcp/Datasets/0/test.csv"

# 2. Lectura -----------------------------------------------------------
train <- read_csv(csv_train, show_col_types = FALSE)
test  <- read_csv(csv_test, show_col_types = FALSE)

out_dir <- "stores/work_jcp/metadata_csv"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

#======================================================================
#  Comparación de variables (nombres y tipos) -----------------------
#======================================================================

v_train   <- names(train)
v_test    <- names(test)
solo_tr   <- setdiff(v_train, v_test)
solo_te   <- setdiff(v_test , v_train)
comunes   <- intersect(v_train, v_test)

# ---- tipos en las columnas comunes ----
tipo_tbl <- tibble(
  variable   = comunes,
  train_type = map_chr(comunes, ~ paste(class(train[[.x]])[1])),
  test_type  = map_chr(comunes, ~ paste(class(test [[.x]])[1])),
  iguales    = train_type == test_type
)

# guardar resultado
write_csv(tipo_tbl, file.path(out_dir, "schema_comparison.csv"))




#----------------------------------------------------------
#  Función helper: gtsave_png()  – ajusta el alto dinámico
#----------------------------------------------------------
gtsave_png <- function(gt_tbl, file_png, n_rows, base_height = 120,
                       row_height = 20, zoom = 2){
  # vheight = base + filas × alto_fila   (px)
  vheight <- base_height + n_rows * row_height
  
  gt::gtsave(
    data = gt_tbl,
    filename = file_png,
    vheight  = vheight,   # alto del viewport
    zoom     = zoom,      # mejora la resolución (2 = 200 %)
    expand   = 5          # agrega margen para que no corte bordes
  )
}






#----------------------------------------------------------
#  A) Descriptivos por variable  (Train / Test)
#----------------------------------------------------------
crear_desc <- function(df, nombre){
  desc <- skimr::skim_without_charts(df)
  write_csv(desc, file.path(out_dir, paste0("desc_", nombre, ".csv")))
  
  png_file <- file.path(out_dir, paste0("desc_", nombre, ".png"))
  n_rows   <- nrow(desc)
  
  gt_tbl <- desc %>%
    gt::gt(rowname_col = "skim_variable") %>%
    gt::tab_header(title = paste("Descriptivos —", str_to_title(nombre))) %>%
    gt::opt_table_font(size = 9)                # letra más compacta
  
  gtsave_png(gt_tbl, png_file, n_rows)
  invisible(desc)
}

desc_train <- crear_desc(train, "train")
desc_test  <- crear_desc(test , "test" )



#----------------------------------------------------------
#  B) Resumen global  (apenas 2 filas, pero se aplica igual)
#----------------------------------------------------------
sum_png <- file.path(out_dir, "summary_datasets.png")

gt_sum <- sum_tbl %>%
  gt() %>%
  gt::tab_header(title = "Resumen global de los CSV originales") %>%
  gt::fmt_number(columns = perc_missing, decimals = 2, suffixing = "%")

gtsave_png(gt_sum, sum_png, n_rows = nrow(sum_tbl), base_height = 80)







#======================================================================
#  C. Resumen global de cada CSV --------------------------------------
#======================================================================

resumen <- function(df, nombre){
  tibble(
    dataset      = str_to_title(nombre),
    n_obs        = nrow(df),
    n_vars       = ncol(df),
    perc_missing = round(mean(is.na(df))*100, 2),
    n_numeric    = sum(map_lgl(df, is.numeric)),
    n_character  = sum(map_lgl(df, is.character)),
    obj_size_MB  = format(object.size(df), units = "MB")
  )
}

sum_tbl <- bind_rows(
  resumen(train, "train"),
  resumen(test , "test")
)

write_csv(sum_tbl, file.path(out_dir, "summary_datasets.csv"))

# tabla bonita PNG
sum_png <- file.path(out_dir, "summary_datasets.png")
sum_tbl %>%
  gt() %>%
  gt::tab_header(title = "Resumen global de los CSV originales") %>%
  gt::fmt_number(columns = perc_missing, decimals = 2, suffixing = "%") %>%
  gt::gtsave(sum_png)

#======================================================================
#  D. Mensaje final ----------------------------------------------------
#======================================================================
cat("\n► Resultados guardados en:", normalizePath(out_dir), "\n")
cat("  - schema_comparison.csv",
    "\n  - desc_train.csv / desc_test.csv + PNG",
    "\n  - summary_datasets.csv  + PNG\n")

# OPTIONAL: imprime las discrepancias en consola ----------------------
if (length(solo_tr) > 0 | length(solo_te) > 0) {
  cat("\n✘ Variables exclusivas de TRAIN (", length(solo_tr), "):\n", sep = "")
  print(solo_tr, quote = FALSE)
  cat("\n✘ Variables exclusivas de TEST  (", length(solo_te), "):\n", sep = "")
  print(solo_te, quote = FALSE)
} else {
  cat("\n✔ Ambos CSV contienen el mismo conjunto de columnas.\n")
}

if (!all(tipo_tbl$iguales)) {
  cat("\n⚠ Columnas con tipo distinto:\n")
  print(filter(tipo_tbl, !iguales), n = Inf)
} else {
  cat("\n✔ Los tipos de dato coinciden para todas las columnas comunes.\n")
}







### OTRA VERSION PARA CONSIDERAR EL TAMAÑO DE LA TABLA EN PNG
#################################################################

#======================================================================
#  Funciones para (i) reducir alto por fila   y/o
#                     (ii) dividir la tabla en NUMERIC vs NO-NUMERIC
#======================================================================

# A.  gtsave_png():  captura GT → PNG con alto dinámico ----------------
gtsave_png <- function(gt_tbl, file_png,
                       n_rows,
                       base_height = 100,   # cabecera
                       row_height  = 18,    # px por fila  (⇐ ajústalo aquí)
                       zoom        = 2,
                       expand      = 5) {
  vheight <- base_height + n_rows * row_height
  gt::gtsave(data = gt_tbl, filename = file_png,
             vheight = vheight, zoom = zoom, expand = expand)
}

# B.  crear_desc_split():  divide en num / no-num si la tabla es grande
#     • Si n_rows > umbral → genera dos tablas y PNGs
#     • Caso contrario → genera una sola (igual que antes)
#----------------------------------------------------------------------
crear_desc_split <- function(df, nombre, umbral = 45,
                             row_height = 18) {
  
  desc <- skimr::skim_without_charts(df)
  
  # --- si la tabla es pequeña, se guarda entera ---
  if (nrow(desc) <= umbral) {
    png_file <- file.path(out_dir, sprintf("desc_%s.png", nombre))
    
    gtsave_png(
      gt_tbl   = gt::gt(desc, rowname_col = "skim_variable") %>%
                 gt::tab_header(title = paste("Descriptivos —", str_to_title(nombre))) %>%
                 gt::opt_table_font(size = 9),
      file_png = png_file,
      n_rows   = nrow(desc),
      row_height = row_height
    )
    write_csv(desc, file.path(out_dir, sprintf("desc_%s.csv", nombre)))
    return(invisible(list(full = desc)))
  }
  
  # --- tabla grande: separar numeric vs resto ------------------------
  desc_num  <- dplyr::filter(desc, skim_type == "numeric")
  desc_otros<- dplyr::filter(desc, skim_type != "numeric")
  
  # --- helper interno para guardar cada parte -----------------------
  save_part <- function(tbl, sufijo){
    png_file <- file.path(out_dir, sprintf("desc_%s_%s.png", nombre, sufijo))
    csv_file <- file.path(out_dir, sprintf("desc_%s_%s.csv", nombre, sufijo))
    
    gtsave_png(
      gt_tbl   = gt::gt(tbl, rowname_col = "skim_variable") %>%
                 gt::tab_header(title = paste("Descriptivos —", str_to_title(nombre), sufijo)) %>%
                 gt::opt_table_font(size = 9),
      file_png = png_file,
      n_rows   = nrow(tbl),
      row_height = row_height
    )
    write_csv(tbl, csv_file)
  }
  
  save_part(desc_num , "numeric")
  save_part(desc_otros, "otros")
  
  invisible(list(numeric = desc_num, otros = desc_otros))
}

#======================================================================
#  USO con Train / Test  (ajusta 'row_height' si lo quieres más compacto)
#======================================================================

desc_train <- crear_desc_split(train, "train", umbral = 45, row_height = 16)
desc_test  <- crear_desc_split(test , "test",  umbral = 45, row_height = 16)













##############################################################
### VERSION QUE EVALUA FILAS Y COLUMNAS MULTIPLES
##############################################################


#======================================================================
#  Diccionario en bloques cuando el nº de variables es muy grande
#======================================================================

pacman::p_load(readr, dplyr, tibble, skimr, gt, webshot2, stringr)

out_dir <- "stores/work_jcp/metadata_csv"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# --- helper para escoger columnas clave segun tipo -------------------
skim_reducido <- function(df){
  skimr::skim_without_charts(df) %>%             # tabla larga
    # numeric.*  | character.*  | Date.* | etc.
    select(
      skim_variable, skim_type, n_missing,
      tidyselect::matches("(numeric|character)\\.(mean|sd|p0|p50|p100)$"),
      tidyselect::matches("character\\.n_unique$")
    )
}

# --- divide en chunks de N filas -------------------------------------
partir_en_bloques <- function(tbl, chunk_size = 40){
  split(tbl, ceiling(seq_len(nrow(tbl)) / chunk_size))
}

# --- guarda CSV + PNG para cada bloque --------------------------------
guardar_bloque <- function(tbl, nombre, idx, row_height = 18){
  
  sufijo   <- sprintf("%02d", idx)          # 01, 02, …
  basefile <- paste0("dict_", nombre, "_", sufijo)
  
  csv_file <- file.path(out_dir, paste0(basefile, ".csv"))
  png_file <- file.path(out_dir, paste0(basefile, ".png"))
  
  write_csv(tbl, csv_file)
  
  n_rows <- nrow(tbl)
  vheight <- 90 + n_rows * row_height       # alto dinámico
  
  gt(tbl, rowname_col = "skim_variable") %>%
    tab_header(title = paste("Descriptivos —", str_to_title(nombre),
                             "(bloque", sufijo, ")")) %>%
    opt_table_font(size = 8) %>%            # letra compacta
    gtsave(png_file, vheight = vheight, zoom = 2, expand = 5)
}

# ======================================================================
#  FUNCIÓN PRINCIPAL  ---------------------------------------------------
# ======================================================================
crear_diccionario_bloques <- function(df, nombre,
                                      chunk_size = 40,
                                      row_height = 18) {
  
  dict <- skim_reducido(df)                   # tabla slim
  bloques <- partir_en_bloques(dict, chunk_size)
  
  # usamos map2(): .x = bloque, .y = índice 1..N  ---------------------
  purrr::map2(
    bloques,
    seq_along(bloques),                      # ← índice numérico
    ~ guardar_bloque(.x, nombre, .y, row_height)
  )
  
  invisible(dict)
}

# --------------------  USO  -------------------------------------------
train <- readr::read_csv("stores/work_jcp/Datasets/0/train.csv",
                         show_col_types = FALSE)
test  <- readr::read_csv("stores/work_jcp/Datasets/0/test.csv",
                         show_col_types = FALSE)

crear_diccionario_bloques(train, "train", chunk_size = 40, row_height = 16)
crear_diccionario_bloques(test , "test",  chunk_size = 40, row_height = 16)

cat(
  "\n✔ PNG y CSV generados en:",
  normalizePath(out_dir),
  "\n  - dict_train_01.png / .csv,  dict_train_02.png, …",
  "\n  - dict_test_01.png  / .csv,  dict_test_02.png, …\n"
)








































































# ================================================================
#  Listado definitivo de variables en las bases ya pre-procesadas
#  (Train_localizado y Test_localizado) – script para VS Code
# ================================================================

# 1.  Librerías ----------------------------------------------------
pacman::p_load(sf, dplyr, readr, tibble, janitor)

# 2.  Rutas a los shapefiles finales ------------------------------
path_train <- "stores/work/Train/Train.shp"
path_test  <- "stores/work/Test/Test.shp"

# 3.  Lectura (silenciosa) y eliminación de la geometría ----------
train_sf <- st_read(path_train, quiet = TRUE)
test_sf  <- st_read(path_test , quiet = TRUE)

train_tbl <- st_drop_geometry(train_sf)
test_tbl  <- st_drop_geometry(test_sf)

# 4.  Tabla con variable + tipo de dato ---------------------------
var_train <- tibble(
  variable = names(train_tbl),
  tipo     = sapply(train_tbl, function(x) paste(class(x), collapse = "/"))
)

var_test <- tibble(
  variable = names(test_tbl),
  tipo     = sapply(test_tbl, function(x) paste(class(x), collapse = "/"))
)

# 5.  Limpieza de nombres (opcional) ------------------------------
var_train <- var_train %>% clean_names()
var_test  <- var_test  %>% clean_names()

# 6.  Resultados en consola ---------------------------------------
cat("\n► Variables en TRAIN (", nrow(var_train), "):\n", sep = "")
print(var_train, n = Inf)

cat("\n► Variables en TEST (",  nrow(var_test), "):\n", sep = "")
print(var_test,  n = Inf)

# 7.  Exportar a CSV (por si se requiere en LaTeX) -----------------
dir.create("/stores/work_jcp/metadata", showWarnings = FALSE, recursive = TRUE)

write_csv(var_train, "/stores/work_jcp/metadata/variables_train.csv")
write_csv(var_test,  "/stores/work_jcp/metadata/variables_test.csv")







# ================================================================
#  Verificar que TRAIN.shp y TEST.shp contengan las mismas variables
# ================================================================
#  • Lee ambos shapefiles
#  • Compara nombre y tipo de dato por columna
#  • Reporta diferencias (si las hay)
#  Uso: ejecutar en VS Code con la carpeta 'stores/work' ya creada
# ================================================================

pacman::p_load(sf, dplyr, purrr, tibble)

# 1.  Rutas --------------------------------------------------------
shp_train <- "stores/work/Train/Train.shp"
shp_test  <- "stores/work/Test/Test.shp"

# 2.  Lectura silenciosa y sin geometría --------------------------
train <- st_read(shp_train, quiet = TRUE) %>% st_drop_geometry()
test  <- st_read(shp_test, quiet = TRUE) %>% st_drop_geometry()

# 3.  Conjuntos de nombres ---------------------------------------
v_train <- names(train)
v_test  <- names(test)

# 4.  Diferencias de nombres -------------------------------------
solo_en_train <- setdiff(v_train, v_test)
solo_en_test  <- setdiff(v_test , v_train)
comunes       <- intersect(v_train, v_test)

cat("\n──────── Comparación de nombres ────────\n")
cat("Total TRAIN:", length(v_train), " —  Total TEST:", length(v_test), "\n")

if (length(solo_en_train) == 0 && length(solo_en_test) == 0) {
  cat("✔ Las dos bases tienen exactamente las mismas columnas.\n")
} else {
  cat("⚠ Columnas SOLO en TRAIN (", length(solo_en_train), "):\n", sep = "")
  print(solo_en_train, quote = FALSE)
  cat("\n⚠ Columnas SOLO en TEST  (", length(solo_en_test), "):\n", sep = "")
  print(solo_en_test , quote = FALSE)
}

# 5.  Concordancia de tipos para columnas comunes -----------------
tipo_tbl <- tibble(
  variable = comunes,
  train_type = map_chr(comunes, ~ paste(class(train[[.x]]), collapse = "/")),
  test_type  = map_chr(comunes, ~ paste(class(test [[.x]]), collapse = "/"))
) %>%
  mutate(iguales = train_type == test_type)

cat("\n──────── Concordancia de tipos ────────\n")
if (all(tipo_tbl$iguales)) {
  cat("✔ Todos los tipos de dato coinciden entre TRAIN y TEST.\n")
} else {
  cat("⚠ Columnas con tipos distintos:\n")
  print(filter(tipo_tbl, !iguales), n = Inf)
}

# 6.  Resultado lógico (útil en scripts) --------------------------
mismos_nombres <- length(solo_en_train) == 0 && length(solo_en_test) == 0
mismos_tipos   <- all(tipo_tbl$iguales)

identico_esquema <- mismos_nombres && mismos_tipos
cat("\n► Identidad completa de esquema:", identico_esquema, "\n")












#======================================================================
#  Descriptivos detallados y resumen global de TRAIN.shp / TEST.shp
#  ▸ genera:  CSV + imagen (.png) de cada tabla (para insertar en LaTeX)
#  ▸ pensado para ejecutarse en VS Code (Windows/Mac/Linux)
#======================================================================

# 0. Paquetes ----------------------------------------------------------
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
  sf, dplyr, purrr, readr, tidyr,
  skimr,                      # estadísticos descriptivos
  gt, webshot2,               # tablas bonitas + exportar a PNG
  janitor, stringr
)

# 1. Rutas a los shapefiles finales -----------------------------------
shp_train <- "stores/work_jcp/Datasets/1/Train/Train_localizado.shp"
shp_test  <- "stores/work_jcp/Datasets/1/Test/Test_localizado.shp"

# 2. Lectura (sin geometría) ------------------------------------------
train <- st_read(shp_train, quiet = TRUE) %>% st_drop_geometry()
test  <- st_read(shp_test , quiet = TRUE) %>% st_drop_geometry()

# carpeta de salida
out_dir <- "stores/work_jcp/metadata"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

#======================================================================
#  (1)  Estadísticos descriptivos completos  --------------------------
#======================================================================

crear_desc <- function(df, nombre){
  
  # --- skimr: sin histograma spark, compacta ---
  desc_tbl <- skimr::skim_without_charts(df) %>%
    select(-starts_with("complete"))  # tabla más compacta
  
  # --- guardar CSV ---
  write_csv(desc_tbl, file.path(out_dir, paste0("desc_localizado", nombre, ".csv")))
  
  # --- convertir a tabla gt y exportar PNG ---
  desc_png <- file.path(out_dir, paste0("desc_localizado", nombre, ".png"))
  desc_tbl %>%
    gt::gt(rowname_col = "skim_variable") %>%
    gt::tab_header(title = paste("Estadísticos descriptivos —", str_to_title(nombre))) %>%
    gt::gtsave(desc_png)
  
  invisible(desc_tbl)
}

desc_train <- crear_desc(train, "train")
desc_test  <- crear_desc(test , "test" )

#======================================================================
#  (2)  Resumen global de cada dataset  -------------------------------
#======================================================================

extraer_resumen <- function(df, nombre){
  tibble(
    dataset       = str_to_title(nombre),
    n_observ      = nrow(df),
    n_variables   = ncol(df),
    perc_missing  = round(mean(is.na(df))*100, 2),
    n_numeric     = sum(map_lgl(df, is.numeric)),
    n_factor      = sum(map_lgl(df, is.factor)),
    obj_size_MB   = format(object.size(df), units = "MB")
  )
}

summary_tbl <- bind_rows(
  extraer_resumen(train, "train"),
  extraer_resumen(test , "test")
)

# --- guardar CSV y PNG ---
write_csv(summary_tbl, file.path(out_dir, "summary_datasets_localizado.csv"))

summary_png <- file.path(out_dir, "summary_datasets_localizado.png")
summary_tbl %>%
  gt() %>%
  gt::tab_header(title = "Resumen global de los conjuntos de datos") %>%
  gt::fmt_number(columns = perc_missing, decimals = 2, suffixing = "%") %>%
  gt::gtsave(summary_png)

#======================================================================
message("Tablas generadas en: ", normalizePath(out_dir))
