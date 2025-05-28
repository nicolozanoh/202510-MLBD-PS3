# Taller 3 Preprocesamiento de datos:

# install and load required packages
Packages <- c("tidyverse", 
              "ggplot2", 
              "pacman", 
              "dplyr",
              "haven",
              "boot",
              "broom",
              "lmtest", 
              "fixest", 
              "gridExtra", 
              "writexl", 
              "readxl",
              "glmnet",
              "VIM",
              "caret", 
              "MLmetrics",
              "Metrics",
              "pROC",
              "rpart",
              "rpart.plot",
              "ranger",
              "randomForest",
              "parallel",
              "doParallel",
              "adabag",
              "themis",
              "rattle",
              "stargazer",
              "sf",
              "tmaptools",
              "osmdata",
              "visdat",
              "leaflet")

invisible(lapply(Packages, function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)){ 
    install.packages(pkg)}
  library(pkg, character.only = TRUE)}))




# Recolección de los datos:
# Train <- read.csv("C:\\Users\\samue\\OneDrive\\Escritorio\\Economia\\Big Data y Machine Learning\\Taller 3\\train.csv")
# Test <- read.csv("C:\\Users\\samue\\OneDrive\\Escritorio\\Economia\\Big Data y Machine Learning\\Taller 3\\test.csv")
# Template <- read.csv("C:\\Users\\samue\\OneDrive\\Escritorio\\Economia\\Big Data y Machine Learning\\Taller 3\\submission_template.csv")

Train <- read.csv("stores\\raw\\train.csv")
Test <- read.csv("stores\\raw\\test.csv")
Template <- read.csv("stores\\raw\\submission_template.csv")


##-----------------------------------------------------------------------------##

# organización y Pre procesamiento:

Train %>%
  count(property_type)
Test%>%
  count(property_type)
# Hay un desbalance en el test hacia apartamentos.

# Redefinimos como no numericas las variables de años y meses:
Train <- Train %>%
  mutate(
    year = as.factor(year),
    month = as.factor(month))
Test <- Test %>%
  mutate(
    year = as.factor(year),
    month = as.factor(month))

# Inspeccionamos los valores faltantes:
Train<- Train %>% dplyr:: select(property_id, price, month, year,
  surface_total, surface_covered, rooms, bedrooms, bathrooms,
  property_type, lat, lon, title, description)

Train <- Train %>%
  mutate(title = na_if(title, "")) %>%
  mutate( description= na_if(description, ""))

vis_dat(Train)


Test<- Test %>% dplyr:: select(property_id, price, month, year,
                                 surface_total, surface_covered, rooms, bedrooms, bathrooms,
                                 property_type, lat, lon, title, description)

Test <- Test %>%
  mutate(title = na_if(title, "")) %>%
  mutate( description= na_if(description, ""))

vis_dat(Test)

# Existe un serio caso de valores faltantes en surface_covered & surface_total
# Un menor numero para rooms & bathrooms.

# Vamos a imputar valores para el número de habitaciones,cuartos, baños, área de 
#superficie total y cubierta. Los dos primeros con la moda al tomar valores enteros
# y los dos últimos con la mediana.

Train %>%
  count(rooms) %>% head() 
Train %>%
  count(bedrooms)
Train %>%
  count(bathrooms)

Test %>%
  count(rooms) %>% head() 
Test %>%
  count(bedrooms)
Test %>%
  count(bathrooms)


Train  <- Train %>% 
  mutate(
    rooms = ifelse(is.na(rooms) == T, as.numeric(names(sort(table(rooms))[1])), rooms),
    bathrooms = ifelse(is.na(bathrooms) == T, as.numeric(names(sort(table(bathrooms))[1])), bathrooms),
    bedrooms = ifelse(is.na(bedrooms) == T, as.numeric(names(sort(table(bedrooms))[1])), bedrooms)
  )

Test  <- Test %>% 
  mutate(
    rooms = ifelse(is.na(rooms) == T, as.numeric(names(sort(table(rooms))[1])), rooms),
    bathrooms = ifelse(is.na(bathrooms) == T, as.numeric(names(sort(table(bathrooms))[1])), bathrooms),
    bedrooms = ifelse(is.na(bedrooms) == T, as.numeric(names(sort(table(bedrooms))[1])), bedrooms)
  )

Train <- Train %>%
  mutate(
    rooms = as.integer(as.character(rooms)),
    bathrooms = as.integer(as.character(bathrooms)),
    bedrooms = as.integer(as.character(bedrooms))
  )

Test <- Test %>%
  mutate(
    rooms = as.integer(as.character(rooms)),
    bathrooms = as.integer(as.character(bathrooms)),
    bedrooms = as.integer(as.character(bedrooms))
  )

linear_imput_model_covered  <- lm(
  surface_covered ~ 
    property_type + rooms + bathrooms + bedrooms,
  data = Train, na.action = na.exclude
)   

linear_imput_model_total  <- lm(
  surface_total ~ 
    property_type + rooms + bathrooms + bedrooms,
  data = Train, na.action = na.exclude
)

Train$pred_total  <- predict(
  linear_imput_model_total,
  newdata = Train
)

Train$pred_covered  <- predict(
  linear_imput_model_covered,
  newdata = Train
)

linear_imput_model_covered  <- lm(
  surface_covered ~ 
    property_type + rooms + bathrooms + bedrooms,
  data = Test, na.action = na.exclude
)   

linear_imput_model_total  <- lm(
  surface_total ~ 
    property_type + rooms + bathrooms + bedrooms,
  data = Test, na.action = na.exclude
)

Test$pred_covered  <- predict(
  linear_imput_model_covered,
  newdata = Test
)

Test$pred_total  <- predict(
  linear_imput_model_total,
  newdata = Test
)

Train  <- Train  %>% 
  mutate(
    surface_covered = ifelse(is.na(surface_covered) == T, pred_covered, surface_covered),
    surface_total = ifelse(is.na(surface_total) == T, pred_total, surface_total)
  )

Test  <- Test  %>% 
  mutate(
    surface_covered = ifelse(is.na(surface_covered) == T, pred_covered, surface_covered),
    surface_total = ifelse(is.na(surface_total) == T, pred_total, surface_total)
  )
vis_dat(Train)
vis_dat(Test)

Train  <- Train  %>% 
  select(-pred_covered,-pred_total)

Test  <- Test  %>% 
  select(-pred_covered,-pred_total)

# Evaluamos anomalias de las variables numericas:
stargazer(Train,type="text")
# No hay precensia de incositencias dentro de la suerficie cubierta o total.
stargazer(Test,type="text")
# No hay precensia de incositencias dentro de la suerficie cubierta o total.

# En la muestra de entrenamieto se calcula el precio por metro cuadrado, 
# para inspeccionar inconsistencias:
Train <- Train %>%
  mutate(precio_por_mt2 = round(price / surface_total, 0))%>%
  mutate(precio_por_mt2  =precio_por_mt2/1000000 )  ## precio x Mt2 en millones. 
stargazer(Train["precio_por_mt2"],type="text")

# Detectamos valores atipicos en el precio de nuestro trainig set:
low <- round(mean(Train$precio_por_mt2) - 2*sd(Train$precio_por_mt2))
up <- round(mean(Train$precio_por_mt2) + 2*sd(Train$precio_por_mt2))
perc1 <- unname(round(quantile(Train$precio_por_mt2, probs = c(.01)),2))

Graph_1 <- Train %>%
  ggplot(aes(y = precio_por_mt2)) +
  geom_boxplot(fill = "darkblue", alpha = 0.4) +
  labs(
    title = "Muestra con valores atipicos",
    y = "Precio por metro cuadrado (millones)", x = "") +
  theme_bw()
Graph_2 <- Train %>%
  filter(between(precio_por_mt2, perc1,  up)) %>% 
  ggplot(aes(y = precio_por_mt2)) +
  geom_boxplot(fill = "darkblue", alpha = 0.4) +
  labs(
    title = "Muestra sin los valores atipicos",
    y = "Precio por metro cuadrado (millones)", x = "") +
  theme_bw()
grid.arrange(Graph_1, Graph_2, ncol = 2)

# Es conguente procedemos a eliminar esos valores atipicos:
Train <- Train %>% filter(between(precio_por_mt2, perc1, up))


# Procedemos con inspeccion espacial de la muestra Train:

# Eliminamos los valores faltantes de latitud o longitud
Train <- Train %>%
  filter(!is.na(lat) & !is.na(lon))
Test <- Test %>%
  filter(!is.na(lat) & !is.na(lon))

# Observamos la primera visualización
leaflet() %>%
  addTiles() %>%
  addCircles(lng = Train$lon, 
             lat = Train$lat)
# Nos aseguramos que solo tengamos observaciones dentro del limite politico administrativo de Bogota
limes_Bog <- getbb("Bogota Colombia")
limes_Bog

# Intente filtrar pero tambien se lleva observaciones de la localidad de teusaquillo
# Toca intentar por texto
#limes_chapinero <- getbb("Chapinero Bogota Colombia")
#limes_chapinero

Train <- Train %>%
  filter(between(lon, limes_Bog[1, "min"], limes_Bog[1, "max"]) & 
      between(lat, limes_Bog[2, "min"], limes_Bog[2, "max"]))

Test <- Test %>%
  filter(between(lon, limes_Bog[1, "min"], limes_Bog[1, "max"]) & 
           between(lat, limes_Bog[2, "min"], limes_Bog[2, "max"]))
#Train <- Train %>%
#  filter(!( between(lon, limes_chapinero[1, "min"], limes_chapinero[1, "max"]) &
#       between(lat, limes_chapinero[2, "min"], limes_chapinero[2, "max"])))


# Eliminamos los inmuebles con área menor a 15
Train <- Train %>% filter(surface_covered > 15)
Test <- Test %>% filter(surface_covered > 15)
# Escalamos para que se pueda graficar
Train <- Train %>% 
  mutate(precio_por_mt2_sc = (precio_por_mt2 - min(precio_por_mt2, na.rm = TRUE)) / 
           (max(precio_por_mt2, na.rm = TRUE) - min(precio_por_mt2, na.rm = TRUE)))

# Asignamos colores según tipo de inmueble
Train <- Train %>%
  mutate(color = case_when(property_type == "Apartamento" ~ "#EE6363",
                           property_type == "Casa" ~ "#7AC5CD",
                           TRUE ~ "gray"))  # Por si hay más tipos

# Creamos el popup en HTML
Train <- Train %>%
  mutate(popup_html = paste0("<b>Precio:</b> ", scales::dollar(price),
                             "<br> <b>Área:</b> ", as.integer(surface_total), " mt2",
                             "<br> <b>Tipo de inmueble:</b> ", property_type,
                             "<br> <b>Alcobas:</b> ", as.integer(rooms),
                             "<br> <b>Baños:</b> ", as.integer(bathrooms)))

# Coordenadas centrales
latitud_central <- mean(Train$lat, na.rm = TRUE)
longitud_central <- mean(Train$lon, na.rm = TRUE)

# Creamos el plot
leaflet() %>%
  addTiles() %>%
  setView(lng = longitud_central, lat = latitud_central, zoom = 12) %>%
  addCircles(lng = Train$lon, 
             lat = Train$lat, 
             col = Train$color,
             fillOpacity = 1,
             opacity = 1,
             radius = Train$precio_por_mt2_sc*10,
             popup = Train$popup_html)


#setwd("C:\\Users\\samue\\OneDrive\\Escritorio\\Economia\\Big Data y Machine Learning\\Taller 3\\")

localidades <- st_read(dsn = "stores\\datos espaciales\\localidades_bog", layer = "Loca")

names(localidades)
plot(localidades["LocNombre"])  
class(localidades)    # sf + data.frame 
head(localidades)     # primeras filas (tabla de atributos)
str(localidades)      # estructura (geometría + atributos)


Manzanas <- st_read(dsn = "stores\\datos espaciales\\manzanas", layer = "ManzanaEstratificacion")

names(Manzanas)
plot(Manzanas["CODIGO_MAN"])  
class(Manzanas)    # sf + data.frame 
head(Manzanas)     # primeras filas (tabla de atributos)
str(Manzanas)      # estructura (geometría + atributos)


# DEBERIAMOS DE PENSAR EN ELIMINAR SUBA Y CIUDAD BOLIVAR MUY POCAS OBSERVACIONES EL MAPA SE DISTORCIONA MUCHO
# Filtrar localidades urbanas
localidades_urbanas <- localidades %>%
  filter(LocNombre %in% c("USAQUEN", "CHAPINERO", "SANTA FE", "SAN CRISTOBAL",
                          "USME", "TUNJUELITO", "BOSA", "KENNEDY", "FONTIBON", 
                          "ENGATIVA", "SUBA", "BARRIOS UNIDOS", "TEUSAQUILLO", 
                          "LOS MARTIRES", "ANTONIO NARIÑO", "PUENTE ARANDA", 
                          "CANDELARIA", "RAFAEL URIBE URIBE", "CIUDAD BOLIVAR"))

# Plot con ggplot2
ggplot() +
  geom_sf(data = localidades_urbanas, fill = "red", color = "black", alpha = 0.4) +
  geom_sf(data = Manzanas, fill = NA, color = "blue", size = 0.3) +
  labs(title = "Localidades Urbanas y Sectores Urbanos Coincidentes") +
  theme_minimal()



#Tranformacion de nuestros datos de viviendas en Train a sf utilizando MAGNA-SIRGAS Bogotá proyectado
sf_Train<- st_as_sf(Train, coords = c("lon", "lat"),  crs = 4626)
sf_Train <- st_transform(sf_Train, crs = 3116)
localidades_urbanas <- st_transform(localidades_urbanas, crs = 3116)
Manzanas <- st_transform(Manzanas, crs = 3116)
#lo mismo para Test:
sf_Test<- st_as_sf(Test, coords = c("lon", "lat"),  crs = 4626)
sf_Test <- st_transform(sf_Test, crs = 3116)


# Definimos limites de bogota en Magnas sirgas proyectado 
pry_lim_x = c(999000, 1015000)
pry_lim_y = c(1007000, 1019000)

ggplot()+
  geom_sf(data=localidades_urbanas, color = "red") + 
  geom_sf(data=sf_Train,aes(color = precio_por_mt2) ,shape=15, size=0.3)+
  theme_bw()

# Deberiamos pensar en quitar Usme y ciudad Bolivar

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


#De las features del parque nos interesa su geomoetría y donde estan ubicados 
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


# El area urbana de bogota es lo sufcientemente plana para guiarse fielmente por los centorides:

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

#Creamos el mapa de Bogota con las ammenity seleccionadas:
leaflet() %>%
  addTiles() %>%
  setView(lng = longitud_central, lat = latitud_central, zoom = 12) %>%
  addPolygons(data = amenities_geometria, col = "red",weight = 10,
              opacity = 0.8, popup = amenities_geometria$name) %>%
  addCircles(lng = amenities_centroides$x, 
             lat = amenities_centroides$y, 
             col = "darkblue", opacity = 0.5, radius = 1)
#Creamos el mapa de Bogota con las public_transport seleccionadas:
leaflet() %>%
  addTiles() %>%
  setView(lng = longitud_central, lat = latitud_central, zoom = 12) %>%
  addPolygons(data = public_transport_geometria, col = "red",weight = 10,
              opacity = 0.8, popup = public_transport_geometria$name) %>%
  addCircles(lng = public_transport_centroides$x, 
             lat = public_transport_centroides$y, 
             col = "darkblue", opacity = 0.5, radius = 1)
#Creamos el mapa de Bogota con las leisure seleccionadas:
leaflet() %>%
  addTiles() %>%
  setView(lng = longitud_central, lat = latitud_central, zoom = 12) %>%
  addPolygons(data = leisure_geometria, col = "red",weight = 10,
              opacity = 0.8, popup = leisure_geometria$name) %>%
  addCircles(lng = leisure_centroides$x, 
             lat = leisure_centroides$y, 
             col = "darkblue", opacity = 0.5, radius = 1)

#Creamos el mapa de Bogota con las leisure seleccionadas:
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



#Creamos el mapa de Bogota con los mall:
leaflet() %>%
  addTiles() %>%
  setView(lng = longitud_central, lat = latitud_central, zoom = 12) %>%
  addPolygons(data = shop_geometria, col = "red",weight = 10,
              opacity = 0.8, popup = shop_geometria$name) %>%
  addCircles(lng = shop_centroides$x, 
             lat = shop_centroides$y, 
             col = "darkblue", opacity = 0.5, radius = 1)

#Creamos el mapa de Bogota con los supermarkets: NO TIENE MUCHOS NO SE SI ELIMINAR???
leaflet() %>%
  addTiles() %>%
  setView(lng = longitud_central, lat = latitud_central, zoom = 12) %>%
  addPolygons(data = building_geometria, col = "red",weight = 10,
              opacity = 0.8, popup = building_geometria$name) %>%
  addCircles(lng = building_centroides$x, 
             lat = building_centroides$y, 
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
Train_localizado <- calculo_distancias_por_feature(supermarket, "dist_supermarket")
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
Test_localizado <- calculo_distancias_Test_localizado_por_feature_(supermarket, "dist_supermarket")
Test_localizado <- calculo_distancias_Test_localizado_por_feature_(mall, "dist_mall")


# Porcedemos a crear variables a paritrde la informaicon de la vivienda:
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
  "prop_id", "price", "month", "year", "surf_total", "surf_cov", "rooms", "bdrms", "bathrm", "prop_typ",
  "title", "descript", "prec_mt2", "prec_mt2sc", "color", "popuphtml", "loc_nomb", "loc_admin", "loc_area",
  "loc_cod", "shp_leng", "shp_area", "obj_id", "cod_man", "estrato", "cod_zon", "cod_cri", "normatv",
  "acto_admin", "num_act", "fecha_act", "escala_cp", "fecha_cap", "responsab", "shp_area2", "shp_len", 
  "geometry", "dist_univ", "dist_hosp", "dist_rest", "dist_pol", "dist_esc", "dist_culto", "dist_disco", 
  "dist_parq", "dist_teatr", "dist_bar", "dist_plat", "dist_estac", "dist_gym", "dist_jard", "dist_park", 
  "dist_juego", "dist_vprim", "dist_vsec", "dist_vped", "dist_cycl", "dist_super", "dist_mall", 
  "bath_perb", "bath_perr", "ratio_cov", "bdrm_perr"
)


# Renombrar columnas de Test_localizado
names(Test_localizado) <- c(
  "prop_id", "price", "month", "year", "surf_total", "surf_cov", "rooms", "bdrms", "bathrm", "prop_typ", 
  "title", "descript", "obj_id", "cod_man", "estrato", "cod_zon", "cod_cri", "normatv", "acto_admin", "num_act", 
  "fecha_act", "escala_cp", "fecha_cap", "responsab", "shp_area", "shp_len", "geometry", "dist_univ", "dist_hosp", 
  "dist_rest", "dist_pol", "dist_esc", "dist_culto", "dist_disco", "dist_parq", "dist_teatr", "dist_bar", 
  "dist_plat", "dist_estac", "dist_gym", "dist_jard", "dist_park", "dist_juego", "dist_vprim", "dist_vsec", "dist_vped", 
  "dist_cycl", "dist_super", "dist_mall", "bath_perb", "bath_perr", "ratio_cov", "bdrm_perr")




# Guardamos en un archivo shapefield:
# para el trianing set:
st_write(Train_localizado, "stores\\work\\Train_localizado_reg\\Train_localizado_reg.shp")

#para el testing set:
st_write(Test_localizado, "stores\\work\\Test_localizado_reg\\Test_localizado_reg.shp")
