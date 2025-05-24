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

train <- read_sf("\\stores\\work_jcp\\Train\\Train.shp")
test <- read_sf("\\stores\\work_jcp\\Test\\Test.shp")