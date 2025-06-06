# ML & BD Problem Set 3:Making Money with ML?
Nicolas Lozano Huertas, Andres Gerardo Rendon, Samuel Narváez Muñoz & Jhan Camilo Pulido

## Descripción

Este repositorio contiene los Scripts con los modelos utilizados para predicir el precio de las propiedades de acuerdo a la ubicación y descripción del inmueble. El  conjunto  de  datos  utilizado  en  este  análisis  proviene  de  Properati,  Este  estudio  abarca  la  lista  de  propiedades  en  
Bogotá,  Colombia,  entre  2019  y  2021, utilizados para el desarrollo del Problem Set 3 de la 
clase de Machine learning y big data para economía aplicada.

## Estructura

- scripts: Contiene todos los codigos de R con el proprocesamiento de datos geográficos, textuales y de los modelos implementados. 
  - En la carpeta "Preprocesamiento de datos" debe correrse primero el script "Datos texto.R" ya después se debe correr el script "Preprocesamiento de datos completos.R" donde se hace pre-procesamiento de los datos, se cargan los paquetes y se crean funciones que se utilizan en los modelos.
  - Modelos: Contiene los modelos que fueron enviados a las submissions de Kaggle. 
- stores: Aquí seguardan los datos.
  - datos espaciales: Se guardan las bases de datos que se utilizaron para el procesamiento de datos espaciales.
  - provisionales: Se guardan diferentes códigos del procesamiento de datos en las bases. 
  - raw: En esta carpeta se deben dejar los datos "crudos". Debido al peso de los archivos no se pudieron dejar en el repositorio.
  - work: Aquí se guardan las bases de datos despues del pre-procesamiento.
  - sub: En esta carpeta quedan los archivos en formato para ser enviados a kaggle.
- views: Aquí quedan guardados imagenes y figuras relevantes para el desarrollo del PS.
- document: En esta carpeta se puede encontrar el PDF con todo el desarrollo del taller.
