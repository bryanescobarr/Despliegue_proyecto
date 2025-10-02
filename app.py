
# Despliegue
"""
- Cargamos el modelo
- Cargamos los datos futuros
- Preparar los datos futuros
- Aplicamos el modelo para la predicción
"""

#Cargamos librerías principales
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Cargamos el modelo
import pickle
filename = 'modelo.pkl'
modelo, labelencoder, variables, min_max_scaler, = pickle.load(open(filename, 'rb'))

#Cargamos los datos futuros
#data = pd.read_csv("videojuegos-datosFuturos.csv")
#data.head()

#Interfaz gráfica
#Se crea interfaz gráfica con streamlit para captura de los datos

import streamlit as st

st.title('Clasificación dsociodemográfica de nuevos afiliados')

TipoAfiliado = st.selectbox('TipoAfiliado', ["'Titular'","'Persona a cargo'", "'Conyuge'"])
TipoTit  = st.selectbox('TipoTit', ["'Afiliado Dependiente'","'Afiliado Facultativo'", "'Independiente'","'Pensionado'"])
GeneroAfil = st.selectbox('GeneroAfil', ["'Femenino'","'BMasculino'"])
EdadAfil = st.slider('EdadAfil', min_value=0, max_value=80, value=0, step=1)
Municipio = st.selectbox('Municipio', ["'MEDELLÍN'"])
CategoriaAfil = st.selectbox('CategoriaAfil', ["'A'","'B'", "'C'"])
BenefSubAfil  = st.selectbox('BenefSubAfil', ["'S'","'N'"])
ValorSubsidoAfil = st.slider('ValorSubsidoAfil', min_value=0, max_value=1000000, value=0, step=1)
Parentesco = st.selectbox('Parentesco', ["'Titular'", "'Familiar'"])
NroPersonasFamilia = st.slider('NroPersonasFamilia', min_value=0, max_value=20, value=0, step=1)
TipoFamilia =  st.selectbox('TipoFamilia', ["'Familia nuclear'", "'Familia unipersonal'", "'Familia sin hijos'",
       "'Familia monoparental'", "'Familia extensa'", "'Sin Clasificar'",
       "'Familia homoparental'"])
PercapitaSalario = st.slider('PercapitaSalario', min_value=0, max_value=1000000000, value=0, step=1)
ClaseSocialSalario = st.selectbox('ClaseSocialSalario', ["'Sin dato'", "'Clase media'", "'Clase baja'", "'Clase alta'"])
NroServiciosUsuario = st.slider('NroServiciosUsuario', min_value=0, max_value=200, value=0, step=1)
NroUsos = st.slider('NroUsos', min_value=0, max_value=200000, value=0, step=1)
MontoPagado = st.slider('MontoPagado', min_value=0, max_value=1000000000, value=0, step=1)





#Dataframe
datos = [[TipoAfiliado, TipoTit, GeneroAfil, EdadAfil,
       Municipio, CategoriaAfil, BenefSubAfil, ValorSubsidoAfil,
       Parentesco, NroPersonasFamilia, TipoFamilia, PercapitaSalario,
       ClaseSocialSalario, NroServiciosUsuario, NroUsos, MontoPagado]]
data = pd.DataFrame(datos, columns=['TipoAfiliado', 'TipoTit', 'GeneroAfil', 'EdadAfil',
       'Municipio', 'CategoriaAfil', 'BenefSubAfil', 'ValorSubsidoAfil',
       'Parentesco', 'NroPersonasFamilia', 'TipoFamilia', 'PercapitaSalario',
       'ClaseSocialSalario', 'NroServiciosUsuario', 'NroUsos', 'MontoPagado']) #Dataframe con los mismos nombres de variables

#Se realiza la preparación
data_preparada=data.copy()

#En despliegue drop_first= False

data_preparada = pd.get_dummies(data_preparada, columns=['BenefSubAfil','GeneroAfil','Parentesco'], drop_first=True, dtype = int)

#Esto aquí es porque estas columnas tienen más de 2 categorías
data_preparada = pd.get_dummies(data_preparada, columns=['TipoAfiliado', 'TipoTit', 'Municipio','CategoriaAfil', 
'TipoFamilia', 'ClaseSocialSalario'], drop_first=False, dtype = int) 

data_preparada = data_preparada.drop(['TipoAfiliado_Persona a cargo', 'ClaseSocialSalario_Sin dato', 'TipoTit_Afiliado Facultativo','PercapitaSalario', 'NroUsos', 'MontoPagado',
       'TipoTit_Afiliado Dependiente', 'TipoTit_Afiliado Facultativo',
       'TipoTit_Independiente', 'TipoTit_Pensionados', 'Municipio_BARBOSA',
       'Municipio_GIRARDOTA', 'CategoriaAfil_C',
       'TipoFamilia_Familia homoparental', 'TipoFamilia_Familia unipersonal',
       'TipoFamilia_Sin Clasificar', 'ClaseSocialSalario_Clase alta',
       'ClaseSocialSalario_Clase baja', 'ClaseSocialSalario_Clase media',
       'ClaseSocialSalario_Sin dato'],axis=1)

data_preparada[['EdadAfil', 'ValorSubsidoAfil','NroPersonasFamilia','NroServiciosUsuario']] = min_max_scaler.fit_transform(data[['EdadAfil', 'ValorSubsidoAfil','NroPersonasFamilia','NroServiciosUsuario']])

#Se normaliza la edad para predecir con Knn, Red
#En los despliegues no se llama fit
#data_preparada[['Edad']]= min_max_scaler.transform(data_preparada[['Edad']])
#data_preparada.head()

"""#Predicciones

CLÚSTER 0 - Adultos mayormente hombres, titulares, dependientes, de categoría A y que viviendo solos en Medellín
CLÚSTER 1 - Jóvenes de 22 años, mayormente mujeres, dependientes, de clase baja, que viven en familias monoparentales en Medellín
CLÚSTER 2 - Mujeres jóvenes, de 31 años en promedio, que viven principalmente en Medellín, en familias numerosas
CLÚSTER 3 - Adultos de 33 años, con familias nucleares y monoparentales en su mayoría, que viven principalmente en Medellín
CLÚSTER 4 - Adultos de nivel socioeconómico alto (Categoría C), con ingresos per cápita muy elevados (~$6M), que residen principalmente en Medellín
CLÚSTER 5 - Mujeres adultas, cónyuges, de familias nucleares y sin hijos en su mayoría, con ingresos muy bajos y concentradas en la clase baja
CLÚSTER 6 - Hombres jóvenes-adultos (30-40 años), viven solos, en su mayoría empleados dependientes, con ingresos relativamente altos
CLÚSTER 7 - Hombres jóvenes, Con hogares nucleares y varios dependientes (3-4 personas en promedio), en su mayoría beneficiarios o titulares dependientes
CLÚSTER 8 - Adultos maduros (casi 50 años), en hogares pequeños y sin hijos dependientes, mayoritariamente titulares dependientes
"""

#Hacemos la predicción con el Tree
Y_pred = modelo.predict(data_preparada)
print(Y_pred)

data['Prediccion']=Y_pred
data.head()


data

