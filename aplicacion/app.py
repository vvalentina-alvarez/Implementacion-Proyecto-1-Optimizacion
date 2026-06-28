import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(
    page_title="Clasificador Adult Income - SVM",
    page_icon="💼",
    layout="wide"
)

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    model_path = os.path.join(base_dir, "resultados/svm_adult_pipeline.pkl")
    
    with open(model_path, "rb") as file:
        objeto = pickle.load(file)
    return objeto

objeto = load_model()

model = objeto["model"]
scaler = objeto["scaler"]
columns = objeto["columns"]

def preprocess_input(input_data, columns, scaler):
    input_encoded = pd.get_dummies(input_data, drop_first=False)
    input_encoded = input_encoded.reindex(columns=columns, fill_value=0)
    input_scaled = scaler.transform(input_encoded)
    return input_scaled

st.title("Clasificador de ingresos con SVM")

st.sidebar.header("Parámetros de entrada")

age = st.sidebar.number_input("Edad", min_value=17, max_value=90, value=35, step=1)

mapeo_workclass = {
    "Sector Privado": "Private",
    "Autoempleado (No incorporado)": "Self-emp-not-inc",
    "Autoempleado (Incorporado)": "Self-emp-inc",
    "Gobierno Federal": "Federal-gov",
    "Gobierno Local": "Local-gov",
    "Gobierno Estatal": "State-gov",
    "Trabajador sin pago / Voluntario": "Without-pay",
    "Nunca ha trabajado": "Never-worked"
}

mapeo_education = {
    "Licenciatura (Bachelors)": "Bachelors",
    "Universidad incompleta": "Some-college",
    "11vo grado": "11th",
    "Graduado de Secundaria": "HS-grad",
    "Escuela Profesional": "Prof-school",
    "Técnico Académico": "Assoc-acdm",
    "Técnico Vocacional": "Assoc-voc",
    "9no grado": "9th",
    "7mo-8vo grado": "7th-8th",
    "12vo grado": "12th",
    "Maestría (Masters)": "Masters",
    "1ro-4to grado": "1st-4th",
    "10mo grado": "10th",
    "Doctorado": "Doctorate",
    "5to-6to grado": "5th-6th",
    "Preescolar": "Preschool"
}

mapeo_marital_status = {
    "Casado/a": "Married-civ-spouse",
    "Divorciado/a": "Divorced",
    "Nunca casado/a (Soltero/a)": "Never-married",
    "Separado/a": "Separated",
    "Viudo/a": "Widowed",
    "Casado/a (Cónyuge ausente)": "Married-spouse-absent",
    "Casado/a (Fuerzas Armadas)": "Married-AF-spouse"
}

mapeo_occupation = {
    "Soporte Técnico": "Tech-support",
    "Artesanía y Reparación": "Craft-repair",
    "Otros Servicios": "Other-service",
    "Ventas": "Sales",
    "Ejecutivo / Gerencial": "Exec-managerial",
    "Especialidad Profesional": "Prof-specialty",
    "Limpiadores": "Handlers-cleaners",
    "Operador de Maquinaria e Inspector": "Machine-op-inspct",
    "Administrativo / Oficinista": "Adm-clerical",
    "Agricultura y Pesca": "Farming-fishing",
    "Transporte y Mudanza": "Transport-moving",
    "Servicio Doméstico Privado": "Priv-house-serv",
    "Servicios de Protección": "Protective-serv",
    "Fuerzas Armadas": "Armed-Forces"
}

mapeo_relationship = {
    "Esposa": "Wife",
    "Hijo/a propio/a": "Own-child",
    "Esposo": "Husband",
    "No en la familia": "Not-in-family",
    "Otro pariente": "Other-relative",
    "Soltero/a (Sin compromiso)": "Unmarried"
}

mapeo_race = {
    "Blanco": "White",
    "Asiático": "Asian-Pac-Islander",
    "Nativo Americano": "Amer-Indian-Eskimo",
    "Negro": "Black",
    "Otro": "Other"
}

mapeo_sex = {
    "Masculino": "Male",
    "Femenino": "Female"
}

mapeo_native_country = {
    "Estados Unidos": "United-States",
    "México": "Mexico",
    "Perú": "Peru",
    "Canadá": "Canada",
    "Alemania": "Germany",
    "Filipinas": "Philippines",
    "India": "India",
    "Inglaterra": "England",
    "China": "China",
    "Cuba": "Cuba",
    "Japón": "Japan",
    "Otro": "Other"
}


workclass_es = st.sidebar.selectbox(
    "Tipo de empleo",
    list(mapeo_workclass.keys())
)

workclass = mapeo_workclass[workclass_es]
workclass_es = st.sidebar.selectbox("Tipo de trabajo", list(mapeo_workclass.keys()))
workclass = mapeo_workclass[workclass_es]

education_es = st.sidebar.selectbox("Nivel educativo", list(mapeo_education.keys()))
education = mapeo_education[education_es]

marital_status_es = st.sidebar.selectbox("Estado civil", list(mapeo_marital_status.keys()))
marital_status = mapeo_marital_status[marital_status_es]

occupation_es = st.sidebar.selectbox("Ocupación", list(mapeo_occupation.keys()))
occupation = mapeo_occupation[occupation_es]

relationship_es = st.sidebar.selectbox("Relación familiar", list(mapeo_relationship.keys()))
relationship = mapeo_relationship[relationship_es]

race_es = st.sidebar.selectbox("Raza", list(mapeo_race.keys()))
race = mapeo_race[race_es]

sex_es = st.sidebar.selectbox("Sexo", list(mapeo_sex.keys()))
sex = mapeo_sex[sex_es]

native_country_es = st.sidebar.selectbox("País de origen", list(mapeo_native_country.keys()))
native_country = mapeo_native_country[native_country_es]


fnlwgt = st.sidebar.number_input(
    "Peso poblacional (fnlwgt)",
    min_value=10000,
    max_value=1500000,
    value=200000,
    step=1000
)

education_num = st.sidebar.number_input(
    "Años de Educación",
    min_value=1,
    max_value=16,
    value=10,
    step=1
)

capital_gain = st.sidebar.number_input(
    "Ganancia de capital",
    min_value=0,
    max_value=100000,
    value=0,
    step=100
)

capital_loss = st.sidebar.number_input(
    "Pérdida de capital",
    min_value=0,
    max_value=100000,
    value=0,
    step=100
)

hours_per_week = st.sidebar.number_input(
    "Horas trabajadas por semana",
    min_value=1,
    max_value=99,
    value=40,
    step=1
)



input_data = pd.DataFrame({
    "age": [age],
    "fnlwgt": [fnlwgt],
    "education.num": [education_num],
    "capital.gain": [capital_gain],
    "capital.loss": [capital_loss],
    "hours.per.week": [hours_per_week],
    "workclass": [workclass],
    "education": [education],
    "marital.status": [marital_status],
    "occupation": [occupation],
    "relationship": [relationship],
    "race": [race],
    "sex": [sex],
    "native.country": [native_country]
})

st.subheader("Datos seleccionados")

nombres_amigables = {
    "age": "Edad",
    "fnlwgt": "Peso poblacional",
    "education.num": "Años de educación",
    "capital.gain": "Ganancia de capital",
    "capital.loss": "Pérdida de capital",
    "hours.per.week": "Horas por semana",
    "workclass": "Clase laboral",
    "education": "Nivel educativo",
    "marital.status": "Estado civil",
    "occupation": "Ocupación",
    "relationship": "Relación familiar",
    "race": "Raza",
    "sex": "Sexo",
    "native.country": "País de origen"
}

df_visualizacion = input_data.rename(columns=nombres_amigables)


st.dataframe(df_visualizacion, use_container_width=True)

if st.button("Clasificar ingreso"):
    input_encoded = pd.get_dummies(input_data, drop_first=False)
    input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

    columnas_activas = input_encoded.loc[:, input_encoded.iloc[0] != 0]

    input_scaled = scaler.transform(input_encoded)
    prediction = model.predict(input_scaled)

    resultado = prediction[0]

    st.subheader("Resultado")

    if resultado == 1:
        st.success("La persona probablemente gana más de 50K.")
        st.write("Clase predicha: >50K")
    else:
        st.warning("La persona probablemente gana menor o igual a 50K.")
        st.write("Clase predicha: <=50K")