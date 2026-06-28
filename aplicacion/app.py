import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Clasificador Adult Income - SVM",
    page_icon="💼",
    layout="wide"
)

@st.cache_resource
def load_model():
    with open("resultados/svm_adult_pipeline.pkl", "rb") as file:
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

workclass = st.sidebar.selectbox(
    "Tipo de trabajo",
    [
        "Private",
        "Self-emp-not-inc",
        "Self-emp-inc",
        "Federal-gov",
        "Local-gov",
        "State-gov",
        "Without-pay",
        "Never-worked"
    ]
)

fnlwgt = st.sidebar.number_input(
    "fnlwgt",
    min_value=10000,
    max_value=1500000,
    value=200000,
    step=1000
)

education = st.sidebar.selectbox(
    "Nivel educativo",
    [
        "Bachelors",
        "Some-college",
        "11th",
        "HS-grad",
        "Prof-school",
        "Assoc-acdm",
        "Assoc-voc",
        "9th",
        "7th-8th",
        "12th",
        "Masters",
        "1st-4th",
        "10th",
        "Doctorate",
        "5th-6th",
        "Preschool"
    ]
)

education_num = st.sidebar.number_input(
    "Education num",
    min_value=1,
    max_value=16,
    value=10,
    step=1
)

marital_status = st.sidebar.selectbox(
    "Estado civil",
    [
        "Married-civ-spouse",
        "Divorced",
        "Never-married",
        "Separated",
        "Widowed",
        "Married-spouse-absent",
        "Married-AF-spouse"
    ]
)

occupation = st.sidebar.selectbox(
    "Ocupación",
    [
        "Tech-support",
        "Craft-repair",
        "Other-service",
        "Sales",
        "Exec-managerial",
        "Prof-specialty",
        "Handlers-cleaners",
        "Machine-op-inspct",
        "Adm-clerical",
        "Farming-fishing",
        "Transport-moving",
        "Priv-house-serv",
        "Protective-serv",
        "Armed-Forces"
    ]
)

relationship = st.sidebar.selectbox(
    "Relación familiar",
    [
        "Wife",
        "Own-child",
        "Husband",
        "Not-in-family",
        "Other-relative",
        "Unmarried"
    ]
)

race = st.sidebar.selectbox(
    "Raza",
    [
        "White",
        "Asian-Pac-Islander",
        "Amer-Indian-Eskimo",
        "Other",
        "Black"
    ]
)

sex = st.sidebar.selectbox(
    "Sexo",
    ["Male", "Female"]
)

capital_gain = st.sidebar.number_input(
    "Capital gain",
    min_value=0,
    max_value=100000,
    value=0,
    step=100
)

capital_loss = st.sidebar.number_input(
    "Capital loss",
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

native_country = st.sidebar.selectbox(
    "País de origen",
    [
        "United-States",
        "Mexico",
        "Peru",
        "Canada",
        "Germany",
        "Philippines",
        "India",
        "England",
        "China",
        "Cuba",
        "Japan",
        "Other"
    ]
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
st.dataframe(input_data, use_container_width=True)

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