import streamlit as st
import pandas as pd
import joblib

# Title
st.header("Streamlit Machine Learning App")

# Input bar 1
gp = st.number_input("Games played")

# Input bar 2
fgm = st.number_input("average field goals attempted per game in the season")

# Input bar 3
fg3m = st.number_input("average 3 point field goals made per game in the season")

# Input bar 4
ftm = st.number_input("average free throws made per game in the season")

# If button is pressed
if st.button("Submit"):
    
    # Unpickle classifier
    clf = joblib.load("model/NBA_model2.pkl")


    valores = [[gp, fgm, fg3m, ftm]]
    prediccion = clf.predict(valores)

    tolerancia_pts = 0.1

    X_completo= pd.read_csv('data/Lista_completa.csv')

    resultado = X_completo.loc[(X_completo['games_played'] == valores[0][0]) & 
                                (X_completo['fgm'] == valores[0][1]) &
                                (X_completo['fg3m'] == valores[0][2]) & 
                                (X_completo['ftm'] == valores[0][3]) &
                                (abs(X_completo['pts'] - prediccion) < tolerancia_pts),
                                ['first', 'last', 'year', 'team', 'pts']]

    # Verificar si se encontraron resultados
    if not resultado.empty:
        jugador = resultado.iloc[0]  # Obtener la primera fila del resultado
        st.text(f"Nombre del jugador: {jugador['first']} {jugador['last']}")

        st.text(f"Año: {jugador['year']}")
        st.text(f"Equipo: {jugador['team']}")
        st.text(f"Puntos: {jugador['pts']}")
    else:
        st.text("No se encontraron resultados.")

    st.text(f"Predicción del modelo: {prediccion}")

