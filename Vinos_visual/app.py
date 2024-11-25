import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle
from sklearn import svm
import streamlit as st


# Path del modelo preentrenado
MODEL_PATH = 'models/pickle_modelsvm.pkl'

# Se recibe la imagen y el modelo, devuelve la predicción
def model_prediction(x_in, model):

    x = np.asarray(x_in).reshape(1,-1)
    preds=model.predict(x)

    return preds


def main():
    
    model=''

    # Se carga el modelo
    if model=='':
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
    
    # Título
    html_temp = """
    <h1 style="color:#181082;text-align:center;">SISTEMA INTELIGENTE DE RECOMENDACIÓN DE CULTIVOS EN TOLIMA</h1>
    </div>
    """
   
    st.markdown(html_temp,unsafe_allow_html=True)

    # Lecctura de datos
    #Datos = st.text_input("Ingrese los valores para clasificar el vino:")
    Alcohol= st.text_input("Alcohol(N)(%)):")
    Malic_Acid = st.text_input("Ácido malicidio(mg/L)):")
    Ash = st.text_input("Ash(K en %):")
    Ash_Alcanity = st.text_input("Ash_Alcanitya(C):")
    Magnesium = st.text_input("Magnesium %:")
    Total_Phenols = st.text_input("Total_Phenols:")
    Flavanoids = st.text_input("Flavanoids:")
    Nonflavanoid_Phenols = st.text_input("Nonflavanoid_Phenols:")
    Proanthocyanins = st.text_input("Proanthocyanins:")
    Color_Intensity = st.text_input("Color_Intensit:")
    Hue = st.text_input("Hue:")
    OD280 = st.text_input("OD280:")
    Proline = st.text_input("Proline:")
   
   
    # El botón predicción se usa para iniciar el procesamiento
    if st.button("Predicción del vino:"): 
        #x_in = list(np.float_((Datos.title().split('\t'))))
        x_in =[np.float_(N.title()),
                    np.float_( Alcohol.title()),
                    np.float_(Malic_Acid.title()),
                    np.float_(Ash.title()),
                    np.float_(Ash_Alcanity.title()),
                    np.float_(Magnesium.title()),
                    np.float_(Total_Phenols.title()),
                    np.float_(Flavanoids.title()),
                    np.float_(Nonflavanoid_Phenols.title()),
                    np.float_(Proanthocyanins.title()),
                    np.float_(Color_Intensity.title()),
                    np.float_(Hue.title()),
                    np.float_(OD280.title()),
                    np.float_(Proline.title())]                    
                                      
                    

        predictS = model_prediction(x_in, model)
        st.success('EL VINO PREDICHO ES: {}'.format(predictS[0]).upper())

    #st.image("clustering.jpg", caption="clustering")

    # Botón para cerrar la aplicación
    if st.button("Cerrar aplicación"):
    # Mensaje para notificar al usuario que la ventana se cerrará
        st.write("Intentando cerrar la ventana del navegador...")
    
    # Ejecutar JavaScript para cerrar la pestaña del navegador
    close_script = """
    <script>
    function closeWindow() {
        if (confirm("¿Estás seguro de que deseas cerrar la aplicación?")) {
            window.open('', '_self', ''); 
            window.close();
        } else {
            alert("La ventana no se cerró.");
        }
    }
    closeWindow();
    </script>
    """
    st.markdown(close_script, unsafe_allow_html=True)

    # Contenido adicional que solo se muestra si la ventana no se cierra
    st.write("Gracias por utilizar la aplicación.")

if __name__ == '__main__':
    main()