import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, f1_score
import pickle
import numpy as np
#  Despues de ejecutar con F5 se digita: "streamlit  run app1.py"

# corriendo el modelo entrenado
pkl_filename = "models/pickle_modelsvm.pkl"
with open(pkl_filename, 'rb') as file:
    model = pickle.load(file)

def predict_wine_type(input_data):
    prediction = model.predict(input_data)
    return prediction[0]

def main():
    st.title("App Predicción de Clase de Vino")

    uploaded_file = st.file_uploader("Seleccionar un archivo CSV para testeo", type="csv")
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("Conjunto de Datos de Testeo:")
            st.dataframe(data.head())

            if st.button("Predictor"):
                # Assuming the input features are in the correct columns in the uploaded CSV
                X = data.drop(["wine_type", "cluster"], axis=1, errors='ignore')  # Handle potential missing columns
                y_true = data['wine_type']  # Get true values if available

                try:
                    y_pred = model.predict(X)
                    
                    st.write("Métricas de Clasificación:")
                    st.text(classification_report(y_true, y_pred))
                    st.write("Accuracy:", accuracy_score(y_true, y_pred))
                    st.write("Recall:", recall_score(y_true, y_pred, average='weighted'))
                    st.write("F1-score:", f1_score(y_true, y_pred, average='weighted'))
                    
                    cm = confusion_matrix(y_true, y_pred)
                    st.write("Matriz de Confusión")
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                                xticklabels=['blanco', 'rosado', 'vino tinto'],
                                yticklabels=['blanco', 'rosado', 'vino tinto'])
                    plt.xlabel("Predicted")
                    plt.ylabel("Actual")
                    plt.title("Confusion Matrix")
                    st.pyplot(plt)
                    
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    
        except Exception as e:
            st.error(f"Error loading CSV file: {e}")

    if st.button("Cerrar la App"):
        st.stop()
if __name__ == "__main__":
    main()