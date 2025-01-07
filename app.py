import streamlit as st
import pandas as pd
import pickle
import os
import joblib

# Loaded the model.pkl from files
model = joblib.load('https://github.com/sid-almeida/pc_calculator/blob/main/model.pkl')

# Criei um discionário com as previsões e número e depois o label que corresponde

pc_label = {0 : 'Nulo', 1 : 'Baixo', 2 : 'Alto', 3 : 'Muito Alto'}

# Create a sidebar header
with st.sidebar:
    st.image("https://github.com/sid-almeida/pc_calculator/blob/main/image.png?raw=true", width=250)
    st.title("Potencial de Crescimento")
    choice = st.radio("**Navegação:**", ("Sobre", "Previsão em Lote"))
    st.info('**Nota:** Por favor, esteja ciente de que este aplicativo é destinado exclusivamente para fins educacionais. É fortemente desaconselhável utilizar esta ferramenta para tomar quaisquer decisões financeiras.')


if choice == "Sobre":
    # Create a title and sub-title
    st.write("""
    # Previsão de Potencial de Crescimento Empresarial
    Este app busca prever o Potencial de Crescimento de Empresas do Continente Americano!
    """)
    st.write('---')
    st.write('**Sobre o App:**')
    st.write('Este aplicativo foi desenvolvido com o objetivo de classificar empresas com base em seu Potencial de Crescimento, utilizando o modelo de aprendizado de máquina **XGBoost Classifier**. O modelo foi treinado com um conjunto de dados abrangente, que inclui diversos parâmetros econômicos e macroeconômicos de empresas do continente americano.')
    st.info('**Nota:** Por favor, esteja ciente de que este aplicativo é destinado exclusivamente para fins educacionais. É fortemente desaconselhável utilizar esta ferramenta para tomar quaisquer decisões financeiras.')
    st.write('---')
    st.write('**Sobre os Dados**')
    st.write('O dataset utilizado neste trabalho foi desenvolvido por meio da coleta e integração de dados públicos provenientes de diversas fontes. Ele reúne uma série de parâmetros econômicos e macroeconômicos sobre empresas do continente americano, proporcionando uma visão abrangente sobre o desempenho e as características dessas organizações. A combinação dessas informações em um único conjunto de dados permite a análise de tendências econômicas e o comportamento das empresas, possibilitando a aplicação de técnicas avançadas de aprendizado de máquina para prever diversos aspectos relacionados ao seu crescimento e performance no mercado. O dataset abrange variáveis como indicadores financeiros, taxas de crescimento econômico, e outros parâmetros que impactam diretamente o setor empresarial americano.')
    st.write('---')

if choice == 'Previsão em Lote':
    # Create a title and sub-title
    st.write("""
    # Previsão de Potencial de Crescimento Empresarial
    Este app busca prever o Potencial de Crescimento de Empresas do Continente Americano!
    """)
    st.write('---')
    st.info('**Guia:** Por favor, envie o dataset com os parâmetros de previsão.')
    st.write('---')
    # Create a file uploader to upload the dataset of predicting features
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")
    if uploaded_file is not None:
        df_pred = pd.read_csv(uploaded_file)
        st.write('---')
        st.write('**Dataset:**')
        st.write(df_pred)
        # Create a button to predict the probability of bankruptcy
        if st.button("Prever"):
            # Predict the growth potential of the companies in the uploaded dataset
            df_pred['Potencial PC'] = model.predict(df_pred)
            # Convert 'Potencial PC' que será numérico nas labels correspondentesda 'pc_label'
            df_pred['Potencial PC'] = df_pred['Potencial PC'].map(pc_label)
            # Create a success message
            st.success('O Potencial de Crescimento foi Calculado com Sucesso!')
            st.write('---')
            st.write('**Dataset Calculado:**')
            st.write(df_pred)
            # Create a button to download the dataset with the predicted probability of bankruptcy
            if st.download_button(label='Baixar o Dataset Calculado', data=df_pred.to_csv(index=False), file_name='predicted.csv', mime='text/csv'):
                pass
        else:
            st.write('---')
            st.info('Clique no botão para prever o Potencial de Crescimento!')
    else:
        st.write('---')



st.write('Made with ❤️ by [Sidnei Almeida](https://www.linkedin.com/in/saaelmeida93/)')
