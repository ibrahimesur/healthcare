import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Hidrojel Analiz Uygulaması")

uploaded_file = st.file_uploader("Excel dosyası yükle", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("Veri Tablosu", df)

    # Zamanla İlaç Salımı Çizgi Grafiği
    if "Zaman" in df.columns and "İlaç Salımı" in df.columns and "Hidrojel" in df.columns:
        fig, ax = plt.subplots()
        sns.lineplot(data=df, x="Zaman", y="İlaç Salımı", hue="Hidrojel", ax=ax)
        st.pyplot(fig)

    # Boxplot
    if "Hidrojel" in df.columns and "İlaç Salımı" in df.columns and "pH" in df.columns:
        fig2, ax2 = plt.subplots()
        sns.boxplot(data=df, x="Hidrojel", y="İlaç Salımı", hue="pH", ax=ax2)
        st.pyplot(fig2)

    # Isı Haritası
    if "Hidrojel" in df.columns and "pH" in df.columns and "Zaman" in df.columns and "İlaç Salımı" in df.columns:
        pivot = df.pivot_table(index="Hidrojel", columns=["pH", "Zaman"], values="İlaç Salımı")
        fig3, ax3 = plt.subplots(figsize=(8,4))
        sns.heatmap(pivot, annot=True, cmap="YlGnBu", ax=ax3)
        st.pyplot(fig3) 