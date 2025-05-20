import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit
# from scipy import stats # ANOVA şimdilik kaldırıldı
import io # PDF kaydetme için kullanılabilir

st.title("Hidrojel Analiz Uygulaması")

# Veriyi session state'te sakla
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["Hidrojel", "pH", "Zaman", "Şişme Oranı", "İlaç Salımı"])

# --- Veri Girişi ve Ekleme ---
st.header("Veri Girişi")

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    hidrojel = st.selectbox("Hidrojel", ["Kitosan", "Poli(akrilik asit)", "Jelatin"])
with col2:
    ph = st.text_input("pH")
with col3:
    zaman = st.text_input("Zaman (saat)")
with col4:
    sisme = st.text_input("Şişme Oranı (%)")
with col5:
    ilac = st.text_input("İlaç Salımı (%)")
with col6:
    st.write("\n") # Butonu hizalamak için boşluk
    if st.button("Ekle"):
        try:
            new_row = pd.DataFrame([{
                "Hidrojel": hidrojel,
                "pH": float(ph),
                "Zaman": float(zaman),
                "Şişme Oranı": float(sisme),
                "İlaç Salımı": float(ilac)
            }])
            st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
            st.success("Veri eklendi!")
        except ValueError:
            st.error("Lütfen sayısal alanlara geçerli değerler girin.")

# --- Veri Tablosu ---
st.header("Veri Tablosu")
# st.dataframe(st.session_state.data) # Sadece göstermek için
st.data_editor(st.session_state.data, key='editor') # Düzenlenebilir tablo

# --- Analiz ve Kaydetme Butonları ---
st.header("Analiz ve Kaydetme")

btn_col1, btn_col2, btn_col3, btn_col4, btn_col5, btn_col6 = st.columns(6)

# Bu butonların işlevselliği daha sonra eklenecek
with btn_col1:
    if st.button("Çizgi Grafiği"):
        st.session_state.current_plot = "line"
with btn_col2:
    if st.button("Boxplot"):
        st.session_state.current_plot = "box"
with btn_col3:
    if st.button("Isı Haritası"):
        st.session_state.current_plot = "heatmap"
with btn_col4:
    if st.button("Kinetik Model"):
        st.session_state.current_plot = "kinetic"

# Kaydetme butonları için farklı sütunlar kullanabiliriz veya yan yana devam edebiliriz
btn_col_save1, btn_col_save2, btn_col_save3 = st.columns(3)

with btn_col_save1:
    # Excel Kaydetme (Fonksiyon daha sonra eklenecek)
    if st.session_state.data.empty:
        st.download_button(label="Excel Kaydet", data="", file_name="hidrojel_veri.xlsx", disabled=True)
    else:
        excel_data = io.BytesIO()
        st.session_state.data.to_excel(excel_data, index=False)
        st.download_button(
            label="Excel Kaydet",
            data=excel_data.getvalue(),
            file_name="hidrojel_veri.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

with btn_col_save2:
    # PDF Kaydetme (Fonksiyon daha sonra eklenecek)
     if st.button("PDF Kaydet", disabled=st.session_state.data.empty):
         st.session_state.save_pdf_trigger = True

with btn_col_save3:
     # Veri Yükle (Zaten vardı, entegre edilecek)
     uploaded_file = st.file_uploader("Veri Yükle", type=["xlsx"], key='file_uploader')
     if uploaded_file is not None:
         try:
             loaded_df = pd.read_excel(uploaded_file)
             # Sütunları kontrol et
             if all(col in loaded_df.columns for col in ["Hidrojel", "pH", "Zaman", "Şişme Oranı", "İlaç Salımı"]):
                 st.session_state.data = pd.concat([st.session_state.data, loaded_df], ignore_index=True)
                 st.success("Veri yüklendi!")
             else:
                 st.error("Yüklenen Excel dosyası gerekli sütunları içermiyor. (Hidrojel, pH, Zaman, Şişme Oranı, İlaç Salımı)")
         except Exception as e:
             st.error(f"Dosya yüklenirken hata oluştu: {e}")

# Grafik ve Metin Sonuçları Gösterim Alanı
# Bu kısım, buton tıklamalarına göre güncellenecek

# Initial state for plot display
if 'current_plot' not in st.session_state:
    st.session_state.current_plot = None

# Initial state for text results
if 'text_results' not in st.session_state:
    st.session_state.text_results = ""

# Initial state for PDF save trigger
if 'save_pdf_trigger' not in st.session_state:
    st.session_state.save_pdf_trigger = False


st.header("Analiz Sonuçları")

# Grafik alanı için placeholder (isteğe bağlı)
plot_placeholder = st.empty()

# Metin sonuçları alanı
text_results_placeholder = st.empty()

# --- Grafik ve Analiz Mantığı (Fonksiyonlar Buraya Gelecek) ---

# plot_line, plot_box, plot_heatmap, plot_kinetic, save_pdf fonksiyonları analysis.py'den Streamlit'e uyarlanacak

# Örnek Fonksiyon Yapısı (placeholder):
def plot_line(df):
    # Clear previous plot (Streamlit handles this somewhat, but good practice)
    plt.close('all')
    if not df.empty and "Zaman" in df.columns and "İlaç Salımı" in df.columns and "Hidrojel" in df.columns:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.lineplot(data=df, x="Zaman", y="İlaç Salımı", hue="Hidrojel", ax=ax)
        ax.set_title("Zamanla İlaç Salımı Değişimi")
        ax.set_ylabel("İlaç Salımı (%)")
        ax.set_xlabel("Zaman (saat)")
        return fig
    return None

def plot_box(df):
     plt.close('all')
     if not df.empty and "Hidrojel" in df.columns and "İlaç Salımı" in df.columns and "pH" in df.columns:
         fig, ax = plt.subplots(figsize=(6,4))
         sns.boxplot(data=df, x="Hidrojel", y="İlaç Salımı", hue="pH", ax=ax)
         ax.set_title("Farklı pH Değerlerinde İlaç Salımı Dağılımı (Boxplot)")
         ax.set_ylabel("İlaç Salımı (%)")
         ax.set_xlabel("Hidrojel")
         return fig
     return None

def plot_heatmap(df):
     plt.close('all')
     if not df.empty and "Hidrojel" in df.columns and "pH" in df.columns and "Zaman" in df.columns and "İlaç Salımı" in df.columns:
         try:
             pivot = df.pivot_table(index="Hidrojel", columns=["pH", "Zaman"], values="İlaç Salımı")
             fig, ax = plt.subplots(figsize=(8,4))
             sns.heatmap(pivot, annot=True, cmap="YlGnBu", ax=ax)
             ax.set_title("Hidrojel, pH ve Zaman'a Göre İlaç Salımı Isı Haritası")
             plt.tight_layout() # Grafiğin sıkışmasını önle
             return fig
         except Exception as e:
              st.warning(f"Isı Haritası oluşturulamadı: {e}. Lütfen veri formatını ve eksik değerleri kontrol edin.")
              return None
     return None

def plot_kinetic(df):
    plt.close('all')
    results = []
    figs = []
    text_output = ""
    if not df.empty and "Zaman" in df.columns and "İlaç Salımı" in df.columns:
        for (hidrojel, pH), grup in df.groupby(["Hidrojel", "pH"]):
            t = grup["Zaman"].values
            Q = grup["İlaç Salımı"].values
            if len(t) >= 2:
                def higuchi_model(t, k):
                    return k * np.sqrt(t)
                def korsmeyer_peppas_model(t, k, n):
                    return k * (t ** n)
                try:
                    popt_h, _ = curve_fit(higuchi_model, t, Q, maxfev=10000)
                    popt_kp, _ = curve_fit(korsmeyer_peppas_model, t, Q, bounds=(0, [np.inf, 1]), maxfev=10000)
                    fig, ax = plt.subplots(figsize=(5,3))
                    ax.scatter(t, Q, label="Deneysel")
                    ax.plot(t, higuchi_model(t, *popt_h), label=f"Higuchi (k={popt_h[0]:.2f})")
                    ax.plot(t, korsmeyer_peppas_model(t, *popt_kp), label=f"Korsmeyer-Peppas (k={popt_kp[0]:.2f}, n={popt_kp[1]:.2f})")
                    ax.set_title(f"{hidrojel} - pH {pH} Kinetik Model")
                    ax.set_xlabel("Zaman (saat)")
                    ax.set_ylabel("İlaç Salımı (%)")
                    ax.legend()
                    figs.append(fig)
                    results.append(f"{hidrojel} - pH {pH}: Higuchi k={popt_h[0]:.2f}, Korsmeyer-Peppas k={popt_kp[0]:.2f}, n={popt_kp[1]:.2f}")
                except Exception as e:
                     st.warning(f"{hidrojel} - pH {pH} için kinetik model oluşturulamadı: {e}. Yeterli veri veya uygun formatta olmayabilir.")
                     results.append(f"{hidrojel} - pH {pH}: Model hatası.")
            else:
                 st.warning(f"{hidrojel} - pH {pH} için kinetik model oluşturulamadı: En az 2 veri noktası gerekli.")
                 results.append(f"{hidrojel} - pH {pH}: Yetersiz veri.")
    text_output = '\n'.join(results)
    return figs, text_output

def save_pdf(df):
     plt.close('all')
     figs = []
     # Çizgi grafiği
     if "Zaman" in df.columns and "İlaç Salımı" in df.columns and "Hidrojel" in df.columns:
         fig1, ax1 = plt.subplots(figsize=(6,4))
         sns.lineplot(data=df, x="Zaman", y="İlaç Salımı", hue="Hidrojel", style="pH", markers=True, dashes=False, ax=ax1)
         ax1.set_title("Zamanla İlaç Salımı Değişimi")
         figs.append(fig1)
     # Boxplot
     if "Hidrojel" in df.columns and "İlaç Salımı" in df.columns and "pH" in df.columns:
         fig2, ax2 = plt.subplots(figsize=(6,4))
         sns.boxplot(data=df, x="Hidrojel", y="İlaç Salımı", hue="pH", ax=ax2)
         ax2.set_title("Boxplot")
         figs.append(fig2)
     # Isı Haritası
     if "Hidrojel" in df.columns and "pH" in df.columns and "Zaman" in df.columns and "İlaç Salımı" in df.columns:
          try:
              pivot = df.pivot_table(index="Hidrojel", columns=["pH", "Zaman"], values="İlaç Salımı")
              fig3, ax3 = plt.subplots(figsize=(8,4))
              sns.heatmap(pivot, annot=True, cmap="YlGnBu", ax=ax3)
              ax3.set_title("Isı Haritası")
              plt.tight_layout()
              figs.append(fig3)
          except Exception as e:
               st.warning(f"PDF için ısı haritası oluşturulamadı: {e}")
     # Kinetik model (tüm uygun gruplar)
     kinetic_figs, kinetic_text = plot_kinetic(df) # Kinetik model grafikleri ve textini al
     figs.extend(kinetic_figs) # Kinetik grafikleri figs listesine ekle


     if figs:
         pdf_buffer = io.BytesIO()
         with matplotlib.backends.backend_pdf.PdfPages(pdf_buffer) as pdf:
             for fig in figs:
                 pdf.savefig(fig)
                 plt.close(fig) # Kaydettikten sonra figürü kapat
         pdf_buffer.seek(0)
         return pdf_buffer, kinetic_text # PDF bufferını ve kinetik texti döndür
     return None, None

# --- Buton Aksiyonları ---

if st.session_state.current_plot == "line":
    fig = plot_line(st.session_state.data)
    if fig:
        plot_placeholder.pyplot(fig)

if st.session_state.current_plot == "box":
    fig = plot_box(st.session_state.data)
    if fig:
        plot_placeholder.pyplot(fig)

if st.session_state.current_plot == "heatmap":
    fig = plot_heatmap(st.session_state.data)
    if fig:
        plot_placeholder.pyplot(fig)

if st.session_state.current_plot == "kinetic":
    figs, text_output = plot_kinetic(st.session_state.data)
    if figs:
        for fig in figs:
             plot_placeholder.pyplot(fig)
             plt.close(fig) # Streamlit'te gösterildikten sonra figürü kapat
    text_results_placeholder.text_area("Kinetik Model Sonuçları", value=text_output, height=150)

# PDF Kaydetme Trigger
if st.session_state.save_pdf_trigger:
    pdf_buffer, kinetic_text = save_pdf(st.session_state.data)
    if pdf_buffer:
        st.download_button(
             label="PDF İndir",
             data=pdf_buffer,
             file_name="hidrojel_analiz_grafikleri.pdf",
             mime="application/pdf"
        )
    if kinetic_text:
         text_results_placeholder.text_area("PDF Kaydedilirken Hesaplanan Kinetik Model Sonuçları", value=kinetic_text, height=150)
    st.session_state.save_pdf_trigger = False # Trigger'ı sıfırla 