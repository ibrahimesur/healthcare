import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib.backends.backend_pdf
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import tkinter as tk

# --- SADECE ARAYÜZ KALSIN, OTOMATİK GRAFİK VE ANALİZ KODLARI SİLİNDİ ---

class HidrojelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hidrojel Analiz Uygulaması")
        self.data = []
        self.create_widgets()

    def create_widgets(self):
        frame = tb.Frame(self.root, padding=10)
        frame.pack(pady=5)
        tb.Label(frame, text="Hidrojel").grid(row=0, column=0)
        tb.Label(frame, text="pH").grid(row=0, column=1)
        tb.Label(frame, text="Zaman (saat)").grid(row=0, column=2)
        tb.Label(frame, text="Şişme Oranı (%)").grid(row=0, column=3)
        tb.Label(frame, text="İlaç Salımı (%)").grid(row=0, column=4)
        self.entry_hidrojel = tb.Combobox(frame, values=["Kitosan", "Poli(akrilik asit)", "Jelatin"])
        self.entry_hidrojel.grid(row=1, column=0)
        self.entry_ph = tb.Entry(frame, width=5)
        self.entry_ph.grid(row=1, column=1)
        self.entry_zaman = tb.Entry(frame, width=5)
        self.entry_zaman.grid(row=1, column=2)
        self.entry_sisme = tb.Entry(frame, width=7)
        self.entry_sisme.grid(row=1, column=3)
        self.entry_ilac = tb.Entry(frame, width=7)
        self.entry_ilac.grid(row=1, column=4)
        tb.Button(frame, text="Ekle", bootstyle=SUCCESS, command=self.add_row).grid(row=1, column=5, padx=5)
        tb.Button(frame, text="Verileri Kaydet", bootstyle=INFO, command=self.save_data).grid(row=1, column=6, padx=5)
        tb.Button(frame, text="Veri Yükle", bootstyle=INFO, command=self.load_data).grid(row=1, column=7, padx=5)

        self.tree = tb.Treeview(self.root, columns=("Hidrojel", "pH", "Zaman", "Şişme Oranı", "İlaç Salımı"), show="headings", height=8, bootstyle=PRIMARY)
        for col in ("Hidrojel", "pH", "Zaman", "Şişme Oranı", "İlaç Salımı"):
            self.tree.heading(col, text=col)
        self.tree.pack(pady=5)

        btn_frame = tb.Frame(self.root)
        btn_frame.pack(pady=5)
        tb.Button(btn_frame, text="Çizgi Grafiği", bootstyle=SECONDARY, command=self.plot_line).pack(side=LEFT, padx=2)
        tb.Button(btn_frame, text="Boxplot", bootstyle=SECONDARY, command=self.plot_box).pack(side=LEFT, padx=2)
        tb.Button(btn_frame, text="Isı Haritası", bootstyle=SECONDARY, command=self.plot_heatmap).pack(side=LEFT, padx=2)
        tb.Button(btn_frame, text="Kinetik Model", bootstyle=SECONDARY, command=self.plot_kinetic).pack(side=LEFT, padx=2)
        tb.Button(btn_frame, text="PDF Kaydet", bootstyle=WARNING, command=self.save_pdf).pack(side=LEFT, padx=2)
        tb.Button(btn_frame, text="Excel Kaydet", bootstyle=WARNING, command=self.save_excel).pack(side=LEFT, padx=2)

        self.fig_frame = tb.Frame(self.root)
        self.fig_frame.pack(fill="both", expand=True)
        self.text_frame = tb.Frame(self.root)
        self.text_frame.pack(fill="x", expand=False)
        self.text_box = tk.Text(self.text_frame, height=5)
        self.text_box.pack(fill="x", expand=True)

    def add_row(self):
        try:
            row = [
                self.entry_hidrojel.get(),
                float(self.entry_ph.get()),
                float(self.entry_zaman.get()),
                float(self.entry_sisme.get()),
                float(self.entry_ilac.get())
            ]
            self.data.append(row)
            self.tree.insert('', 'end', values=row)
            self.entry_hidrojel.set("")
            self.entry_ph.delete(0, 'end')
            self.entry_zaman.delete(0, 'end')
            self.entry_sisme.delete(0, 'end')
            self.entry_ilac.delete(0, 'end')
        except Exception as e:
            messagebox.showerror("Hata", f"Veri girişinde hata: {e}")

    def save_data(self):
        if not self.data:
            messagebox.showwarning("Uyarı", "Kaydedilecek veri yok!")
            return
        file = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
        if file:
            df = pd.DataFrame(self.data, columns=["Hidrojel", "pH", "Zaman", "Şişme Oranı", "İlaç Salımı"])
            df.to_excel(file, index=False)
            messagebox.showinfo("Başarılı", f"Veriler kaydedildi: {file}")

    def load_data(self):
        file = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if file:
            df = pd.read_excel(file)
            self.data = df.values.tolist()
            for i in self.tree.get_children():
                self.tree.delete(i)
            for row in self.data:
                self.tree.insert('', 'end', values=row)
            messagebox.showinfo("Başarılı", f"Veriler yüklendi: {file}")

    def clear_fig(self):
        for widget in self.fig_frame.winfo_children():
            widget.destroy()
        self.text_box.delete('1.0', 'end')

    def get_df(self):
        return pd.DataFrame(self.data, columns=["Hidrojel", "pH", "Zaman", "Şişme Oranı", "İlaç Salımı"])

    def plot_line(self):
        self.clear_fig()
        df = self.get_df()
        fig, ax = plt.subplots(figsize=(6,4))
        sns.lineplot(data=df, x="Zaman", y="İlaç Salımı", hue="Hidrojel", style="pH", markers=True, dashes=False, ax=ax)
        ax.set_title("Zamanla İlaç Salımı Değişimi")
        ax.set_ylabel("İlaç Salımı (%)")
        ax.set_xlabel("Zaman (saat)")
        canvas = FigureCanvasTkAgg(fig, master=self.fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def plot_box(self):
        self.clear_fig()
        df = self.get_df()
        fig, ax = plt.subplots(figsize=(6,4))
        sns.boxplot(data=df, x="Hidrojel", y="İlaç Salımı", hue="pH", ax=ax)
        ax.set_title("Farklı pH Değerlerinde İlaç Salımı Dağılımı (Boxplot)")
        ax.set_ylabel("İlaç Salımı (%)")
        ax.set_xlabel("Hidrojel")
        canvas = FigureCanvasTkAgg(fig, master=self.fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def plot_heatmap(self):
        self.clear_fig()
        df = self.get_df()
        pivot = df.pivot_table(index="Hidrojel", columns=["pH", "Zaman"], values="İlaç Salımı")
        fig, ax = plt.subplots(figsize=(8,4))
        sns.heatmap(pivot, annot=True, cmap="YlGnBu", ax=ax)
        ax.set_title("Hidrojel, pH ve Zaman'a Göre İlaç Salımı Isı Haritası")
        canvas = FigureCanvasTkAgg(fig, master=self.fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def plot_kinetic(self):
        self.clear_fig()
        df = self.get_df()
        results = []
        for (hidrojel, pH), grup in df.groupby(["Hidrojel", "pH"]):
            t = grup["Zaman"].values
            Q = grup["İlaç Salımı"].values
            if len(t) >= 2:
                def higuchi_model(t, k):
                    return k * np.sqrt(t)
                def korsmeyer_peppas_model(t, k, n):
                    return k * (t ** n)
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
                canvas = FigureCanvasTkAgg(fig, master=self.fig_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True)
                results.append(f"{hidrojel} - pH {pH}: Higuchi k={popt_h[0]:.2f}, Korsmeyer-Peppas k={popt_kp[0]:.2f}, n={popt_kp[1]:.2f}")
        self.text_box.insert('end', '\n'.join(results))

    def save_pdf(self):
        if not self.data:
            messagebox.showwarning("Uyarı", "Kaydedilecek veri yok!")
            return
        file = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
        if file:
            df = self.get_df()
            figs = []
            # Çizgi grafiği
            fig1, ax1 = plt.subplots(figsize=(6,4))
            sns.lineplot(data=df, x="Zaman", y="İlaç Salımı", hue="Hidrojel", style="pH", markers=True, dashes=False, ax=ax1)
            ax1.set_title("Zamanla İlaç Salımı Değişimi")
            figs.append(fig1)
            # Boxplot
            fig2, ax2 = plt.subplots(figsize=(6,4))
            sns.boxplot(data=df, x="Hidrojel", y="İlaç Salımı", hue="pH", ax=ax2)
            ax2.set_title("Boxplot")
            figs.append(fig2)
            # Heatmap
            fig3, ax3 = plt.subplots(figsize=(8,4))
            pivot = df.pivot_table(index="Hidrojel", columns=["pH", "Zaman"], values="İlaç Salımı")
            sns.heatmap(pivot, annot=True, cmap="YlGnBu", ax=ax3)
            ax3.set_title("Isı Haritası")
            figs.append(fig3)
            # Kinetik model (ilk uygun grup)
            for (hidrojel, pH), grup in df.groupby(["Hidrojel", "pH"]):
                t = grup["Zaman"].values
                Q = grup["İlaç Salımı"].values
                if len(t) >= 2:
                    def higuchi_model(t, k):
                        return k * np.sqrt(t)
                    def korsmeyer_peppas_model(t, k, n):
                        return k * (t ** n)
                    popt_h, _ = curve_fit(higuchi_model, t, Q, maxfev=10000)
                    popt_kp, _ = curve_fit(korsmeyer_peppas_model, t, Q, bounds=(0, [np.inf, 1]), maxfev=10000)
                    fig4, ax4 = plt.subplots(figsize=(5,3))
                    ax4.scatter(t, Q, label="Deneysel")
                    ax4.plot(t, higuchi_model(t, *popt_h), label=f"Higuchi (k={popt_h[0]:.2f})")
                    ax4.plot(t, korsmeyer_peppas_model(t, *popt_kp), label=f"Korsmeyer-Peppas (k={popt_kp[0]:.2f}, n={popt_kp[1]:.2f})")
                    ax4.set_title(f"{hidrojel} - pH {pH} Kinetik Model")
                    ax4.set_xlabel("Zaman (saat)")
                    ax4.set_ylabel("İlaç Salımı (%)")
                    ax4.legend()
                    figs.append(fig4)
                    break
            pdf = matplotlib.backends.backend_pdf.PdfPages(file)
            for fig in figs:
                pdf.savefig(fig)
            pdf.close()
            messagebox.showinfo("Başarılı", f"PDF kaydedildi: {file}")

    def save_excel(self):
        if not self.data:
            messagebox.showwarning("Uyarı", "Kaydedilecek veri yok!")
            return
        file = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
        if file:
            df = self.get_df()
            df.to_excel(file, index=False)
            messagebox.showinfo("Başarılı", f"Excel kaydedildi: {file}")

if __name__ == "__main__":
    app = tb.Window(themename="superhero")
    HidrojelApp(app)
    app.mainloop() 