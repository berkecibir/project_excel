import streamlit as st
import pandas as pd
import io
import logging
import os
import sys
import re
import numpy as np

class LoggerSetup:
    """Uygulama loglama ayarlarını yöneten sınıf."""
    @staticmethod
    def setup():
        logger = logging.getLogger('NCRLogger')
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            
            # Dosyaya yazdırma
            file_handler = logging.FileHandler('ncr_app.log', encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            # Terminale (Console) yazdırma
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)
            
        return logger

class NCRClassifier:
    """Gelen veriyi işleme ve sınıflandırma kurallarını yöneten sınıf."""
    
    KATEGORILER = {
        "Kaba İş": ["beton", "demir", "kalıp", "paspayı", "kolon", "kiriş", "döşeme", "çatlak", "segregasyon"],
        "İnce İş": ["seramik", "boya", "sıva", "alçı", "diş yapmış", "fayans", "derz", "parke", "şap", "kaplama"],
        "Doğrama": ["pencere", "kapı", "cam", "menteşe", "alüminyum", "pvc", "kulp", "fitil", "pervaz"],
        "İş Güvenliği (İSG)": ["güvenlik", "levha", "baret", "kemer", "file", "korkuluk", "isg", "uyarı", "şantiye", "iskele"],
        "Elektrik İşleri": ["elektrik", "kablo", "pano", "sigorta", "priz", "anahtar", "aydınlatma", "armatür", "tava"]
    }
    
    def __init__(self, logger):
        self.logger = logger
        
    def classify_dataframe(self, df, column_name):
        """Vektörel (Regex) skorlama kullanarak DataFrame'i yüksek performansla sınıflandırır."""
        self.logger.info(f"'{column_name}' sütunu için skor tabanlı sınıflandırma başlatılıyor. Toplam satır: {len(df)}")
        
        # Skorları tutacağımız DataFrame (aynı index ile)
        scores_df = pd.DataFrame(index=df.index)
        
        # Sütunu string olarak al
        text_series = df[column_name].astype(str)
        
        # Bazı kategorilerin önceliği (ağırlığı) daha fazladır.
        # Örneğin: İnce İş detaydır, "Doğrama" ve "İnce" çakışırsa İnce kazansın diye ağırlığı 1.6 yaptık.
        AGIRLIKLAR = {
            "İnce İş": 1.6,
            "İş Güvenliği (İSG)": 2.0
        }
        
        for kategori, kelimeler in self.KATEGORILER.items():
            # Tüm kelimeleri aramak için pattern oluşturuyoruz.
            # Her bir kelimenin metin içinde kaç defa geçtiğini sayıyoruz.
            pattern = '|'.join([re.escape(k) for k in kelimeler])
            count = text_series.str.count(pattern, flags=re.IGNORECASE)
            
            # Kategoriye özel ağırlık çarpanını uygula (yoksa 1.0 ile çarp)
            weight = AGIRLIKLAR.get(kategori, 1.0)
            scores_df[kategori] = count * weight
        
        # En yüksek skora sahip kategoriyi bul
        # max(axis=1) ile her satırın en yüksek skorunu buluyoruz.
        max_scores = scores_df.max(axis=1)
        
        # Skorlarin en büyüğünü alan kategoriyi idxmax ile seçiyoruz
        best_categories = scores_df.idxmax(axis=1)
        
        # Eğer en yüksek skor 0 ise, hiçbir kelime bulunamamıştır -> "Diğer / Belirsiz"
        df['İş Kalemi'] = best_categories.where(max_scores > 0, "Diğer / Belirsiz")
            
        return df

class NCRAppUI:
    """Uygulamanın Streamlit arayüzünü çizen ve işlemleri koordine eden ana sınıf."""
    
    def __init__(self):
        self.logger = LoggerSetup.setup()
        self.classifier = NCRClassifier(self.logger)
        
    def render_header(self):
        st.set_page_config(page_title="NCR Analiz Aracı", page_icon="🏗️", layout="wide")
        st.title("🏗️ İnşaat Mühendisleri İçin NCR (Uygunsuzluk) Analiz Aracı")
        st.markdown("""
        Bu araç, yüklediğiniz Uygunsuzluk Raporu (NCR) Excel dosyasındaki **Açıklama** sütununu analiz ederek uygunsuzlukları **Kaba İş**, **İnce İş** veya **Doğrama** olarak sınıflandırır.
        """)

    def process_file(self, uploaded_file):
        try:
            self.logger.info(f"Yeni dosya işleniyor: {uploaded_file.name}")
            with st.spinner("Excel dosyası okunuyor, lütfen bekleyin (Büyük dosyalarda biraz zaman alabilir)..."):
                df = pd.read_excel(uploaded_file, engine="calamine")
            kolonlar = df.columns.tolist()
            
            aciklama_kolonu = "Açıklama"
            if aciklama_kolonu not in kolonlar:
                olasi_kolonlar = [k for k in kolonlar if "açıklama" in k.lower() or "aciklama" in k.lower() or "description" in k.lower() or "uygunsuzluk" in k.lower()]
                st.warning(f"Excel dosyanızda 'Açıklama' adında bir sütun bulunamadı. Lütfen analiz edilecek sütunu seçin.")
                aciklama_kolonu = st.selectbox("Uygunsuzluk Açıklaması Sütunu", options=kolonlar, index=kolonlar.index(olasi_kolonlar[0]) if olasi_kolonlar else 0)
            
            if aciklama_kolonu:
                with st.spinner("Veriler analiz ediliyor..."):
                    df = self.classifier.classify_dataframe(df, aciklama_kolonu)
                
                # Sınıflandırılamayanları bul
                belirsiz_df = df[df['İş Kalemi'] == "Diğer / Belirsiz"]
                belirsizler = belirsiz_df[aciklama_kolonu].dropna().unique() if not belirsiz_df.empty else []
                if len(belirsizler) > 0:
                    self.logger.warning(f"Sınıflandırılamayan {len(belirsizler)} farklı açıklama bulundu.")
                
                self.logger.info(f"Analiz tamamlandı. Toplam kayıt: {len(df)}")

                col_success, col_reset = st.columns([8, 2])
                col_success.success(f"✅ Sınıflandırma tamamlandı! Toplam {len(df)} kayıt analiz edildi.")
                if col_reset.button("🔄 Yeni Dosya Yükle", type="secondary", use_container_width=True):
                    st.session_state.clear()
                    st.rerun()
                
                self.render_filters_and_results(df, aciklama_kolonu, belirsizler)
                
        except Exception as e:
            self.logger.error(f"Hata oluştu: {str(e)}", exc_info=True)
            st.error(f"Dosya okunurken bir hata oluştu: {e}")
            st.info("Lütfen geçerli bir Excel dosyası yüklediğinizden emin olun.")

    def render_filters_and_results(self, df, aciklama_kolonu, belirsizler):


        st.subheader("Verileri Filtrele")
        options = ["Kaba İş", "İnce İş", "Doğrama", "İş Güvenliği (İSG)", "Elektrik İşleri", "Diğer / Belirsiz"]
        
        if "pill_kategoriler" not in st.session_state:
            st.session_state.pill_kategoriler = options

        col_b1, col_b2, _ = st.columns([2, 2, 8])
        if col_b1.button("✅ Tümünü Seç", use_container_width=True):
            st.session_state.pill_kategoriler = options
            st.rerun()
            
        if col_b2.button("🗑️ Tümünü Temizle", use_container_width=True):
            st.session_state.pill_kategoriler = []
            st.rerun()

        secilen_kategoriler = st.pills(
            "Görüntülemek istediğiniz iş kalemlerini seçin:",
            options=options,
            selection_mode="multi",
            key="pill_kategoriler"
        )
        
        if not secilen_kategoriler:
            st.warning("Lütfen görüntülemek için en az bir iş kalemi seçin.")
            return

        filtered_df = df[df['İş Kalemi'].isin(secilen_kategoriler)].copy()
        
        # Kayıt Numaralarını 1'den başlatma (Kullanıcı talebi)
        filtered_df.index = np.arange(1, len(filtered_df) + 1)
        filtered_df.index.name = "Kayıt No"
        
        st.dataframe(filtered_df, use_container_width=True)
        
        st.subheader("Genel İstatistikler")
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        kategori_sayilari = df['İş Kalemi'].value_counts()
        
        col1.metric("Toplam", len(df))
        col2.metric("Kaba İş", kategori_sayilari.get("Kaba İş", 0))
        col3.metric("İnce İş", kategori_sayilari.get("İnce İş", 0))
        col4.metric("Doğrama", kategori_sayilari.get("Doğrama", 0))
        col5.metric("İSG", kategori_sayilari.get("İş Güvenliği (İSG)", 0))
        col6.metric("Elektrik", kategori_sayilari.get("Elektrik İşleri", 0))
        col7.metric("Belirsiz", kategori_sayilari.get("Diğer / Belirsiz", 0))
        
        
        st.subheader("Sonuçları İndir")
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            filtered_df.to_excel(writer, index=True, sheet_name='Filtrelenmiş_NCR')
        
        excel_data = output.getvalue()
        st.download_button(
            label="📥 Filtrelenmiş Veriyi Excel Olarak İndir",
            data=excel_data,
            file_name="filtrelenmis_ncr_raporu.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        self.render_developer_tools(belirsizler)

    def render_developer_tools(self, belirsizler):
        st.markdown("---")
        st.subheader("🛠️ Geliştirici / Test Analiz Ekranı")
        with st.expander("Uygulama Logları ve Geliştirme Önerileri (Tıklayıp Genişletin)"):
            st.write("Bu alan, test yaparken uygulamanın nasıl çalıştığını incelemek ve eksikleri bulmak için eklenmiştir.")
            
            tab1, tab2 = st.tabs(["Tanınmayan İfadeler (Geliştirme İçin)", "Sistem Logları"])
            
            with tab1:
                if len(belirsizler) > 0:
                    st.warning(f"Sınıflandırılamayan (Diğer / Belirsiz) {len(belirsizler)} farklı ifade bulundu. Bu ifadeleri analiz ederek 'KATEGORILER' sözlüğüne ekleyebiliriz.")
                    st.dataframe(pd.DataFrame(belirsizler, columns=["Tanınmayan Açıklamalar"]), use_container_width=True)
                else:
                    st.success("Tüm açıklamalar başarıyla bir kategoriye atandı! Tanınmayan ifade yok.")
                    
            with tab2:
                if os.path.exists('ncr_app.log'):
                    with open('ncr_app.log', 'r', encoding='utf-8') as f:
                        log_icerik = f.read()
                    st.text_area("Log Kayıtları (Dosyadan)", value=log_icerik, height=300)
                else:
                    st.info("Henüz log kaydı bulunmuyor.")

    def run(self):
        self.render_header()
        uploaded_file = st.file_uploader("Lütfen NCR Excel dosyanızı yükleyin", type=["xlsx", "xls"])
        if uploaded_file is not None:
            self.process_file(uploaded_file)

if __name__ == "__main__":
    app = NCRAppUI()
    app.run()
