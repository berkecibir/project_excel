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
        "EVRAK / MALZEME ONAY / YÖNTEM": ["yapım yöntemi", "iş yapım yöntemi", "evrak", "shopdraw", "shop drawing", "malzeme onay", "malzeme onayı", "malzeme sunum", "malzeme girdi", "teknik şartname", "uygulama yöntemi", "numune", "muayene", "test"],
        "TUĞLA / DUVAR / BİMS / HATIL / KÖPÜK": ["tuğla", "bims", "duvar", "kama", "köpük", "hatıl", "lento", "lentolu"],
        "BETON KOT / SEHİM / TOLERANS": ["sehim", "şakül", "aks okuma", "tolerans", "tölerans", "deplase", "kanat açma", "kanat açmaları", "beton kot okuması hata", "beton okuma kot", "beton kot okuma"],
        "SOĞUK DERZ / SEGREGASYON / BOŞLUK": ["soğuk derz", "segregasyon", "boşluk", "peteklenme"],
        "DONATI / DEMİR / ETRİYE": ["donatı", "demir", "etriye", "filiz", "çiroz", "ankraj", "paspayı"],
        "KALIP / İSKELE / DİREKLEME": ["kalıp", "kalıb", "iskele", "direkleme", "plywood", "diş", "dis"],
        "PERDAH / ÇATLAK": ["perdah", "çatlak", "kılcal", "pürüz"],
        "ALÇI / SIVA / BOYA / DEKORATİF": ["alçı", "sıva", "boya", "dekoratif", "saten", "kartonpiyer"],
        "CEPHE / DOĞRAMA / TAŞ YÜNÜ / KÜPTAŞ": ["cephe", "doğrama", "küptaş", "taş yünü", "taşyünü", "tasyünü", "kör kasa", "körkasa"],
        "TESİSAT (ELEKTRİK / MEKANİK) / PEX": ["tesisat", "pex", "boru", "klima", "havalandırma", "yangın dolap", "yangın dolabı", "pano", "elektrik", "kablo", "doğalgaz", "dogalgaz"],
        "YALITIM / İZOLASYON": ["drenaj", "su alması", "su dolu", "su tahliye", "yalıtım", "izolasyon", "su yalıtım"],
        "TEMİZLİK / KORUMA / KİR": ["temizlik", "koruma", "kir", "moloz", "toz", "şerbet", "artık", "kalıntı"],
        "ŞAP / KAPLAMA / TAŞ / DERZ": ["şap", "döşeme", "kaplama", "taş", "seramik", "derz"],
        "HASARLI MALZEME": ["hasarlı", "kırık", "deformasyon", "deforme", "zarar görmüş", "zarar"],
        "PAH ÇITASI EKSİKLİĞİ": ["pah çıtası", "pah hatası", "pah eksik"],
        "BETON İŞÇİLİK HATA": ["beton işçilik", "yüksek dökül", "kötü beton", "düzensiz beton"],
        "BETON KÜR/BAKIM EKSİKLİĞİ": ["kür", "sulama eksik", "beton bakım"],
        "PROFİL İŞLERİ": ["kutu profil"]
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
        # İSG ağırlığı yüksek tutulur: "kolon kalıp sökümünde baretsiz çalışıyor" gibi
        # cümlelerde Kaba İş kelimeleri (kolon+kalıp=2) ile İSG çakışmaması için
        # İSG ağırlığı 2.5 yapıldı (1 İSG kelimesi >= 2.5 skor > 2 Kaba İş skoru).
        AGIRLIKLAR = {
            "İnce İş": 1.6,
            "İş Güvenliği (İSG)": 2.5
        }
        
        for kategori, kelimeler in self.KATEGORILER.items():
            # Tüm kelimeleri aramak için pattern oluşturuyoruz.
            # NOT: Python regex \b ASCII tabanlıdır, Türkçe karakterlerde çalışmaz.
            # Bu nedenle düz substring eşleşmesi kullanılır.
            # İSG için 'baretsiz', 'kemersiz' gibi olumsuz türevler listeye eklenmiştir.
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
        Bu araç, yüklediğiniz Uygunsuzluk Raporu (NCR) Excel dosyasındaki **Açıklama** sütununu analiz ederek uygunsuzlukları sınıflandırır.
        """)

    def _find_header_row(self, uploaded_file) -> int:
        """
        Excel dosyasının ilk 10 satırını tarayarak gerçek başlık satırını bulur.
        'Açıklama' veya benzeri bir sütun adı içeren satır numarasını döndürür.
        Bulamazsa 0 (varsayılan) döndürür.
        """
        HEDEF_KELIMELER = {"açıklama", "aciklama", "description", "uygunsuzluk"}
        try:
            df_raw = pd.read_excel(uploaded_file, engine="calamine", header=None, nrows=10)
            for i, row in df_raw.iterrows():
                for cell in row:
                    if isinstance(cell, str) and any(kw in cell.lower() for kw in HEDEF_KELIMELER):
                        self.logger.info(f"Gerçek başlık satırı {i}. satırda bulundu.")
                        return i
        except Exception as e:
            self.logger.warning(f"Başlık satırı taraması başarısız: {e}")
        return 0

    def _temizle_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Okunan DataFrame'i aşağıdaki işlemlerle temizler:
        1. Sütun adlarındaki satır sonu (\\n) karakterlerini boşluğa çevirir
        2. Excel formula hatalarını (#REF!, #VALUE! vb.) NaN ile değiştirir
        3. Tamamen boş satırları siler
        """
        # 1. Sütun adlarını normalize et: \n → boşluk, başını sonunu kırp
        df.columns = [
            ' '.join(str(c).split()).strip() if not isinstance(c, str)
            else ' '.join(c.split()).strip()
            for c in df.columns
        ]

        # 2. Excel formula hatalarını temizle
        EXCEL_HATALARI = ['#REF!', '#VALUE!', '#NAME?', '#DIV/0!',
                          '#N/A', '#NULL!', '#NUM!', '#ERROR!', '#BÖLÜ/0!']
        df = df.replace(EXCEL_HATALARI, np.nan)

        # 3. Tamamen boş satırları sil (başlık ile veri arasındaki boş satır vb.)
        df = df.dropna(how='all').reset_index(drop=True)

        return df

    def process_file(self, uploaded_file):
        try:
            self.logger.info(f"Yeni dosya işleniyor: {uploaded_file.name}")

            # --- ADIM 1: Gerçek başlık satırını bul ---
            header_row = self._find_header_row(uploaded_file)

            # --- ADIM 2: Dosyayı doğru başlık satırıyla oku ---
            with st.spinner("Excel dosyası okunuyor, lütfen bekleyin..."):
                df = pd.read_excel(uploaded_file, engine="calamine", header=header_row)

            # --- ADIM 3: Veriyi temizle (sütun adları, #REF! hataları, boş satırlar) ---
            df = self._temizle_dataframe(df)

            kolonlar = df.columns.tolist()
            self.logger.info(f"Temiz sütunlar (header={header_row}): {kolonlar}")

            # --- ADIM 4: Açıklama sütununu belirle ---
            aciklama_kolonu = "Açıklama"
            if aciklama_kolonu not in kolonlar:
                olasi_kolonlar = [
                    k for k in kolonlar
                    if any(kw in k.lower() for kw in ["açıklama", "aciklama", "description", "uygunsuzluk"])
                ]
                st.warning("Excel dosyanızda 'Açıklama' adında bir sütun bulunamadı. Lütfen analiz edilecek sütunu seçin.")
                aciklama_kolonu = st.selectbox(
                    "Uygunsuzluk Açıklaması Sütunu",
                    options=kolonlar,
                    index=kolonlar.index(olasi_kolonlar[0]) if olasi_kolonlar else 0
                )

            if aciklama_kolonu:
                # --- ADIM 5: Açıklaması boş/NaN olan satırları düşür ---
                onceki_sayi = len(df)
                df = df[df[aciklama_kolonu].notna()].copy()
                df = df[df[aciklama_kolonu].astype(str).str.strip() != ''].copy()
                df = df[df[aciklama_kolonu].astype(str).str.strip() != 'nan'].copy()
                atilan = onceki_sayi - len(df)
                if atilan > 0:
                    self.logger.info(f"Açıklaması boş {atilan} satır atıldı.")

                # --- ADIM 6: Sınıflandır ---
                with st.spinner("Veriler analiz ediliyor..."):
                    df = self.classifier.classify_dataframe(df, aciklama_kolonu)

                belirsiz_df = df[df['İş Kalemi'] == "Diğer / Belirsiz"]
                belirsizler = belirsiz_df[aciklama_kolonu].dropna().unique() if not belirsiz_df.empty else []
                if len(belirsizler) > 0:
                    self.logger.warning(f"Sınıflandırılamayan {len(belirsizler)} farklı açıklama bulundu.")

                self.logger.info(f"Analiz tamamlandı. Toplam kayıt: {len(df)}")
                st.success(f"✅ Sınıflandırma tamamlandı! **{len(df)}** kayıt işlendi.")

                # İmalat Sınıfı sütununu bul (temizlenmiş adıyla)
                imalat_kolonu = next(
                    (k for k in df.columns if "İmalat" in k and "Sınıf" in k),
                    None
                )
                self.render_filters_and_results(df, aciklama_kolonu, belirsizler, imalat_kolonu)


        except Exception as e:
            self.logger.error(f"Hata oluştu: {str(e)}", exc_info=True)
            st.error(f"Dosya okunurken bir hata oluştu: {e}")
            st.info("Lütfen geçerli bir Excel dosyası yüklediğinizden emin olun.")


    def render_filters_and_results(self, df, aciklama_kolonu, belirsizler, imalat_kolonu=None):
        """
        Dinamik filtreleme paneli: Proje, Alt Yüklenici, İmalat Sınıfı ve Durum.
        """
        st.subheader("📊 Gelişmiş Filtreleme Paneli")
        
        # 1. Sütun Tespitleri
        proje_kolonu = next((c for c in df.columns if "Proje" in c), None)
        yuklenici_kolonu = next((c for c in df.columns if "Yüklenici" in c), None)
        durum_kolonu = next((c for c in df.columns if "Durum" in c), None)
        filtre_kolonu = imalat_kolonu if imalat_kolonu else 'İş Kalemi'

        # --- FİLTRELER (Varsayılan Boş) ---
        col1, col2 = st.columns(2)
        
        secilen_projeler = []
        if proje_kolonu:
            p_options = sorted(df[proje_kolonu].dropna().unique().tolist())
            secilen_projeler = col1.pills("🏗️ Projeler:", options=p_options, selection_mode="multi", key="pill_proje")

        secilen_yukleniciler = []
        if yuklenici_kolonu:
            y_options = sorted(df[yuklenici_kolonu].dropna().unique().tolist())
            secilen_yukleniciler = col2.pills("👷 Alt Yükleniciler:", options=y_options, selection_mode="multi", key="pill_yuklenici")

        st.markdown("---")
        col3, col4 = st.columns([2, 1])
        
        options = sorted(df[filtre_kolonu].dropna().astype(str).str.strip().str.upper().unique().tolist())
        options = [o for o in options if o and o != 'NAN']
        secilen_kategoriler = col3.pills("🧱 İmalat Sınıfları:", options=options, selection_mode="multi", key="pill_imalat_sinifi")

        secilen_durumlar = []
        if durum_kolonu:
            durum_options = ["Açık", "Kapalı"]
            secilen_durumlar = col4.pills("🚨 NCR Durumu:", options=durum_options, selection_mode="multi", key="pill_ncr_durumu")

        # --- Filtre Uygulama Mantığı ---
        # Eğer hiçbir seçim yapılmadıysa veri gösterme
        if not (secilen_projeler or secilen_yukleniciler or secilen_kategoriler or secilen_durumlar):
            st.info("💡 Devam etmek için lütfen yukarıdaki filtrelerden en az bir Proje, Yüklenici veya Durum seçin.")
            return

        df_filtre = df.copy()
        if secilen_projeler:
            df_filtre = df_filtre[df_filtre[proje_kolonu].isin(secilen_projeler)]
        if secilen_yukleniciler:
            df_filtre = df_filtre[df_filtre[yuklenici_kolonu].isin(secilen_yukleniciler)]
        if secilen_kategoriler:
            df_filtre['_tmp'] = df_filtre[filtre_kolonu].astype(str).str.strip().str.upper()
            df_filtre = df_filtre[df_filtre['_tmp'].isin(secilen_kategoriler)]
            df_filtre = df_filtre.drop(columns=['_tmp'])
        if secilen_durumlar:
            mask = df_filtre[durum_kolonu].astype(str).apply(lambda x: any(d.lower() in x.lower() for d in secilen_durumlar))
            df_filtre = df_filtre[mask]

        if df_filtre.empty:
            st.warning("⚠️ Seçtiğiniz kriterlere uygun veri bulunamadı.")
            return

        # --- TEK TABLO GÖSTERİMİ ---
        st.markdown(f"### 📋 Analiz Sonuçları ({len(df_filtre)} Kayıt)")
        
        istenen_sutunlar = ['Proje', 'Alt Yüklenici', 'Disiplin', 'İmalat Sınıfı', 'Açıklama', 'Ncr Durumu', 'İş Kalemi']
        available_cols = []
        for target in istenen_sutunlar:
            match = next((c for c in df.columns if target.lower() in c.lower()), None)
            if match: available_cols.append(match)
        
        display_df = df_filtre[available_cols] if available_cols else df_filtre
        st.dataframe(display_df, use_container_width=True)

        # İstatistikler ve İndirme
        c1, c2 = st.columns([3, 1])
        with c2:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_filtre.to_excel(writer, index=False, sheet_name='Analiz_Sonuclari')
            st.download_button("📥 Excel Olarak İndir", output.getvalue(), "ncr_raporu.xlsx", use_container_width=True)
        
        with c1:
            st.caption(f"Toplam {len(df)} kayıt içerisinden {len(df_filtre)} adet kayıt filtrelendi.")

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
