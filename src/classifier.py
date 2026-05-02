import pandas as pd
import re

class NCRClassifier:
    """Gelen veriyi işleme ve semantik sınıflandırma (skorlama) kurallarını yöneten sınıf."""
    
    KATEGORILER = {
        "Kaba İş": ["beton", "demir", "kalıp", "paspayı", "kolon", "kiriş", "döşeme", "çatlak", "segregasyon"],
        "İnce İş": ["seramik", "boya", "sıva", "alçı", "diş", "fayans", "derz", "parke", "şap", "kaplama"],
        "Doğrama": ["pencere", "kapı", "cam", "menteşe", "alüminyum", "pvc", "kulp", "fitil", "pervaz"]
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
        
        for kategori, kelimeler in self.KATEGORILER.items():
            # Tüm kelimeleri aramak için pattern oluşturuyoruz.
            # Her bir kelimenin metin içinde kaç defa geçtiğini sayıyoruz.
            pattern = '|'.join([re.escape(k) for k in kelimeler])
            scores_df[kategori] = text_series.str.count(pattern, flags=re.IGNORECASE)
        
        # En yüksek skora sahip kategoriyi bul
        # max(axis=1) ile her satırın en yüksek skorunu buluyoruz.
        max_scores = scores_df.max(axis=1)
        
        # Skorlarin en büyüğünü alan kategoriyi idxmax ile seçiyoruz
        best_categories = scores_df.idxmax(axis=1)
        
        # Eğer en yüksek skor 0 ise, hiçbir kelime bulunamamıştır -> "Diğer / Belirsiz"
        df['İş Kalemi'] = best_categories.where(max_scores > 0, "Diğer / Belirsiz")
            
        return df
