import sys
import subprocess

# Streamlit uygulamasını `python app.py` ile çalıştırsalar bile,
# otomatik olarak `streamlit run app.py` formatına çeviririz.
if __name__ == "__main__":
    if "streamlit" not in sys.modules:
        print("Uyarı: Uygulama doğrudan 'python app.py' olarak başlatıldı.")
        print("Arka planda 'streamlit run app.py' olarak yeniden başlatılıyor...")
        # Kendi dosya yolumuzu alıyoruz
        script_path = sys.argv[0]
        # Streamlit modülünü subprocess ile çalıştırıyoruz
        sys.exit(subprocess.run([sys.executable, "-m", "streamlit", "run", script_path]).returncode)
    
    # Gerçek Streamlit uygulaması çalıştırılıyor
    from src.ui import NCRAppUI
    app = NCRAppUI()
    app.run()
