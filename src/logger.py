import logging
import sys

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
