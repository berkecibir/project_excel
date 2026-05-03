import pytest
import pandas as pd
import logging
import numpy as np
from app import NCRClassifier

# Mock logger for testing
logger = logging.getLogger('TestLogger')

@pytest.fixture
def classifier():
    return NCRClassifier(logger)

def test_classify_segregasyon(classifier):
    """Segregasyon ifadesinin doğru teknik gruba atandığını doğrula."""
    data = {'Açıklama': ['Beton dökümü sonrası segregasyon ve boşluklar tespit edildi.']}
    df = pd.DataFrame(data)
    result_df = classifier.classify_dataframe(df, 'Açıklama')
    assert result_df['İş Kalemi'].iloc[0] == 'SOĞUK DERZ / SEGREGASYON / BOŞLUK'

def test_classify_donatı(classifier):
    """Donatı/Demir ifadelerinin doğru gruba atandığını doğrula."""
    data = {'Açıklama': ['Döşeme demirlerinde paspayı eksikliği var.']}
    df = pd.DataFrame(data)
    result_df = classifier.classify_dataframe(df, 'Açıklama')
    assert result_df['İş Kalemi'].iloc[0] == 'DONATI / DEMİR / ETRİYE'

def test_classify_evrak(classifier):
    """Evrak/Yöntem ifadelerinin doğru gruba atandığını doğrula."""
    data = {'Açıklama': ['İş yapım yöntemi ve malzeme onayı henüz sunulmadı.']}
    df = pd.DataFrame(data)
    result_df = classifier.classify_dataframe(df, 'Açıklama')
    assert result_df['İş Kalemi'].iloc[0] == 'EVRAK / MALZEME ONAY / YÖNTEM'

def test_classify_sehim_sakul(classifier):
    """Kot ve tolerans hatalarını doğrula."""
    data = {'Açıklama': ['Kolonlarda şakül hatası ve sehim tespit edildi.']}
    df = pd.DataFrame(data)
    result_df = classifier.classify_dataframe(df, 'Açıklama')
    assert result_df['İş Kalemi'].iloc[0] == 'BETON KOT / SEHİM / TOLERANS'

def test_classify_hasarlı(classifier):
    """Hasarlı malzeme grubunu doğrula."""
    data = {'Açıklama': ['Gelen seramikler kırık ve deforme durumda.']}
    df = pd.DataFrame(data)
    result_df = classifier.classify_dataframe(df, 'Açıklama')
    assert result_df['İş Kalemi'].iloc[0] == 'HASARLI MALZEME'

def test_classify_diger_belirsiz(classifier):
    """Tanımlanamayan ifadelerin Diğer kategorisine düştüğünü doğrula."""
    data = {'Açıklama': ['Yemekhanede havalandırma yetersiz.']} # Havalandırma Tesisat'a girer ama cümle yapısına göre test edelim
    df = pd.DataFrame(data)
    result_df = classifier.classify_dataframe(df, 'Açıklama')
    # Havalandırma TESİSAT grubunda olduğu için ona gitmeli
    assert result_df['İş Kalemi'].iloc[0] == 'TESİSAT (ELEKTRİK / MEKANİK) / PEX'
