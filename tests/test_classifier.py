import pytest
import pandas as pd
import logging
from src.classifier import NCRClassifier

# Mock logger for testing
logger = logging.getLogger('TestLogger')

@pytest.fixture
def classifier():
    return NCRClassifier(logger)

def test_classify_kaba_is(classifier):
    data = {'Açıklama': ['Beton dökümü sırasında segregasyon oluştu.', 'Kolon demirleri eksik.']}
    df = pd.DataFrame(data)
    
    result_df = classifier.classify_dataframe(df, 'Açıklama')
    
    assert result_df['İş Kalemi'].iloc[0] == 'Kaba İş'
    assert result_df['İş Kalemi'].iloc[1] == 'Kaba İş'

def test_classify_ince_is(classifier):
    data = {'Açıklama': ['Seramik kaplamada diş yapmış.', 'Duvar boyası dökülüyor.']}
    df = pd.DataFrame(data)
    
    result_df = classifier.classify_dataframe(df, 'Açıklama')
    
    assert result_df['İş Kalemi'].iloc[0] == 'İnce İş'
    assert result_df['İş Kalemi'].iloc[1] == 'İnce İş'

def test_classify_dograma(classifier):
    data = {'Açıklama': ['Pencere camı kırık gelmiş.', 'Kapı menteşesi ayarsız.']}
    df = pd.DataFrame(data)
    
    result_df = classifier.classify_dataframe(df, 'Açıklama')
    
    assert result_df['İş Kalemi'].iloc[0] == 'Doğrama'
    assert result_df['İş Kalemi'].iloc[1] == 'Doğrama'

def test_classify_diger_belirsiz(classifier):
    data = {'Açıklama': ['İşçiler baret takmıyor.', 'Şantiye alanı çok dağınık.']}
    df = pd.DataFrame(data)
    
    result_df = classifier.classify_dataframe(df, 'Açıklama')
    
    assert result_df['İş Kalemi'].iloc[0] == 'Diğer / Belirsiz'
    assert result_df['İş Kalemi'].iloc[1] == 'Diğer / Belirsiz'

def test_classify_empty_and_nan(classifier):
    data = {'Açıklama': ['', None, float('nan')]}
    df = pd.DataFrame(data)
    
    result_df = classifier.classify_dataframe(df, 'Açıklama')
    
    assert result_df['İş Kalemi'].iloc[0] == 'Diğer / Belirsiz'
    assert result_df['İş Kalemi'].iloc[1] == 'Diğer / Belirsiz'
    assert result_df['İş Kalemi'].iloc[2] == 'Diğer / Belirsiz'

def test_classify_case_insensitivity(classifier):
    data = {'Açıklama': ['BETON çok sulu.', 'SERAMİK KIRIK.']}
    df = pd.DataFrame(data)
    
    result_df = classifier.classify_dataframe(df, 'Açıklama')
    
    assert result_df['İş Kalemi'].iloc[0] == 'Kaba İş'
    assert result_df['İş Kalemi'].iloc[1] == 'İnce İş'

def test_semantic_scoring(classifier):
    # Test text containing both kaba ("beton", "kolon") and ince ("sıva") keywords.
    # Score for Kaba = 2, Score for Ince = 1 -> Result should be "Kaba İş"
    data = {'Açıklama': ['Beton kolon yüzeyinde ince sıva bozuklukları var.']}
    df = pd.DataFrame(data)
    
    result_df = classifier.classify_dataframe(df, 'Açıklama')
    assert result_df['İş Kalemi'].iloc[0] == 'Kaba İş'
