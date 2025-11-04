# tests/conftest.py
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import io
from sklearn.datasets import make_classification, make_regression
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def mock_matplotlib():
    """Автоматически мокаем matplotlib для всех тестов"""
    with patch('modules.data_explorer.plt') as mock_plt:
        # Создаем мок для subplots который возвращает фигуру и оси
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        yield mock_plt


@pytest.fixture(autouse=True)
def mock_seaborn():
    """Автоматически мокаем seaborn для всех тестов"""
    with patch('modules.data_explorer.sns') as mock_sns:
        yield mock_sns


@pytest.fixture
def sample_dataframe():
    """Фикстура с тестовым DataFrame"""
    return pd.DataFrame({
        'numeric_1': [1, 2, 3, 4, 5, None, 7, 8, 9, 10],
        'numeric_2': [10.5, 20.3, 30.1, 40.7, 50.2, 60.8, 70.4, 80.9, 90.1, 100.5],
        'categorical': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
        'target_regression': [100, 200, 150, 250, 300, 180, 220, 280, 190, 260],
        'target_classification': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })


@pytest.fixture
def sample_csv_file(sample_dataframe):
    """Фикстура с временным CSV файлом"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        sample_dataframe.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def sample_excel_file(sample_dataframe):
    """Фикстура с временным Excel файлом"""
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.xlsx', delete=False) as f:
        with pd.ExcelWriter(f.name, engine='openpyxl') as writer:
            sample_dataframe.to_excel(writer, index=False, sheet_name='Sheet1')
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def regression_data():
    """Фикстура с данными для регрессии"""
    X, y = make_regression(
        n_samples=100,
        n_features=5,
        n_informative=3,
        noise=0.1,
        random_state=42
    )
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    return df


@pytest.fixture
def classification_data():
    """Фикстура с данными для классификации"""
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    return df


@pytest.fixture
def mock_streamlit():
    """Мок для Streamlit функций"""
    import sys
    from unittest.mock import MagicMock
    
    mock_st = MagicMock()
    
    # Моки для конкретных методов
    mock_st.header.return_value = None
    mock_st.subheader.return_value = None
    mock_st.write.return_value = None
    mock_st.info.return_value = None
    mock_st.warning.return_value = None
    mock_st.error.return_value = None
    mock_st.success.return_value = None
    mock_st.text.return_value = None
    mock_st.dataframe.return_value = None
    mock_st.table.return_value = None
    mock_st.markdown.return_value = None
    mock_st.pyplot.return_value = None
    
    # Моки для интерактивных элементов
    mock_st.file_uploader.return_value = None
    mock_st.selectbox.return_value = "test"
    mock_st.radio.return_value = "option1"
    mock_st.slider.return_value = 0.2
    mock_st.number_input.return_value = 0
    mock_st.button.return_value = False
    mock_st.multiselect.return_value = []
    mock_st.text_input.return_value = ""
    mock_st.checkbox.return_value = False
    
    # Заменяем streamlit в sys.modules
    sys.modules['streamlit'] = mock_st
    
    return mock_st


@pytest.fixture
def csv_bytes_io(sample_dataframe):
    """Фикстура с CSV данными в BytesIO"""
    csv_data = sample_dataframe.to_csv(index=False)
    return io.BytesIO(csv_data.encode('utf-8'))


@pytest.fixture
def excel_bytes_io(sample_dataframe):
    """Фикстура с Excel данными в BytesIO"""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        sample_dataframe.to_excel(writer, index=False, sheet_name='Sheet1')
    buffer.seek(0)
    return buffer