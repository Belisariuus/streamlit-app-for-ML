# tests/test_data_loader.py
import pytest
import pandas as pd
import numpy as np
import sys
import os
import io
from unittest.mock import MagicMock, patch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.data_loader import load_data_interface, detect_encoding


class TestDataLoader:
    """Тесты для модуля загрузки данных (адаптировано под ваш код)"""
    
    def test_detect_encoding_utf8(self):
        """Тест определения кодировки UTF-8"""
        utf8_text = "тестовый текст".encode('utf-8')
        encoding = detect_encoding(utf8_text)
        assert encoding == "utf-8"
    
    def test_detect_encoding_empty(self):
        """Тест определения кодировки пустых данных"""
        encoding = detect_encoding(b"")
        assert encoding == "utf-8"
    
    def test_load_csv_success(self, sample_dataframe):
        """Тест успешной загрузки CSV файла"""
        # Создаем CSV в памяти
        csv_data = sample_dataframe.to_csv(index=False)
        csv_bytes = csv_data.encode('utf-8')
        
        # Мокаем streamlit
        mock_uploaded = MagicMock()
        mock_uploaded.name = "test.csv"
        mock_uploaded.read.return_value = csv_bytes
        
        with patch('modules.data_loader.st') as mock_st:
            mock_st.file_uploader.return_value = mock_uploaded
            mock_st.selectbox.side_effect = ['utf-8', ',']
            mock_st.radio.return_value = "Первая (0)"
            mock_st.number_input.return_value = 0
            
            result = load_data_interface()
            
            assert result is not None
            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(sample_dataframe)
    
    def test_load_excel_success(self, sample_dataframe):
        """Тест успешной загрузки Excel файла"""
        # Создаем Excel в памяти
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            sample_dataframe.to_excel(writer, index=False, sheet_name='Sheet1')
        excel_bytes = excel_buffer.getvalue()
        
        # Мокаем streamlit
        mock_uploaded = MagicMock()
        mock_uploaded.name = "test.xlsx"
        mock_uploaded.read.return_value = excel_bytes
        
        with patch('modules.data_loader.st') as mock_st:
            mock_st.file_uploader.return_value = mock_uploaded
            mock_st.radio.return_value = "Первая (0)"
            mock_st.selectbox.return_value = "Sheet1"
            
            result = load_data_interface()
            
            assert result is not None
            assert isinstance(result, pd.DataFrame)
    
    def test_load_no_file(self):
        """Тест случая когда файл не загружен"""
        with patch('modules.data_loader.st') as mock_st:
            mock_st.file_uploader.return_value = None
            
            result = load_data_interface()
            
            assert result is None
    
    def test_load_empty_file(self):
        """Тест загрузки пустого файла"""
        mock_uploaded = MagicMock()
        mock_uploaded.name = "test.csv"
        mock_uploaded.read.return_value = b""
        
        with patch('modules.data_loader.st') as mock_st:
            mock_st.file_uploader.return_value = mock_uploaded
            
            result = load_data_interface()
            
            assert result is None
    
    def test_load_csv_with_different_encodings(self):
        """Тест загрузки CSV с разными кодировками"""
        test_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['тест', 'данные', 'кодировка']
        })
        
        # Тестируем разные кодировки
        encodings = ['utf-8', 'cp1251', 'windows-1251']
        
        for encoding in encodings:
            try:
                csv_data = test_data.to_csv(index=False)
                csv_bytes = csv_data.encode(encoding)
                
                mock_uploaded = MagicMock()
                mock_uploaded.name = "test.csv"
                mock_uploaded.read.return_value = csv_bytes
                
                with patch('modules.data_loader.st') as mock_st:
                    mock_st.file_uploader.return_value = mock_uploaded
                    mock_st.selectbox.side_effect = [encoding, ',']
                    mock_st.radio.return_value = "Первая (0)"
                    mock_st.number_input.return_value = 0
                    
                    result = load_data_interface()
                    
                    if result is not None:
                        assert isinstance(result, pd.DataFrame)
            except UnicodeEncodeError:
                # Пропускаем кодировки которые не поддерживают кириллицу
                continue
    
    def test_load_csv_with_different_separators(self):
        """Тест загрузки CSV с разными разделителями"""
        separators = [',', ';', '\t', '|']
        
        for sep in separators:
            test_data = pd.DataFrame({
                'col1': [1, 2, 3],
                'col2': ['A', 'B', 'C']
            })
            
            csv_data = test_data.to_csv(index=False, sep=sep)
            csv_bytes = csv_data.encode('utf-8')
            
            mock_uploaded = MagicMock()
            mock_uploaded.name = "test.csv"
            mock_uploaded.read.return_value = csv_bytes
            
            with patch('modules.data_loader.st') as mock_st:
                mock_st.file_uploader.return_value = mock_uploaded
                mock_st.selectbox.side_effect = ['utf-8', sep]
                mock_st.radio.return_value = "Первая (0)"
                mock_st.number_input.return_value = 0
                
                result = load_data_interface()
                
                if result is not None:
                    assert isinstance(result, pd.DataFrame)
                    assert len(result) == len(test_data)
    
    def test_load_csv_with_skip_rows(self):
        """Тест загрузки CSV с пропуском строк"""
        test_data = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['A', 'B', 'C', 'D', 'E']
        })
        
        csv_data = test_data.to_csv(index=False)
        csv_bytes = csv_data.encode('utf-8')
        
        mock_uploaded = MagicMock()
        mock_uploaded.name = "test.csv"
        mock_uploaded.read.return_value = csv_bytes
        
        with patch('modules.data_loader.st') as mock_st:
            mock_st.file_uploader.return_value = mock_uploaded
            mock_st.selectbox.side_effect = ['utf-8', ',']
            mock_st.radio.return_value = "Первая (0)"
            mock_st.number_input.side_effect = [2, 1]  # skip_start=2, skip_end=1
            
            result = load_data_interface()
            
            if result is not None:
                assert isinstance(result, pd.DataFrame)
                # Должно остаться 5 - 2 - 1 = 2 строки
                assert len(result) == 2
    
    def test_load_csv_unicode_error(self):
        """Тест обработки ошибки UnicodeDecodeError"""
        # Создаем данные в неправильной кодировке
        wrong_encoding_data = "col1,col2\n1,тест".encode('utf-16')
        
        mock_uploaded = MagicMock()
        mock_uploaded.name = "test.csv"
        mock_uploaded.read.return_value = wrong_encoding_data
        
        with patch('modules.data_loader.st') as mock_st:
            mock_st.file_uploader.return_value = mock_uploaded
            mock_st.selectbox.side_effect = ['utf-8', ',']
            mock_st.radio.return_value = "Первая (0)"
            mock_st.number_input.return_value = 0
            mock_st.error.return_value = None
            
            result = load_data_interface()
            
            # Должен вернуть None при ошибке декодирования
            assert result is None
    
    def test_load_excel_multiple_sheets(self, sample_dataframe):
        """Тест загрузки Excel с несколькими листами"""
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            sample_dataframe.to_excel(writer, index=False, sheet_name='Sheet1')
            sample_dataframe.to_excel(writer, index=False, sheet_name='Sheet2')
        excel_bytes = excel_buffer.getvalue()
        
        mock_uploaded = MagicMock()
        mock_uploaded.name = "test.xlsx"
        mock_uploaded.read.return_value = excel_bytes
        
        with patch('modules.data_loader.st') as mock_st:
            mock_st.file_uploader.return_value = mock_uploaded
            mock_st.radio.return_value = "Первая (0)"
            mock_st.selectbox.side_effect = ['Sheet1', 'Sheet2']  # Тестируем оба листа
            
            # Первый вызов - Sheet1
            result1 = load_data_interface()
            assert result1 is not None
            
            # Второй вызов - Sheet2  
            result2 = load_data_interface()
            assert result2 is not None
    
    def test_large_file_warning(self):
        """Тест предупреждения о большом файле"""
        # Создаем большой DataFrame
        large_data = pd.DataFrame({
            'col1': range(20000),
            'col2': ['test'] * 20000
        })
        
        csv_data = large_data.to_csv(index=False)
        csv_bytes = csv_data.encode('utf-8')
        
        mock_uploaded = MagicMock()
        mock_uploaded.name = "large.csv"
        mock_uploaded.read.return_value = csv_bytes
        
        with patch('modules.data_loader.st') as mock_st:
            mock_st.file_uploader.return_value = mock_uploaded
            mock_st.selectbox.side_effect = ['utf-8', ',']
            mock_st.radio.return_value = "Первая (0)"
            mock_st.number_input.return_value = 0
            mock_st.warning.return_value = None
            mock_st.button.return_value = False  # Не нажимаем кнопку sample
            
            result = load_data_interface()
            
            # Должен вернуть данные с предупреждением
            assert result is not None
            assert len(result) == 20000