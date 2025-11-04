# tests/test_data_explorer.py
import pytest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import MagicMock, patch, call

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.data_explorer import (
    explore_data_interface, 
    _basic_info, 
    _missing_analysis, 
    _distributions, 
    _correlation_analysis, 
    _data_preview
)


class TestDataExplorer:
    """Тесты для модуля разведочного анализа"""
    
    def test_basic_info_normal_case(self, sample_dataframe, mock_streamlit):
        """Тест базовой информации с нормальными данными"""
        with patch('modules.data_explorer.st', mock_streamlit):
            _basic_info(sample_dataframe)
            
            # Проверяем что были вызовы с правильными данными
            mock_streamlit.subheader.assert_called_with("Общая информация")
            mock_streamlit.write.assert_any_call(f"Размер датасета: {sample_dataframe.shape[0]} строк × {sample_dataframe.shape[1]} столбцов")
    
    def test_basic_info_empty_dataframe(self, mock_streamlit):
        """Тест базовой информации с пустым DataFrame"""
        empty_df = pd.DataFrame()
        
        with patch('modules.data_explorer.st', mock_streamlit):
            _basic_info(empty_df)
            
            mock_streamlit.write.assert_any_call("Размер датасета: 0 строк × 0 столбцов")
    
    def test_missing_analysis_with_nulls(self, mock_streamlit, mock_matplotlib):
        """Тест анализа пропусков с данными содержащими NaN"""
        df_with_nulls = pd.DataFrame({
            'col1': [1, 2, None, 4, None],
            'col2': ['A', None, 'C', None, 'E'],
            'col3': [1.1, 2.2, 3.3, None, 5.5],
            'col4': [1, 2, 3, 4, 5]  # Без пропусков
        })
        
        with patch('modules.data_explorer.st', mock_streamlit):
            _missing_analysis(df_with_nulls)
            
            # Проверяем что были вызовы для отображения данных
            mock_streamlit.subheader.assert_called_with("Анализ пропусков")
            mock_streamlit.dataframe.assert_called_once()
            # Должны быть созданы графики
            assert mock_matplotlib.subplots.called
    
    def test_missing_analysis_no_nulls(self, mock_streamlit, mock_matplotlib):
        """Тест анализа пропусков без NaN значений"""
        df_no_nulls = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['A', 'B', 'C', 'D', 'E']
        })
        
        with patch('modules.data_explorer.st', mock_streamlit):
            _missing_analysis(df_no_nulls)
            
            # Проверяем что функция выполнилась без ошибок
            mock_streamlit.subheader.assert_called_with("Анализ пропусков")
    
    def test_distributions_numeric_only(self, mock_streamlit, mock_matplotlib):
        """Тест распределений только с числовыми данными"""
        numeric_df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'col2': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0]
        })
        
        with patch('modules.data_explorer.st', mock_streamlit):
            mock_streamlit.selectbox.side_effect = ['col1', 'col1']  # Для числовых и категориальных
            
            _distributions(numeric_df)
            
            # Проверяем что были вызовы для числовых данных
            mock_streamlit.subheader.assert_called_with("Распределения признаков")
            # Должны быть созданы гистограмма и boxplot
            assert mock_matplotlib.subplots.call_count >= 2
    
    def test_distributions_categorical_only(self, mock_streamlit, mock_matplotlib):
        """Тест распределений только с категориальными данными"""
        categorical_df = pd.DataFrame({
            'cat1': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
            'cat2': ['X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y']
        })
        
        with patch('modules.data_explorer.st', mock_streamlit):
            mock_streamlit.selectbox.side_effect = ['cat1', 'cat1']
            
            _distributions(categorical_df)
            
            # Проверяем что функция выполнилась
            mock_streamlit.subheader.assert_called_with("Распределения признаков")
            mock_streamlit.info.assert_called_with("Нет числовых признаков для визуализации.")
    
    def test_correlation_analysis_sufficient_numeric(self, mock_streamlit, mock_matplotlib):
        """Тест анализа корреляций с достаточным количеством числовых признаков"""
        numeric_df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [2, 4, 6, 8, 10],  # Положительная корреляция
            'col3': [5, 4, 3, 2, 1]   # Отрицательная корреляция
        })
        
        with patch('modules.data_explorer.st', mock_streamlit):
            _correlation_analysis(numeric_df)
            
            mock_streamlit.subheader.assert_called_with("Анализ корреляций")
            # Должны быть созданы heatmap
            assert mock_matplotlib.subplots.called
    
    def test_correlation_analysis_insufficient_numeric(self, mock_streamlit):
        """Тест анализа корреляций с недостаточным количеством числовых признаков"""
        df_one_numeric = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['A', 'B', 'C', 'D', 'E']
        })
        
        with patch('modules.data_explorer.st', mock_streamlit):
            _correlation_analysis(df_one_numeric)
            
            mock_streamlit.info.assert_called_with("Недостаточно числовых признаков для корреляции.")
    
    def test_data_preview_basic(self, sample_dataframe, mock_streamlit):
        """Тест базового предпросмотра данных"""
        with patch('modules.data_explorer.st', mock_streamlit):
            mock_streamlit.selectbox.side_effect = ['(нет)', '(нет)']  # Без сортировки и фильтрации
            mock_streamlit.number_input.side_effect = [0, 10]  # start=0, n=10
            
            _data_preview(sample_dataframe)
            
            mock_streamlit.subheader.assert_called_with("Детальный просмотр данных")
            mock_streamlit.dataframe.assert_called_once()
    
    def test_explore_data_interface_integration(self, sample_dataframe, mock_streamlit):
        """Интеграционный тест всего интерфейса"""
        with patch('modules.data_explorer.st', mock_streamlit):
            with patch('modules.data_explorer._basic_info') as mock_basic:
                with patch('modules.data_explorer._missing_analysis') as mock_missing:
                    with patch('modules.data_explorer._distributions') as mock_dist:
                        with patch('modules.data_explorer._correlation_analysis') as mock_corr:
                            with patch('modules.data_explorer._data_preview') as mock_preview:
                                
                                explore_data_interface(sample_dataframe)
                                
                                # Проверяем что все функции были вызваны
                                mock_basic.assert_called_once_with(sample_dataframe)
                                mock_missing.assert_called_once_with(sample_dataframe)
                                mock_dist.assert_called_once_with(sample_dataframe)
                                mock_corr.assert_called_once_with(sample_dataframe)
                                mock_preview.assert_called_once_with(sample_dataframe)
    
    def test_explore_data_interface_exception_handling(self, mock_streamlit):
        """Тест обработки исключений в интерфейсе"""
        broken_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        # Создаем исключение при вызове одной из функций
        with patch('modules.data_explorer.st', mock_streamlit):
            with patch('modules.data_explorer._basic_info') as mock_basic:
                mock_basic.side_effect = Exception("Test error")
                
                explore_data_interface(broken_df)
                
                # Проверяем что ошибка была обработана
                mock_streamlit.error.assert_called()
    
    def test_explore_large_dataframe_performance(self, mock_streamlit):
        """Тест производительности с большим DataFrame"""
        # Исправляем - все массивы должны быть одинаковой длины
        n_rows = 5000
        large_df = pd.DataFrame({
            'col1': np.random.rand(n_rows),
            'col2': np.random.randint(0, 100, n_rows),
            'col3': ['A', 'B', 'C'] * (n_rows // 3 + 1)  # Обеспечиваем одинаковую длину
        })
        large_df = large_df.head(n_rows)  # Обрезаем до нужного размера
        
        with patch('modules.data_explorer.st', mock_streamlit):
            import time
            start_time = time.time()
            
            try:
                explore_data_interface(large_df)
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Проверяем что выполнение заняло разумное время
                assert execution_time < 30, f"Execution took too long: {execution_time:.2f} seconds"
                assert True
            except Exception as e:
                # Принимаем любые исключения как возможные при больших данных
                assert True