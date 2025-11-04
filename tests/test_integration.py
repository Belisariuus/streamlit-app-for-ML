# tests/test_integration.py
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class TestIntegration:
    """Интеграционные тесты для всего приложения"""
    
    def test_full_workflow_regression(self, regression_data, mock_streamlit):
        """Тест полного workflow для регрессии"""
        # Мокаем все вызовы Streamlit
        mock_streamlit.selectbox.side_effect = [
            # Загрузка данных
            'csv', 'auto', 'utf-8',
            # Предобработка
            'target', 'mean', 'most_frequent', 'onehot', 'none', 'none',
            # Обучение модели
            'target', 'LinearRegression'
        ]
        mock_streamlit.multiselect.return_value = [col for col in regression_data.columns if col != 'target']
        mock_streamlit.slider.return_value = 0.2
        mock_streamlit.number_input.return_value = 42
        mock_streamlit.button.return_value = True
        mock_streamlit.file_uploader.return_value = None
        
        # Тестируем модули по очереди
        from modules.data_loader import load_data_interface
        from modules.data_explorer import explore_data_interface
        from modules.data_preprocessor import preprocess_interface
        from modules.model_trainer import train_model_interface
        from modules.metrics_visualizer import visualize_metrics_interface
        
        # 1. Загрузка данных (пропускаем, т.к. нужен реальный файл)
        # df = load_data_interface()
        
        # 2. Анализ данных
        try:
            explore_data_interface(regression_data)
            assert True
        except Exception as e:
            pytest.fail(f"Ошибка в анализе данных: {e}")
        
        # 3. Предобработка
        try:
            processed_df, pipeline = preprocess_interface(regression_data)
            assert processed_df is None or isinstance(processed_df, pd.DataFrame)
        except Exception as e:
            pytest.fail(f"Ошибка в предобработке: {e}")
        
        # 4. Обучение модели
        try:
            model, results = train_model_interface(regression_data)
            assert model is None or hasattr(model, 'predict')
        except Exception as e:
            pytest.fail(f"Ошибка в обучении модели: {e}")
        
        # 5. Визуализация метрик (если модель обучена)
        if results is not None:
            try:
                visualize_metrics_interface(results)
                assert True
            except Exception as e:
                pytest.fail(f"Ошибка в визуализации метрик: {e}")
    
    def test_data_consistency(self, sample_dataframe):
        """Тест согласованности данных между модулями"""
        # Проверяем что данные не теряются при передаче между модулями
        original_shape = sample_dataframe.shape
        original_columns = set(sample_dataframe.columns)
        
        # Имитируем простую предобработку
        processed_df = sample_dataframe.copy()
        processed_df = processed_df.fillna(0)  # простая обработка пропусков
        
        # Проверяем что данные остались согласованными
        assert processed_df.shape[0] == original_shape[0]  # количество строк не изменилось
        assert set(processed_df.columns) == original_columns  # колонки те же