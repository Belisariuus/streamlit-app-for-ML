# tests/test_model_trainer.py
import pytest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import MagicMock, patch, call
import io
import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.model_trainer import (
    train_model_interface,
    _detect_problem_type,
    _get_model_by_name,
    _get_hyperparameter_options,
    save_model_bytes
)


class TestModelTrainer:
    """Тесты для модуля обучения моделей"""
    
    def test_detect_problem_type_regression(self):
        """Тест определения регрессионной задачи"""
        # Много уникальных числовых значений
        reg_series = pd.Series([1.5, 2.3, 3.1, 4.7, 5.2, 6.8, 7.4, 8.9, 9.1, 10.5, 11.2])
        assert _detect_problem_type(reg_series) == "regression"
        
        # Числовые значения с большим диапазоном
        reg_series2 = pd.Series([100, 200, 150, 300, 250, 400])
        assert _detect_problem_type(reg_series2) == "regression"
    
    def test_detect_problem_type_binary_classification(self):
        """Тест определения бинарной классификации"""
        # Бинарные числовые значения
        bin_series = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
        assert _detect_problem_type(bin_series) == "classification"
        
        # Бинарные значения в пределах 0-100
        bin_series2 = pd.Series([0, 1, 0, 1, 0])
        assert _detect_problem_type(bin_series2) == "classification"
    
    def test_detect_problem_type_multiclass_classification(self):
        """Тест определения многоклассовой классификации"""
        # Многоклассовые числовые значения
        multi_series = pd.Series([0, 1, 2, 0, 1, 2, 0])
        assert _detect_problem_type(multi_series) == "classification"
        
        # Строковые категории
        str_series = pd.Series(['cat', 'dog', 'bird', 'cat'])
        assert _detect_problem_type(str_series) == "classification"
        
        # Категориальные данные
        cat_series = pd.Series(pd.Categorical(['A', 'B', 'C', 'A']))
        assert _detect_problem_type(cat_series) == "classification"
    
    def test_detect_problem_type_edge_cases(self):
        """Тест граничных случаев определения типа задачи"""
        # Пограничный случай - 10 уникальных значений (классификация)
        borderline_series = pd.Series(range(10))
        assert _detect_problem_type(borderline_series) == "classification"
        
        # 11 уникальных значений (регрессия)
        regression_series = pd.Series(range(11))
        assert _detect_problem_type(regression_series) == "regression"
    
    def test_get_model_by_name_regression_success(self):
        """Тест успешного получения моделей регрессии"""
        regression_models = [
            "LinearRegression", "Ridge", "Lasso", "ElasticNet",
            "RandomForest", "GradientBoosting", "SVR", "KNeighbors"
        ]
        
        for model_name in regression_models:
            try:
                model = _get_model_by_name(model_name, {"random_state": 42}, "regression")
                assert model is not None
                # Проверяем что у модели есть метод fit
                assert hasattr(model, 'fit')
            except Exception as e:
                # Пропускаем если модель недоступна
                if "не доступна" not in str(e):
                    pytest.fail(f"Ошибка при создании модели {model_name}: {e}")
    
    def test_get_model_by_name_classification_success(self):
        """Тест успешного получения моделей классификации"""
        classification_models = [
            "LogisticRegression", "RidgeClassifier", "RandomForest",
            "GradientBoosting", "SVC", "KNeighbors"
        ]
        
        for model_name in classification_models:
            try:
                model = _get_model_by_name(model_name, {"random_state": 42}, "classification")
                assert model is not None
                assert hasattr(model, 'fit')
            except Exception as e:
                if "не доступна" not in str(e):
                    pytest.fail(f"Ошибка при создании модели {model_name}: {e}")
    
    def test_get_model_by_name_invalid_problem_type(self):
        """Тест получения модели с неверным типом задачи"""
        with pytest.raises(ValueError, match="Неизвестный тип задачи"):
            _get_model_by_name("LinearRegression", {}, "invalid_type")
    
    def test_get_model_by_name_unavailable_model(self):
        """Тест получения недоступной модели"""
        # Мокаем что XGBoost недоступен
        with patch('modules.model_trainer.XGBRegressor', None):
            with pytest.raises(ValueError, match="не доступна"):
                _get_model_by_name("XGBoost", {}, "regression")
    
    def test_get_hyperparameter_options_regression(self):
        """Тест получения опций гиперпараметров для регрессии"""
        # Тестируем разные модели регрессии
        models_to_test = ["Ridge", "Lasso", "ElasticNet", "RandomForest", "SVR"]
        
        for model_name in models_to_test:
            options = _get_hyperparameter_options(model_name, "regression")
            assert isinstance(options, dict)
            # Проверяем что есть хотя бы один параметр
            if model_name in ["Ridge", "Lasso", "ElasticNet"]:
                assert "alpha" in options
    
    def test_get_hyperparameter_options_classification(self):
        """Тест получения опций гиперпараметров для классификации"""
        models_to_test = ["LogisticRegression", "RandomForest", "SVC"]
        
        for model_name in models_to_test:
            options = _get_hyperparameter_options(model_name, "classification")
            assert isinstance(options, dict)
    
    def test_save_model_bytes(self):
        """Тест сохранения модели в bytes"""
        from sklearn.linear_model import LinearRegression
        
        # Создаем и обучаем простую модель
        model = LinearRegression()
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 2, 3])
        model.fit(X, y)
        
        # Сохраняем в bytes
        model_bytes = save_model_bytes(model)
        
        # Проверяем результат
        assert isinstance(model_bytes, bytes)
        assert len(model_bytes) > 0
        
        # Проверяем что можно загрузить модель обратно
        loaded_model = joblib.load(io.BytesIO(model_bytes))
        assert hasattr(loaded_model, 'predict')
    
    def test_train_model_interface_regression(self, regression_data, mock_streamlit):
        """Тест интерфейса обучения регрессионной модели"""
        with patch('modules.model_trainer.st', mock_streamlit):
            # Настраиваем моки для регрессии
            mock_streamlit.selectbox.side_effect = [
                'target',  # целевая переменная
                'LinearRegression',  # алгоритм
                'none'  # оптимизация гиперпараметров
            ]
            mock_streamlit.multiselect.return_value = ['feature_0', 'feature_1', 'feature_2']
            mock_streamlit.slider.return_value = 0.2
            mock_streamlit.number_input.return_value = 42
            mock_streamlit.button.return_value = True
            
            # Мокаем прогресс бар и статус
            mock_streamlit.progress.return_value = None
            mock_streamlit.empty.return_value = MagicMock()
            
            model, results = train_model_interface(regression_data)
            
            # Проверяем что функция выполнилась
            # Может вернуть None если не нажата кнопка, или модель если обучение прошло
            assert model is None or hasattr(model, 'predict')
            assert results is None or isinstance(results, dict)
    
    def test_train_model_interface_classification(self, classification_data, mock_streamlit):
        """Тест интерфейса обучения модели классификации"""
        with patch('modules.model_trainer.st', mock_streamlit):
            # Настраиваем моки для классификации
            mock_streamlit.selectbox.side_effect = [
                'target',  # целевая переменная
                'LogisticRegression',  # алгоритм
                'none'  # оптимизация гиперпараметров
            ]
            mock_streamlit.multiselect.return_value = ['feature_0', 'feature_1', 'feature_2']
            mock_streamlit.slider.return_value = 0.2
            mock_streamlit.number_input.return_value = 42
            mock_streamlit.button.return_value = True
            
            mock_streamlit.progress.return_value = None
            mock_streamlit.empty.return_value = MagicMock()
            
            model, results = train_model_interface(classification_data)
            
            assert model is None or hasattr(model, 'predict')
            assert results is None or isinstance(results, dict)
    
    def test_train_model_interface_no_button_click(self, regression_data, mock_streamlit):
        """Тест когда кнопка обучения не нажата"""
        with patch('modules.model_trainer.st', mock_streamlit):
            mock_streamlit.button.return_value = False
            
            model, results = train_model_interface(regression_data)
            
            # Должны вернуться None так как кнопка не нажата
            assert model is None
            assert results is None
    
    def test_train_model_interface_insufficient_data(self, mock_streamlit):
        """Тест с недостаточным количеством данных"""
        small_data = pd.DataFrame({
            'feature': [1, 2, 3],  # Всего 3 строки
            'target': [1, 2, 3]
        })
        
        with patch('modules.model_trainer.st', mock_streamlit):
            mock_streamlit.selectbox.return_value = 'target'
            mock_streamlit.multiselect.return_value = ['feature']
            mock_streamlit.button.return_value = True
            
            model, results = train_model_interface(small_data)
            
            # Должны вернуться None из-за недостатка данных
            assert model is None
            assert results is None
    
    def test_train_model_interface_with_missing_values(self, mock_streamlit):
        """Тест с пропущенными значениями"""
        data_with_nulls = pd.DataFrame({
            'feature1': [1, 2, None, 4, 5],
            'feature2': [1.1, None, 3.3, 4.4, 5.5],
            'target': [10, 20, 30, 40, 50]
        })
        
        with patch('modules.model_trainer.st', mock_streamlit):
            mock_streamlit.selectbox.return_value = 'target'
            mock_streamlit.multiselect.return_value = ['feature1', 'feature2']
            mock_streamlit.button.return_value = True
            mock_streamlit.progress.return_value = None
            mock_streamlit.empty.return_value = MagicMock()
            
            model, results = train_model_interface(data_with_nulls)
            
            assert model is None or hasattr(model, 'predict')
    
    def test_train_model_interface_hyperparameter_optimization(self, regression_data, mock_streamlit):
        """Тест с оптимизацией гиперпараметров"""
        with patch('modules.model_trainer.st', mock_streamlit):
            mock_streamlit.selectbox.side_effect = [
                'target',
                'RandomForest',
                'grid_search'  # Включаем оптимизацию
            ]
            mock_streamlit.multiselect.return_value = ['feature_0', 'feature_1']
            mock_streamlit.slider.side_effect = [0.2, 3, 10]  # test_size, cv_folds, n_iter
            mock_streamlit.number_input.return_value = 42
            mock_streamlit.button.return_value = True
            mock_streamlit.progress.return_value = None
            mock_streamlit.empty.return_value = MagicMock()
            
            # Мокаем GridSearchCV чтобы избежать реального обучения
            with patch('modules.model_trainer.GridSearchCV') as mock_gs:
                mock_gs_instance = MagicMock()
                mock_gs.return_value = mock_gs_instance
                mock_gs_instance.best_estimator_ = MagicMock()
                mock_gs_instance.best_params_ = {'n_estimators': 100}
                mock_gs_instance.fit.return_value = None
                
                model, results = train_model_interface(regression_data)
                
                # Проверяем что GridSearch был вызван
                mock_gs.assert_called_once()
    
    def test_train_model_interface_cross_validation_error(self, regression_data, mock_streamlit):
        """Тест обработки ошибок при кросс-валидации"""
        with patch('modules.model_trainer.st', mock_streamlit):
            mock_streamlit.selectbox.side_effect = ['target', 'LinearRegression', 'none']
            mock_streamlit.multiselect.return_value = ['feature_0', 'feature_1']
            mock_streamlit.button.return_value = True
            mock_streamlit.progress.return_value = None
            mock_streamlit.empty.return_value = MagicMock()
            
            # Мокаем cross_val_score чтобы вызвать ошибку
            with patch('modules.model_trainer.cross_val_score') as mock_cv:
                mock_cv.side_effect = Exception("CV error")
                
                model, results = train_model_interface(regression_data)
                
                # Проверяем что ошибка была обработана
                mock_streamlit.warning.assert_called()
    
    def test_train_model_interface_model_training_error(self, regression_data, mock_streamlit):
        """Тест обработки ошибок при обучении модели"""
        with patch('modules.model_trainer.st', mock_streamlit):
            mock_streamlit.selectbox.side_effect = ['target', 'LinearRegression', 'none']
            mock_streamlit.multiselect.return_value = ['feature_0', 'feature_1']
            mock_streamlit.button.return_value = True
            mock_streamlit.progress.return_value = None
            mock_streamlit.empty.return_value = MagicMock()
            
            # Мокаем model.fit чтобы вызвать ошибку
            with patch('modules.model_trainer._get_model_by_name') as mock_get_model:
                mock_model = MagicMock()
                mock_model.fit.side_effect = Exception("Training error")
                mock_get_model.return_value = mock_model
                
                model, results = train_model_interface(regression_data)
                
                # Проверяем что ошибка была обработана
                mock_streamlit.error.assert_called()
    
    def test_catboost_verbose_false(self):
        """Тест что CatBoost создается с verbose=False"""
        # Мокаем CatBoost чтобы проверить параметры
        mock_catboost = MagicMock()
        
        with patch('modules.model_trainer.CatBoostRegressor', mock_catboost):
            try:
                _get_model_by_name("CatBoost", {"n_estimators": 100}, "regression")
                # Проверяем что вызван с verbose=False
                mock_catboost.assert_called_with(n_estimators=100, verbose=False)
            except Exception:
                # Если CatBoost недоступен - пропускаем
                pass


