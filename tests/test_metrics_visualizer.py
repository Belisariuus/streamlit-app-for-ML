# tests/test_metrics_visualizer.py
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.metrics_visualizer import (
    visualize_metrics_interface,
    calculate_gini
)


class TestMetricsVisualizer:
    """Тесты для модуля визуализации метрик"""
    
    def test_calculate_gini_binary(self):
        """Тест расчета коэффициента Джинни для бинарной классификации"""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
        
        gini = calculate_gini(y_true, y_pred)
        
        assert gini is not None
        assert -1 <= gini <= 1
    
    def test_calculate_gini_multiclass(self):
        """Тест расчета коэффициента Джинни для многоклассовой классификации"""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0.1, 0.8, 0.1, 0.2, 0.7, 0.1])
        
        gini = calculate_gini(y_true, y_pred)
        
        # Для многоклассовой должен вернуть None
        assert gini is None
    
    def test_visualize_metrics_regression(self, mock_streamlit):
        """Тест визуализации для регрессии"""
        from sklearn.linear_model import LinearRegression
        
        # Создаем тестовые данные регрессии
        X_train = pd.DataFrame({'feature': [1, 2, 3, 4, 5]})
        X_test = pd.DataFrame({'feature': [6, 7, 8]})
        y_train = np.array([2, 4, 6, 8, 10])
        y_test = np.array([12, 14, 16])
        y_pred_train = np.array([2.1, 3.9, 6.1, 7.9, 10.1])
        y_pred_test = np.array([12.1, 13.9, 16.1])
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        results = {
            "model": model,
            "features": ['feature'],
            "target": 'target',
            "problem_type": "regression",
            "train_metrics": {"MAE": 0.1, "R2": 0.99},
            "test_metrics": {"MAE": 0.15, "R2": 0.98},
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "y_pred_train": y_pred_train,
            "y_pred_test": y_pred_test,
            "cv_scores": {"MAE": {"mean": 0.12, "std": 0.02}}
        }
        
        try:
            visualize_metrics_interface(results)
            assert True
        except Exception as e:
            pytest.fail(f"Ошибка при визуализации регрессии: {e}")
    
    def test_visualize_metrics_classification(self, mock_streamlit):
        """Тест визуализации для классификации"""
        from sklearn.ensemble import RandomForestClassifier
        
        # Создаем тестовые данные классификации
        X_train = pd.DataFrame({'feature': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        X_test = pd.DataFrame({'feature': [11, 12, 13]})
        y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        y_test = np.array([0, 1, 0])
        y_pred_train = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        y_pred_test = np.array([0, 1, 0])
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        results = {
            "model": model,
            "features": ['feature'],
            "target": 'target',
            "problem_type": "classification",
            "train_metrics": {"Accuracy": 1.0, "F1-Score": 1.0},
            "test_metrics": {"Accuracy": 1.0, "F1-Score": 1.0},
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "y_pred_train": y_pred_train,
            "y_pred_test": y_pred_test,
            "cv_scores": {"Accuracy": {"mean": 0.95, "std": 0.03}},
            "class_names": np.array([0, 1])
        }
        
        try:
            visualize_metrics_interface(results)
            assert True
        except Exception as e:
            pytest.fail(f"Ошибка при визуализации классификации: {e}")