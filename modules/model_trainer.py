# modules/model_trainer.py
"""
Модуль 4: Обучение регрессионных и классификационных моделей
Поддерживает выбор алгоритма, настройку гиперпараметров, cross-validation, автоматический подбор параметров и отображение результатов.
"""
from typing import Optional, Dict, Any, Tuple, List, Union
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression, RidgeClassifier, LassoCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
import joblib
import io
import time

# Optional boosters
try:
    from xgboost import XGBRegressor, XGBClassifier
except ImportError:
    XGBRegressor = XGBClassifier = None

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
except ImportError:
    LGBMRegressor = LGBMClassifier = None

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
except ImportError:
    CatBoostRegressor = CatBoostClassifier = None

def _detect_problem_type(y: pd.Series) -> str:
    """Автоматическое определение типа задачи: регрессия или классификация"""
    unique_values = y.nunique()

    # Если <= 10 уникальных значений и они целочисленные - классификация
    if unique_values <= 10 and pd.api.types.is_numeric_dtype(y):
        # Проверяем, что значения в разумных пределах для классификации
        if y.min() >= 0 and y.max() <= 100:
            return "classification"

    # Если строковые значения - классификация
    if pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(y):
        return "classification"

    # Если много уникальных числовых значений - регрессия
    return "regression"

def _get_model_by_name(name: str, params: Dict[str, Any], problem_type: str):
    """Создание модели по имени и типу задачи"""
    model_map = {
        "regression": {
            "LinearRegression": LinearRegression,
            "Ridge": Ridge,
            "Lasso": Lasso,
            "ElasticNet": ElasticNet,
            "RandomForest": RandomForestRegressor,
            "GradientBoosting": GradientBoostingRegressor,
            "SVR": SVR,
            "KNeighbors": KNeighborsRegressor,
            "XGBoost": XGBRegressor,
            "LightGBM": LGBMRegressor,
            "CatBoost": CatBoostRegressor
        },
        "classification": {
            "LogisticRegression": LogisticRegression,
            "RidgeClassifier": RidgeClassifier,
            "RandomForest": RandomForestClassifier,
            "GradientBoosting": GradientBoostingClassifier,
            "SVC": SVC,
            "KNeighbors": KNeighborsClassifier,
            "XGBoost": XGBClassifier,
            "LightGBM": LGBMClassifier,
            "CatBoost": CatBoostClassifier
        }
    }

    if problem_type not in model_map:
        raise ValueError(f"Неизвестный тип задачи: {problem_type}")

    if name not in model_map[problem_type]:
        raise ValueError(f"Модель {name} не доступна для {problem_type}")

    model_class = model_map[problem_type][name]

    if model_class is None:
        raise ValueError(f"Модель {name} не доступна. Убедитесь, что пакет установлен.")

    # Специальная обработка для CatBoost
    if name == "CatBoost":
        params = params.copy()
        params['verbose'] = False

    return model_class(**params)

def save_model_bytes(model) -> bytes:
    """Сохранение модели в bytes"""
    b = io.BytesIO()
    joblib.dump(model, b)
    b.seek(0)
    return b.read()

def _get_hyperparameter_options(model_name: str, problem_type: str) -> Dict[str, Any]:
    """Возвращает опции гиперпараметров для разных моделей"""
    base_options = {}

    if problem_type == "regression":
        if model_name in ["Ridge", "Lasso", "ElasticNet"]:
            base_options = {
                "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
            }
        elif model_name == "ElasticNet":
            base_options["l1_ratio"] = [0.1, 0.3, 0.5, 0.7, 0.9]

    if model_name in ["RandomForest", "GradientBoosting"]:
        base_options = {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    elif model_name == "SVR" or model_name == "SVC":
        base_options = {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["linear", "rbf"]
        }
    elif model_name == "KNeighbors":
        base_options = {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"]
        }
    elif model_name in ["XGBoost", "LightGBM", "CatBoost"]:
        base_options = {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7]
        }
    elif model_name == "LogisticRegression":
        base_options = {
            "C": [0.001, 0.01, 0.1, 1.0, 10.0],
            "penalty": ["l1", "l2", "elasticnet"],
            "solver": ["liblinear", "saga"]
        }

    return base_options

# modules/model_trainer.py (исправленная часть)

def train_model_interface(df: pd.DataFrame) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    st.header("4. Обучение модели")
    
    try:
        cols = df.columns.tolist()
        
        # Выбор целевой переменной
        target = st.selectbox("Выберите целевой столбец", options=cols)
        
        # Автоматическое определение типа задачи
        problem_type = _detect_problem_type(df[target])
        st.info(f"Автоматически определена задача: **{problem_type.upper()}**")
        
        # Показ информации о целевой переменной
        st.write("**Информация о целевой переменной:**")
        if problem_type == "classification":
            value_counts = df[target].value_counts()
            st.write(f"Классы: {value_counts.to_dict()}")
            st.write(f"Всего классов: {len(value_counts)}")
            
            # Проверка на бинарную классификацию для коэффициента Джинни
            if len(value_counts) == 2:
                st.info("✅ Бинарная классификация - будет рассчитан коэффициент Джинни")
            else:
                st.info("ℹ️ Многоклассовая классификация - коэффициент Джинни не рассчитывается")
        else:
            st.write(f"Диапазон значений: {df[target].min():.2f} - {df[target].max():.2f}")
            st.write(f"Среднее: {df[target].mean():.2f}, Стандартное отклонение: {df[target].std():.2f}")
        
        # Выбор признаков
        features = st.multiselect(
            "Выберите признаки (оставьте пустым, чтобы использовать все, кроме таргета)", 
            options=[c for c in cols if c != target],
            default=[c for c in cols if c != target]
        )
        
        if not features:
            features = [c for c in cols if c != target]

        # Настройки обучения
        st.subheader("Настройки обучения")
        col1, col2 = st.columns(2)
        
        with col1:
            test_size = st.slider("Доля тестовой выборки", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
            random_state = st.number_input("Random state", min_value=0, value=42)
            cv_folds = st.slider("Число фолдов CV", min_value=2, max_value=5, value=3)  # Уменьшили максимальное значение
        
        with col2:
            # Выбор алгоритма в зависимости от типа задачи
            if problem_type == "regression":
                algorithms = [
                    "LinearRegression", "Ridge", "Lasso", "ElasticNet", 
                    "RandomForest", "GradientBoosting", "SVR", "KNeighbors"
                ]
                # Добавляем бустинги если доступны
                if XGBRegressor: algorithms.append("XGBoost")
                if LGBMRegressor: algorithms.append("LightGBM")
                if CatBoostRegressor: algorithms.append("CatBoost")
            else:
                algorithms = [
                    "LogisticRegression", "RidgeClassifier", 
                    "RandomForest", "GradientBoosting", "SVC", "KNeighbors"
                ]
                if XGBClassifier: algorithms.append("XGBoost")
                if LGBMClassifier: algorithms.append("LightGBM")
                if CatBoostClassifier: algorithms.append("CatBoost")
            
            alg = st.selectbox("Выберите алгоритм", options=algorithms)
            
            # Метод оптимизации гиперпараметров
            optimization_method = st.selectbox(
                "Метод оптимизации гиперпараметров",
                options=["none", "grid_search", "random_search"],
                index=0
            )

        # Базовые гиперпараметры
        st.subheader("Гиперпараметры")
        params = {}
        
        # Упрощенные параметры для избежания ошибок
        if alg in ["Ridge", "Lasso", "ElasticNet", "RidgeClassifier"]:
            params["alpha"] = st.number_input("alpha", min_value=0.0, value=1.0, step=0.1)
            params["random_state"] = random_state
        
        if alg == "ElasticNet":
            params["l1_ratio"] = st.slider("l1_ratio", min_value=0.0, max_value=1.0, value=0.5)
        
        if alg in ["RandomForest", "GradientBoosting"]:
            col1, col2 = st.columns(2)
            with col1:
                params["n_estimators"] = st.number_input("n_estimators", min_value=10, max_value=200, value=50)  # Уменьшили максимальное значение
                params["max_depth"] = st.selectbox("max_depth", options=[3, 5, 7, 10, None], index=1)
            with col2:
                params["min_samples_split"] = st.number_input("min_samples_split", min_value=2, max_value=20, value=2)
                params["min_samples_leaf"] = st.number_input("min_samples_leaf", min_value=1, max_value=10, value=1)
            params["random_state"] = random_state
        
        if alg in ["SVR", "SVC"]:
            col1, col2 = st.columns(2)
            with col1:
                params["C"] = st.number_input("C", min_value=0.001, value=1.0, step=0.1)
                params["kernel"] = st.selectbox("kernel", options=["rbf", "linear"], index=0)  # Убрали poly
            with col2:
                params["gamma"] = st.selectbox("gamma", options=["scale", "auto"])
            params["random_state"] = random_state
        
        if alg == "KNeighbors":
            params["n_neighbors"] = st.number_input("n_neighbors", min_value=1, max_value=50, value=5)  # Уменьшили максимальное значение
            params["weights"] = st.selectbox("weights", options=["uniform", "distance"])
        
        if alg in ["XGBoost", "LightGBM", "CatBoost"]:
            col1, col2 = st.columns(2)
            with col1:
                params["n_estimators"] = st.number_input("n_estimators", min_value=10, max_value=200, value=50)
                params["learning_rate"] = st.number_input("learning_rate", min_value=1e-4, max_value=1.0, value=0.1, format="%.4f")
            with col2:
                params["max_depth"] = st.number_input("max_depth", min_value=1, max_value=10, value=3)  # Уменьшили максимальное значение
            params["random_state"] = random_state
        
        if alg == "LogisticRegression":
            col1, col2 = st.columns(2)
            with col1:
                params["C"] = st.number_input("C", min_value=0.001, value=1.0, step=0.1)
                params["penalty"] = st.selectbox("penalty", options=["l2", "none"], index=0)  # Упростили варианты
            with col2:
                params["solver"] = st.selectbox("solver", options=["lbfgs", "liblinear"])
            params["random_state"] = random_state
            params["max_iter"] = 1000  # Добавили ограничение итераций

        # Автоматический подбор гиперпараметров
        if optimization_method != "none":
            st.subheader("Автоматический подбор гиперпараметров")
            param_options = _get_hyperparameter_options(alg, problem_type)
            
            if optimization_method == "grid_search":
                st.info("GridSearch будет проверять все комбинации параметров")
            else:
                st.info("RandomSearch будет проверять случайные комбинации")
                n_iter = st.number_input("Количество итераций", min_value=5, max_value=50, value=10)  # Уменьшили

        # Кнопка обучения
        if st.button("Обучить модель", type="primary"):
            # Подготовка данных
            X = df[features]
            y = df[target]
            
            # Преобразование целевой переменной для классификации
            if problem_type == "classification":
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
                class_names = le.classes_
            else:
                y_encoded = y
                class_names = None
            
            # Проверка и обработка пропусков
            if X.isna().any().any():
                st.warning("В признаках есть пропуски — заполняем медианой/модой.")
                # Заполняем числовые колонки медианой, категориальные модой
                for col in X.columns:
                    if X[col].dtype in ['int64', 'float64']:
                        X[col] = X[col].fillna(X[col].median())
                    else:
                        X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "Unknown")
            
            if pd.isna(y_encoded).any():
                st.error("Целевой столбец содержит пропуски — удаляем строки с пропусками.")
                mask = ~pd.isna(y_encoded)
                X = X[mask]
                y_encoded = y_encoded[mask]
                y = y[mask]
            
            # Проверка на достаточность данных
            if len(X) < 10:
                st.error("❌ Недостаточно данных для обучения (меньше 10 строк)")
                return None, None
            
            # Разделение данных
            if problem_type == "classification" and len(np.unique(y_encoded)) > 1:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=test_size, random_state=random_state
                )
            
            st.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
            
            # Создание и обучение модели
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Создание модели...")
                model = _get_model_by_name(alg, params, problem_type)
                progress_bar.progress(20)
                
                # Автоматический подбор гиперпараметров
                best_model = model
                if optimization_method != "none":
                    status_text.text("Подбор гиперпараметров...")
                    param_options = _get_hyperparameter_options(alg, problem_type)
                    
                    # Упрощенные метрики для избежания ошибок
                    scoring = 'accuracy' if problem_type == 'classification' else 'r2'
                    
                    try:
                        if optimization_method == "grid_search":
                            search = GridSearchCV(
                                model, param_options, cv=min(3, cv_folds),  # Уменьшили CV для поиска
                                scoring=scoring,
                                n_jobs=1,  # Убрали параллелизацию для стабильности
                                error_score='raise'
                            )
                        else:
                            search = RandomizedSearchCV(
                                model, param_options, cv=min(3, cv_folds), n_iter=n_iter,
                                scoring=scoring,
                                n_jobs=1, random_state=random_state,
                                error_score='raise'
                            )
                        
                        search.fit(X_train, y_train)
                        best_model = search.best_estimator_
                        st.success(f"Лучшие параметры: {search.best_params_}")
                    except Exception as e:
                        st.warning(f"Не удалось выполнить подбор гиперпараметров: {e}")
                        st.info("Используются базовые параметры")
                        best_model = model
                
                progress_bar.progress(40)
                
                # Кросс-валидация с обработкой ошибок
                status_text.text("Кросс-валидация...")
                cv_scores = {}
                
                if problem_type == "regression":
                    scoring_metrics = {"MAE": "neg_mean_absolute_error", "R2": "r2"}
                else:
                    scoring_metrics = {"Accuracy": "accuracy", "F1": "f1_weighted"}
                
                for name, sc in scoring_metrics.items():
                    try:
                        scores = cross_val_score(
                            best_model, X_train, y_train, 
                            cv=min(cv_folds, 3),  # Уменьшили количество фолдов
                            scoring=sc, 
                            n_jobs=1,  # Убрали параллелизацию
                            error_score='raise'
                        )
                        if name in ["MAE"]:
                            scores = -scores
                        cv_scores[name] = {"mean": float(np.mean(scores)), "std": float(np.std(scores))}
                    except Exception as e:
                        st.warning(f"Не удалось вычислить {name} при CV: {e}")
                        cv_scores[name] = {"mean": 0.0, "std": 0.0}
                
                if cv_scores:
                    st.write("**Результаты кросс-валидации (Train):**")
                    cv_df = pd.DataFrame({k: [v["mean"], v["std"]] for k, v in cv_scores.items()}, 
                                       index=["Mean", "Std"]).T
                    st.dataframe(cv_df.style.format("{:.4f}"))
                else:
                    st.warning("Не удалось выполнить кросс-валидацию")
                
                progress_bar.progress(60)
                
                # Финальное обучение
                status_text.text("Финальное обучение...")
                best_model.fit(X_train, y_train)
                progress_bar.progress(80)
                
                # Предсказания
                y_pred_train = best_model.predict(X_train)
                y_pred_test = best_model.predict(X_test)
                
                # Расчет метрик
                if problem_type == "regression":
                    def regression_metrics(y_true, y_pred):
                        mae = mean_absolute_error(y_true, y_pred)
                        mse = mean_squared_error(y_true, y_pred)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_true, y_pred)
                        n = len(y_true)
                        p = X_train.shape[1]
                        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else None
                        return {
                            "MAE": float(mae), "MSE": float(mse), "RMSE": float(rmse), 
                            "R2": float(r2), "Adj_R2": float(adj_r2) if adj_r2 is not None else None
                        }
                    
                    train_metrics = regression_metrics(y_train, y_pred_train)
                    test_metrics = regression_metrics(y_test, y_pred_test)
                    
                else:
                    def classification_metrics(y_true, y_pred):
                        return {
                            "Accuracy": float(accuracy_score(y_true, y_pred)),
                            "Precision": float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
                            "Recall": float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
                            "F1-Score": float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
                        }
                    
                    train_metrics = classification_metrics(y_train, y_pred_train)
                    test_metrics = classification_metrics(y_test, y_pred_test)
                    
                    # Матрица ошибок
                    if len(np.unique(y_test)) <= 10:  # Только для разумного количества классов
                        st.subheader("Матрица ошибок (Test)")
                        cm = confusion_matrix(y_test, y_pred_test)
                        cm_df = pd.DataFrame(cm, 
                                           index=class_names if class_names is not None else range(len(cm)),
                                           columns=class_names if class_names is not None else range(len(cm)))
                        st.dataframe(cm_df)
                
                progress_bar.progress(100)
                
                # Отображение метрик
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Метрики (Train):**")
                    st.json(train_metrics)
                with col2:
                    st.write("**Метрики (Test):**")
                    st.json(test_metrics)
                
                # Сохранение результатов
                results = {
                    "model": best_model,
                    "features": features,
                    "target": target,
                    "problem_type": problem_type,
                    "train_metrics": train_metrics,
                    "test_metrics": test_metrics,
                    "X_train": X_train,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_test": y_test,
                    "y_pred_train": y_pred_train,
                    "y_pred_test": y_pred_test,
                    "cv_scores": cv_scores,
                    "class_names": class_names
                }
                
                # Скачивание модели
                st.subheader("Скачать модель")
                model_bytes = save_model_bytes(best_model)
                st.download_button(
                    label="Скачать обученную модель",
                    data=model_bytes,
                    file_name=f"model_{alg}_{problem_type}.joblib",
                    mime="application/octet-stream"
                )
                
                status_text.text("Обучение завершено!")
                st.success("Модель успешно обучена и оценена.")
                return best_model, results
                
            except Exception as e:
                st.error(f"Ошибка при обучении модели: {e}")
                st.info("""
                **Возможные причины ошибки:**
                - Несовместимые типы данных в признаках
                - Слишком мало данных для выбранного алгоритма
                - Проблемы с масштабированием данных
                - Попробуйте упростить модель или выбрать другой алгоритм
                """)
                return None, None
                
        else:
            st.info("Настройте параметры и нажмите 'Обучить модель'.")
            return None, None
            
    except Exception as e:
        st.error(f"Ошибка в модуле обучения: {e}")
        return None, None