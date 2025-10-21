# modules/model_trainer.py
"""
Модуль 4: Обучение регрессионных моделей
Поддерживает выбор алгоритма, настройку гиперпараметров, cross-validation и отображение результатов.
"""
from typing import Optional, Dict, Any, Tuple
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import io

# Optional boosters
try:
    from xgboost import XGBRegressor  # type: ignore
except Exception:
    XGBRegressor = None  # type: ignore
try:
    from lightgbm import LGBMRegressor  # type: ignore
except Exception:
    LGBMRegressor = None  # type: ignore
try:
    from catboost import CatBoostRegressor  # type: ignore
except Exception:
    CatBoostRegressor = None  # type: ignore

def _get_model_by_name(name: str, params: Dict[str, Any]):
    if name == "LinearRegression":
        return LinearRegression(**params)
    if name == "Ridge":
        return Ridge(**params)
    if name == "Lasso":
        return Lasso(**params)
    if name == "ElasticNet":
        return ElasticNet(**params)
    if name == "RandomForest":
        return RandomForestRegressor(**params)
    if name == "SVR":
        return SVR(**params)
    if name == "KNeighbors":
        return KNeighborsRegressor(**params)
    if name == "XGBoost" and XGBRegressor is not None:
        return XGBRegressor(**params)
    if name == "LightGBM" and LGBMRegressor is not None:
        return LGBMRegressor(**params)
    if name == "CatBoost" and CatBoostRegressor is not None:
        return CatBoostRegressor(verbose=0, **params)
    raise ValueError(f"Модель {name} не доступна. Убедитесь, что пакет установлен.")

def save_model_bytes(model) -> bytes:
    b = io.BytesIO()
    joblib.dump(model, b)
    b.seek(0)
    return b.read()

def train_model_interface(df: pd.DataFrame) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    st.header("4. Обучение модели (регрессия)")
    try:
        cols = df.columns.tolist()
        target = st.selectbox("Выберите целевой столбец", options=cols)
        features = st.multiselect("Выберите признаки (оставьте пустым, чтобы использовать все, кроме таргета)", options=[c for c in cols if c!=target])
        if not features:
            features = [c for c in cols if c!=target]

        test_size = st.slider("Доля тестовой выборки", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
        random_state = st.number_input("Random state", min_value=0, value=42)
        cv_folds = st.slider("Число фолдов CV", min_value=2, max_value=10, value=5)

        alg = st.selectbox("Выберите алгоритм", options=[
            "LinearRegression", "Ridge", "Lasso", "ElasticNet", "RandomForest", "SVR", "KNeighbors", "XGBoost", "LightGBM", "CatBoost"
        ])
        st.write("Настройка гиперпараметров (базовые):")
        params = {}
        if alg in ["Ridge", "Lasso", "ElasticNet"]:
            params["alpha"] = float(st.number_input("alpha", min_value=0.0, value=1.0))
        if alg == "RandomForest":
            params["n_estimators"] = int(st.number_input("n_estimators", min_value=10, max_value=2000, value=100))
            params["max_depth"] = int(st.number_input("max_depth (0 для None)", min_value=0, max_value=100, value=0))
            if params["max_depth"] == 0:
                params.pop("max_depth", None)
        if alg == "SVR":
            params["C"] = float(st.number_input("C", min_value=0.001, value=1.0))
            kernel = st.selectbox("kernel", options=["rbf", "linear", "poly"], index=0)
            params["kernel"] = kernel
        if alg == "KNeighbors":
            params["n_neighbors"] = int(st.number_input("n_neighbors", min_value=1, max_value=100, value=5))
        if alg in ["XGBoost", "LightGBM", "CatBoost"]:
            params["n_estimators"] = int(st.number_input("n_estimators", min_value=10, max_value=2000, value=100))
            params["learning_rate"] = float(st.number_input("learning_rate", min_value=1e-4, max_value=1.0, value=0.1))
            params["max_depth"] = int(st.number_input("max_depth", min_value=1, max_value=20, value=6))

        # Train button
        if st.button("Обучить модель"):
            X = df[features]
            y = df[target]
            # Validate
            if X.isna().any().any():
                st.warning("В признаках есть пропуски — рекомендуется выполнить предобработку или заполнение пропусков.")
            if y.isna().any():
                st.error("Целевой столбец содержит пропуски — заполните или удалите их перед обучением.")
                return None, None

            # split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))
            st.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

            # instantiate model
            try:
                model = _get_model_by_name(alg, params)
            except Exception as e:
                st.error(f"Не удалось создать модель: {e}")
                return None, None

            # Cross-validation scores
            with st.spinner("Выполняется кросс-валидация..."):
                try:
                    cv_scores = {}
                    scoring = {"MAE": "neg_mean_absolute_error", "MSE": "neg_mean_squared_error", "R2": "r2"}
                    for name, sc in scoring.items():
                        scores = cross_val_score(model, X_train.fillna(0), y_train, cv=cv_folds, scoring=sc, n_jobs=-1)
                        # convert negative metrics
                        if name in ["MAE", "MSE"]:
                            scores = -scores
                        cv_scores[name] = {"mean": float(np.mean(scores)), "std": float(np.std(scores))}
                    st.write("CV результаты (Train):")
                    st.json(cv_scores)
                except Exception as e:
                    st.warning(f"CV не удалась: {e}")
                    cv_scores = {}

            # Fit final
            with st.spinner("Обучаю финальную модель..."):
                try:
                    model.fit(X_train.fillna(0), y_train)
                except Exception as e:
                    st.error(f"Ошибка при обучении модели: {e}")
                    return None, None

            # Metrics
            y_pred_train = model.predict(X_train.fillna(0))
            y_pred_test = model.predict(X_test.fillna(0))

            def metrics(y_true, y_pred):
                mae = mean_absolute_error(y_true, y_pred)
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_true, y_pred)
                n = len(y_true)
                p = X_train.shape[1]
                adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else None
                return {"MAE": float(mae), "MSE": float(mse), "RMSE": float(rmse), "R2": float(r2), "Adj_R2": (float(adj_r2) if adj_r2 is not None else None)}

            train_metrics = metrics(y_train, y_pred_train)
            test_metrics = metrics(y_test, y_pred_test)
            st.write("Метрики (train):")
            st.json(train_metrics)
            st.write("Метрики (test):")
            st.json(test_metrics)

            results = {
                "model": model,
                "features": features,
                "target": target,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "y_pred_train": y_pred_train,
                "y_pred_test": y_pred_test,
                "cv_scores": cv_scores
            }
            st.success("Модель обучена.")
            return model, results
        else:
            st.info("Настройте параметры и нажмите 'Обучить модель'.")
            return None, None
    except Exception as e:
        st.error(f"Ошибка в модуле обучения: {e}")
        return None, None
