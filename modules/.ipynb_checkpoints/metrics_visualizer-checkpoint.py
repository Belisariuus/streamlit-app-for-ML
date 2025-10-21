# modules/metrics_visualizer.py
"""
Модуль 5: Визуализация метрик
Построение графиков: фактические vs предсказанные, остатки, learning/validation curves (упрощённо).
"""
from typing import Any, Dict
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve, validation_curve

def visualize_metrics_interface(results: Dict[str, Any]) -> None:
    st.header("5. Визуализация метрик")
    try:
        model = results.get("model")
        X_train = results.get("X_train")
        X_test = results.get("X_test")
        y_train = results.get("y_train")
        y_test = results.get("y_test")
        y_pred_train = results.get("y_pred_train")
        y_pred_test = results.get("y_pred_test")

        st.subheader("Основные метрики")
        st.write("Train metrics:")
        st.json(results.get("train_metrics"))
        st.write("Test metrics:")
        st.json(results.get("test_metrics"))

        # Actual vs Predicted (test)
        st.subheader("Фактические vs Предсказанные (test)")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred_test, alpha=0.6)
        minv = min(min(y_test), min(y_pred_test))
        maxv = max(max(y_test), max(y_pred_test))
        ax.plot([minv, maxv], [minv, maxv], "--", linewidth=2)
        ax.set_xlabel("Фактические")
        ax.set_ylabel("Предсказанные")
        st.pyplot(fig)

        # Residuals
        st.subheader("Остатки (Residuals) vs Предсказанные")
        residuals = y_test - y_pred_test
        fig2, ax2 = plt.subplots()
        ax2.scatter(y_pred_test, residuals, alpha=0.6)
        ax2.axhline(0, color="red", linestyle="--")
        ax2.set_xlabel("Предсказанные")
        ax2.set_ylabel("Остатки")
        st.pyplot(fig2)

        # Distribution of residuals
        st.subheader("Распределение остатков")
        fig3, ax3 = plt.subplots()
        ax3.hist(residuals, bins=50)
        st.pyplot(fig3)

        # Feature importance (if available)
        st.subheader("Важность признаков / коэффициенты")
        try:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                feat = results.get("features")
                df_imp = pd.DataFrame({"feature": feat, "importance": importances}).sort_values("importance", ascending=False).head(30)
                fig4, ax4 = plt.subplots(figsize=(8,6))
                ax4.barh(df_imp["feature"][::-1], df_imp["importance"][::-1])
                ax4.set_title("Feature importances")
                st.pyplot(fig4)
            elif hasattr(model, "coef_"):
                coefs = model.coef_
                feat = results.get("features")
                df_coef = pd.DataFrame({"feature": feat, "coef": coefs}).sort_values("coef", key=abs, ascending=False).head(30)
                fig5, ax5 = plt.subplots(figsize=(8,6))
                ax5.barh(df_coef["feature"][::-1], df_coef["coef"][::-1])
                ax5.set_title("Коэффициенты модели")
                st.pyplot(fig5)
            else:
                st.info("Модель не предоставляет feature_importances_ или coef_.")
        except Exception as e:
            st.warning(f"Не удалось построить важности признаков: {e}")

        # Learning curve (sampled if large)
        st.subheader("Learning Curve")
        try:
            if X_train.shape[0] > 2000:
                X_sample = X_train.sample(2000, random_state=0)
                y_sample = y_train.loc[X_sample.index]
            else:
                X_sample = X_train
                y_sample = y_train
            train_sizes, train_scores, test_scores = learning_curve(estimator=model, X=X_sample.fillna(0), y=y_sample, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1,1.0,5))
            train_scores_mean = np.mean(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            fig6, ax6 = plt.subplots()
            ax6.plot(train_sizes, train_scores_mean, "o-", label="Train")
            ax6.plot(train_sizes, test_scores_mean, "o-", label="Validation")
            ax6.set_xlabel("Training examples")
            ax6.set_ylabel("Score")
            ax6.legend()
            st.pyplot(fig6)
        except Exception as e:
            st.warning(f"Не удалось построить learning curve: {e}")

    except Exception as e:
        st.error(f"Ошибка в визуализации метрик: {e}")
        st.exception(e)
