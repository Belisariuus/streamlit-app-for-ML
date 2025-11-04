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
from sklearn.metrics import roc_auc_score
from scipy import integrate

def calculate_gini(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Расчет коэффициента Джинни для бинарной классификации
    Gini = 2 * AUC - 1
    """
    try:
        if len(np.unique(y_true)) == 2:  # Бинарная классификация
            auc = roc_auc_score(y_true, y_pred)
            gini = 2 * auc - 1
            return gini
        else:
            return None
    except Exception:
        return None

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
        problem_type = results.get("problem_type", "regression")

        st.subheader("Основные метрики")
        st.write("Train metrics:")
        st.json(results.get("train_metrics"))
        st.write("Test metrics:")
        st.json(results.get("test_metrics"))

        # Расчет и проверка коэффициента Джинни для классификации
        if problem_type == "classification" and len(np.unique(y_test)) == 2:
            try:
                # Для моделей классификации получаем вероятности для положительного класса
                if hasattr(model, "predict_proba"):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                else:
                    # Если нет predict_proba, используем предсказания как псевдо-вероятности
                    y_pred_proba = y_pred_test

                gini_coefficient = calculate_gini(y_test, y_pred_proba)

                if gini_coefficient is not None:
                    st.metric("Коэффициент Джинни", f"{gini_coefficient:.4f}")

                    # Проверка качества модели по коэффициенту Джинни
                    if gini_coefficient < 0.45:
                        st.error("🚨 **ВНИМАНИЕ: Коэффициент Джинни ниже 0.45**")
                        st.warning("""
                        **Рекомендации:**
                        - Модель имеет низкое качество предсказаний
                        - Проверьте баланс классов в данных
                        - Рассмотрите возможность feature engineering
                        - Попробуйте другие алгоритмы или настройку гиперпараметров
                        - Убедитесь в релевантности признаков для целевой переменной
                        """)
                    elif gini_coefficient < 0.6:
                        st.warning("⚠️ **Коэффициент Джинни в диапазоне 0.45-0.6 - удовлетворительное качество**")
                    elif gini_coefficient < 0.75:
                        st.info("✅ **Коэффициент Джинни в диапазоне 0.6-0.75 - хорошее качество**")
                    else:
                        st.success("🎉 **Коэффициент Джинни выше 0.75 - отличное качество!**")

                    # Дополнительная информация о интерпретации
                    with st.expander("📊 Как интерпретировать коэффициент Джинни?"):
                        st.markdown("""
                        **Шкала качества модели по коэффициенту Джинни:**
                        - **< 0.45**: Низкое качество - модель работает ненамного лучше случайного угадывания
                        - **0.45-0.60**: Удовлетворительное качество - модель имеет некоторую предсказательную силу
                        - **0.60-0.75**: Хорошее качество - модель хорошо разделяет классы
                        - **> 0.75**: Отличное качество - модель очень точно предсказывает классы
                        
                        *Примечание: Коэффициент Джинни = 2 × AUC - 1*
                        """)

            except Exception as e:
                st.warning(f"Не удалось рассчитать коэффициент Джинни: {e}")

        # Actual vs Predicted (test)
        st.subheader("Фактические vs Предсказанные (test)")
        fig, ax = plt.subplots()

        if problem_type == "regression":
            ax.scatter(y_test, y_pred_test, alpha=0.6)
            minv = min(min(y_test), min(y_pred_test))
            maxv = max(max(y_test), max(y_pred_test))
            ax.plot([minv, maxv], [minv, maxv], "--", linewidth=2, color='red')
            ax.set_xlabel("Фактические значения")
            ax.set_ylabel("Предсказанные значения")
            ax.set_title("Фактические vs Предсказанные значения")
        else:
            # Для классификации - confusion matrix-like visualization
            unique_classes = np.unique(y_test)
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))

            for i, cls in enumerate(unique_classes):
                mask = y_test == cls
                ax.scatter(y_test[mask], y_pred_test[mask], alpha=0.6,
                           color=colors[i], label=f'Class {cls}')

            ax.set_xlabel("Фактические классы")
            ax.set_ylabel("Предсказанные классы")
            ax.set_title("Фактические vs Предсказанные классы")
            ax.legend()

        st.pyplot(fig)

        # Residuals (только для регрессии)
        if problem_type == "regression":
            st.subheader("Остатки (Residuals) vs Предсказанные")
            residuals = y_test - y_pred_test
            fig2, ax2 = plt.subplots()
            ax2.scatter(y_pred_test, residuals, alpha=0.6)
            ax2.axhline(0, color="red", linestyle="--", linewidth=2)
            ax2.set_xlabel("Предсказанные значения")
            ax2.set_ylabel("Остатки")
            ax2.set_title("Остатки vs Предсказанные значения")
            st.pyplot(fig2)

            # Distribution of residuals
            st.subheader("Распределение остатков")
            fig3, ax3 = plt.subplots()
            ax3.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
            ax3.axvline(0, color="red", linestyle="--", linewidth=2)
            ax3.set_xlabel("Остатки")
            ax3.set_ylabel("Частота")
            ax3.set_title("Распределение остатков")
            st.pyplot(fig3)

        # Feature importance (if available)
        st.subheader("Важность признаков / коэффициенты")
        try:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                feat = results.get("features", [f"Feature_{i}" for i in range(len(importances))])
                df_imp = pd.DataFrame({"feature": feat, "importance": importances}).sort_values("importance", ascending=False).head(20)
                fig4, ax4 = plt.subplots(figsize=(10, 8))
                ax4.barh(df_imp["feature"][::-1], df_imp["importance"][::-1])
                ax4.set_xlabel("Важность")
                ax4.set_title("Важность признаков")
                plt.tight_layout()
                st.pyplot(fig4)

                # Таблица с важностями
                st.write("Топ-10 самых важных признаков:")
                st.dataframe(df_imp.head(10).style.format({'importance': '{:.4f}'}))

            elif hasattr(model, "coef_"):
                coefs = model.coef_
                # Обработка для многоклассовой классификации
                if len(coefs.shape) > 1:
                    coefs = np.mean(np.abs(coefs), axis=0)  # Средняя абсолютная важность по классам

                feat = results.get("features", [f"Feature_{i}" for i in range(len(coefs))])
                df_coef = pd.DataFrame({"feature": feat, "coef": coefs}).sort_values("coef", key=abs, ascending=False).head(20)
                fig5, ax5 = plt.subplots(figsize=(10, 8))
                ax5.barh(df_coef["feature"][::-1], df_coef["coef"][::-1])
                ax5.set_xlabel("Коэффициент")
                ax5.set_title("Коэффициенты модели")
                plt.tight_layout()
                st.pyplot(fig5)

                # Таблица с коэффициентами
                st.write("Топ-10 признаков с наибольшими абсолютными коэффициентами:")
                st.dataframe(df_coef.head(10).style.format({'coef': '{:.4f}'}))
            else:
                st.info("Модель не предоставляет feature_importances_ или coef_.")
        except Exception as e:
            st.warning(f"Не удалось построить важности признаков: {e}")

        # Learning curve (sampled if large)
        st.subheader("Кривая обучения (Learning Curve)")
        try:
            if X_train.shape[0] > 2000:
                X_sample = X_train.sample(2000, random_state=0)
                y_sample = y_train.loc[X_sample.index]
            else:
                X_sample = X_train
                y_sample = y_train

            # Выбор метрики в зависимости от типа задачи
            scoring = 'accuracy' if problem_type == 'classification' else 'r2'
            train_sizes, train_scores, test_scores = learning_curve(
                estimator=model,
                X=X_sample.fillna(0),
                y=y_sample,
                cv=5,
                n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 5),
                scoring=scoring
            )

            train_scores_mean = np.mean(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)

            fig6, ax6 = plt.subplots(figsize=(10, 6))
            ax6.plot(train_sizes, train_scores_mean, "o-", label="Обучающая выборка")
            ax6.plot(train_sizes, test_scores_mean, "o-", label="Валидационная выборка")
            ax6.set_xlabel("Размер обучающей выборки")
            ax6.set_ylabel("Score" if problem_type == 'regression' else "Accuracy")
            ax6.set_title("Кривая обучения")
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            st.pyplot(fig6)

            # Анализ переобучения
            final_train_score = train_scores_mean[-1]
            final_test_score = test_scores_mean[-1]
            gap = final_train_score - final_test_score

            if gap > 0.1:
                st.warning("⚠️ **Возможное переобучение**: Большая разница между train и validation score")
            elif gap < -0.1:
                st.warning("⚠️ **Возможное недообучение**: Validation score значительно выше train score")

        except Exception as e:
            st.warning(f"Не удалось построить learning curve: {e}")

        # Дополнительная информация о модели
        with st.expander("📋 Детальная информация о модели"):
            st.write("**Параметры модели:**")
            st.json(model.get_params())

            st.write("**Информация о данных:**")
            st.write(f"- Обучающая выборка: {X_train.shape}")
            st.write(f"- Тестовая выборка: {X_test.shape}")
            st.write(f"- Количество признаков: {len(results.get('features', []))}")
            if problem_type == 'classification':
                st.write(f"- Количество классов: {len(np.unique(y_train))}")

    except Exception as e:
        st.error(f"Ошибка в визуализации метрик: {e}")
        st.exception(e)