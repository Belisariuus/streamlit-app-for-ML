# Streamlit приложение

Интерактивное Streamlit-приложение для:
- Загрузки и детального анализа данных (CSV / Excel).
- Предобработки данных (импутация, кодирование, масштабирование, текстовые фичи).
- Обучения регрессионных моделей с CV.
- Визуализации метрик и важности признаков.
- Сохранения pipeline и модели.

---

## Структура проекта
- app.py - Главный файл приложения
- requirements.txt - Зависимости
- README.md - Документация
- modules/ - Модули приложения
- modules/init.py
- modules/data_loader.py - Модуль 1: Загрузка данных
- modules/data_explorer.py - Модуль 2: Разведочный анализ
- modules/data_preprocessor.py - Модуль 3: Предобработка
- modules/model_trainer.py - Модуль 4: Обучение модели
- modules/metrics_visualizer.py - Модуль 5: Визуализация метрик

---

## Установка

1. Клонируйте репозиторий или скопируйте файлы.
2. Создайте виртуальное окружение (рекомендуется):
```bash
python -m venv venv
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows
```
3. Установите зависимости:
```bash
pip install -r requirements.txt
```
4. Запустите приложение:
```bash
streamlit run app.py
```
---