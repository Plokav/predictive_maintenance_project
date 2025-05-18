import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

# Настройка заголовка страницы
st.title("Анализ и прогнозирование отказов оборудования")

# Шаг 1: Загрузка данных
st.subheader("Шаг 1: Загрузка данных")
uploaded_file = st.file_uploader("Выберите CSV-файл с данными", type=["csv"])

if uploaded_file is not None:
    # Чтение загруженного файла
    data = pd.read_csv(uploaded_file)
    st.session_state["raw_data"] = data
    st.write("Данные загружены успешно!")
    st.write(data.head())

    # Шаг 2: Предобработка данных
    st.subheader("Шаг 2: Предобработка данных")
    if st.button("Предобработать данные"):
        columns_to_remove = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        if all(col in data.columns for col in columns_to_remove):
            processed_data = data.drop(columns=columns_to_remove)
        else:
            processed_data = data.copy()
            st.warning("Некоторые столбцы для удаления отсутствуют.")

        if 'Type' in processed_data.columns:
            encoder = LabelEncoder()
            processed_data['Type'] = encoder.fit_transform(processed_data['Type'])

        cols_to_scale = ['Air temperature', 'Process temperature', 'Rotational speed', 'Torque']
        available_cols = [col for col in cols_to_scale if col in processed_data.columns]
        if available_cols:
            scaler = StandardScaler()
            processed_data[available_cols] = scaler.fit_transform(processed_data[available_cols])

        st.session_state["processed_data"] = processed_data
        st.write("Данные предобработаны:")
        st.write(processed_data.head())

    # Шаг 3: Обучение модели
    if "processed_data" in st.session_state:
        st.subheader("Шаг 3: Обучение модели")
        processed_data = st.session_state["processed_data"]
        if 'Machine failure' not in processed_data.columns:
            st.error("Целевой столбец 'Machine failure' отсутствует.")
        else:
            # Разделение данных на обучающую и тестовую выборки (80/20)
            X = processed_data.drop('Machine failure', axis=1)
            y = processed_data['Machine failure']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Выбор модели
            model_choice = st.selectbox("Выберите модель:", ["Logistic Regression", "Random Forest", "XGBoost"])
            if st.button("Обучить модель"):
                if model_choice == "Logistic Regression":
                    model = LogisticRegression()
                elif model_choice == "Random Forest":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
                
                # Обучение модели на обучающей выборке
                model.fit(X_train, y_train)
                st.session_state["model"] = model

                # Предсказания на тестовой выборке
                test_predictions = model.predict(X_test)
                test_acc = accuracy_score(y_test, test_predictions)
                test_cm = confusion_matrix(y_test, test_predictions)
                test_roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

                # Предсказания на всей базе данных
                full_predictions = model.predict(X)
                full_acc = accuracy_score(y, full_predictions)

                # Вывод результатов
                st.success(f"Модель {model_choice} обучена!")
                st.write(f"Точность на тестовой выборке (Accuracy): {test_acc:.4f}")
                st.write("Матрица ошибок на тестовой выборке (Confusion Matrix):")
                st.write(test_cm)
                st.write(f"ROC-AUC на тестовой выборке: {test_roc_auc:.4f}")
                st.write(f"Точность на всей базе данных (Accuracy): {full_acc:.4f}")

    # Шаг 4: Прогнозирование
    if "model" in st.session_state and "processed_data" in st.session_state:
        st.subheader("Шаг 4: Прогнозирование")
        X = st.session_state["processed_data"].drop('Machine failure', axis=1)
        input_features = st.text_area("Введите значения признаков (через пробел, порядок: Type Air temperature Process temperature Rotational speed Torque Tool wear):")
        if st.button("Выполнить предсказание") and input_features:
            try:
                features = [float(x) for x in input_features.split()]
                if len(features) == 6:
                    input_df = pd.DataFrame([features], columns=['Type', 'Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear'])
                    prediction = st.session_state["model"].predict(input_df)
                    st.write(f"Результат: {'Отказ' if prediction[0] == 1 else 'Нет отказа'}")
                else:
                    st.error("Неверное количество признаков. Ожидается 6 значений.")
            except ValueError:
                st.error("Пожалуйста, введите числовые значения.")

    # Шаг 5: Визуализация
    if "processed_data" in st.session_state:
        st.subheader("Шаг 5: Визуализация: Распределение крутящего момента")
        st.line_chart(st.session_state["processed_data"]['Torque'].head(50))
