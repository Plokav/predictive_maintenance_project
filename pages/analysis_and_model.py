import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test):
    """
    Оценка модели: возвращает основные метрики и вероятности для ROC
    """
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)
        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_proba)
    return accuracy, conf_matrix, class_report, roc_auc, y_pred, y_proba

def main():
    st.title("Анализ данных и обучение моделей")

    uploaded_file = st.file_uploader("Загрузите CSV-файл с данными", type="csv")
    if uploaded_file is None:
        st.info("Пожалуйста, загрузите файл с данными, чтобы начать")
        return

    data = pd.read_csv(uploaded_file)

    # Удаляем ненужные столбцы
    drop_cols = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    data.drop(columns=drop_cols, inplace=True, errors='ignore')

    # Кодируем категориальный признак Type
    label_encoder = LabelEncoder()
    data['Type'] = label_encoder.fit_transform(data['Type'])

    # Проверяем пропуски
    if data.isnull().any().any():
        st.warning("Обнаружены пропуски в данных, они будут удалены")
        data.dropna(inplace=True)

    # Масштабируем числовые признаки
    numeric_features = ['Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear']
    scaler = StandardScaler()
    data[numeric_features] = scaler.fit_transform(data[numeric_features])

    X = data.drop('Machine failure', axis=1)
    y = data['Machine failure']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "SVM": SVC(kernel='linear', probability=True, random_state=42),
    }

    results = {}
    best_model_name = None
    best_roc_auc = 0
    best_model = None
    best_conf_matrix = None
    best_class_report_df = None
    best_accuracy = None

    st.header("Обучение и оценка моделей")

    for name, model in models.items():
        st.subheader(name)
        model.fit(X_train, y_train)
        accuracy, conf_matrix, class_report_dict, roc_auc, y_pred, y_proba = evaluate_model(model, X_test, y_test)

        st.write(f"Accuracy: {accuracy:.3f}")
        st.write(f"ROC AUC: {roc_auc:.3f}")

        st.write("Confusion Matrix:")
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        class_report_df = pd.DataFrame(class_report_dict).transpose().round(2)
        class_report_df.rename(index={
            '0': 'Class 0',
            '1': 'Class 1',
            'accuracy': 'Accuracy',
            'macro avg': 'Macro Avg',
            'weighted avg': 'Weighted Avg'
        }, inplace=True)
        class_report_df.rename(columns={
            'precision': 'Precision',
            'recall': 'Recall',
            'f1-score': 'F1-score',
            'support': 'Support'
        }, inplace=True)

        st.write("Classification Report:")
        st.dataframe(class_report_df)

        results[name] = (roc_auc, y_proba)

        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model_name = name
            best_model = model
            best_conf_matrix = conf_matrix
            best_class_report_df = class_report_df
            best_accuracy = accuracy

    st.markdown("---")
    st.header(f"Best Model: {best_model_name} (ROC AUC = {best_roc_auc:.3f})")
    st.write(f"Accuracy: {best_accuracy:.3f}")

    st.write("Confusion Matrix of Best Model:")
    fig, ax = plt.subplots()
    sns.heatmap(best_conf_matrix, annot=True, fmt='d', cmap='Greens', ax=ax)
    st.pyplot(fig)

    st.write("Classification Report of Best Model:")
    st.dataframe(best_class_report_df)

    plt.figure()
    for name, (roc_auc, y_proba) in results.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves Comparison")
    plt.legend()
    st.pyplot(plt)

    st.header("Predict Equipment Failure on New Data")

    with st.form("prediction_form"):
        st.write("Input equipment parameters:")
        type_input = st.selectbox("Product Type", ["L", "M", "H"])
        air_temp = st.number_input("Air Temperature [K]", value=300.0)
        process_temp = st.number_input("Process Temperature [K]", value=310.0)
        rotational_speed = st.number_input("Rotational Speed [rpm]", value=1500)
        torque = st.number_input("Torque [Nm]", value=40.0)
        tool_wear = st.number_input("Tool Wear [min]", value=100)

        submit = st.form_submit_button("Predict")

        if submit:
            type_encoded = label_encoder.transform([type_input])[0]
            input_df = pd.DataFrame({
                'Type': [type_encoded],
                'Air temperature': [air_temp],
                'Process temperature': [process_temp],
                'Rotational speed': [rotational_speed],
                'Torque': [torque],
                'Tool wear': [tool_wear]
            })

            input_df[numeric_features] = scaler.transform(input_df[numeric_features])

            prediction = best_model.predict(input_df)[0]
            proba = best_model.predict_proba(input_df)[0, 1]

            st.write(f"Prediction: {'Failure' if prediction == 1 else 'No Failure'}")
            st.write(f"Probability of failure: {proba:.2f}")

if __name__ == "__main__":
    main()