import streamlit as st

# Конфигурация интерфейса
st.set_page_config(
    page_title="Система предиктивного обслуживания",
    layout="wide"
)

# Навигация между страницами
navigator = st.navigation([
    st.Page("pages/analysis_and_model.py", title="Обзор и модель"),
    st.Page("pages/presentation.py", title="Слайд-презентация")
])

# Запуск выбранной страницы
navigator.run()
