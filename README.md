# Проект: Бинарная классификация для предиктивного обслуживания оборудования

## Описание проекта
Цель проекта — разработать модель машинного обучения, которая предсказывает, произойдет ли отказ оборудования (Target = 1) или нет (Target = 0). Результаты работы оформлены в виде многостраничного Streamlit-приложения.

## Датасет
Используется датасет **"AI4I 2020 Predictive Maintenance Dataset"**, содержащий 10 000 записей с 12 признаками (в данной версии без `UDI` и `Product ID`). Подробное описание датасета можно найти в [документации](https://archive.ics.uci.edu/dataset/601/predictive+maintenance+dataset).

## Установка и запуск
1. Клонируйте репозиторий:
```
git clone https://github.com/Plokav/predictive_maintenance_project.git 

cd ~/predictive_maintenance_project

```
2. Создайте виртуальное окружение:
```
python3 -m venv venv

source venv/bin/activate
```
Для Windows:
```
venv\Scripts\activate
```
3. Установите все необходимые библеотеки:
```
 pip install -r requirements.txt
```
4. Запустите приложение:
```
streamlit run app.py
```
5. Откройте `http://localhost:8501` в браузере.

## Структура репозитория
- `app.py` - Основной файл приложения.
- `pages/analysis_and_model.py` - Страница с анализом данных и моделью.
- `pages/presentation.py` - Страница с презентацией проекта.
- `requirements.txt` - Файл с зависимостями.
- `data/predictive_maintenance.csv` - Датасет.
- `README.md` - Описание проекта.

## Метрики модели
- **Accuracy**: 0.997
- **ROC-AUC**: 0.969
- **Confusion Matrix**:``` [[1932 7] [ 23 38]]```

## Видео-демонстрация

Видео, демонстрирующее настройку и функционал приложения, доступно здесь:  
[Смотреть демонстрацию](video/demo.mp4)
