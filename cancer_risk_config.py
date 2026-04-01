
from __future__ import annotations

from collections import OrderedDict

DATA_URL = "https://drive.google.com/uc?id=1Iy5Ke29bYZmIdbVw2TmtlbaJ-39XZ0jN"

# Столбцы в исходном датасете из ноутбука
RAW_TO_UNIFIED_COLUMNS = {
    "Patient_ID": "patient_id",
    "Cancer_Type": "cancer_type",
    "Age": "age",
    "Gender": "gender",
    "Smoking": "smoking",
    "Alcohol_Use": "alcohol_use",
    "Obesity": "obesity",
    "Family_History": "family_history",
    "Diet_Red_Meat": "diet_red_meat",
    "Diet_Salted_Processed": "diet_salted_processed",
    "Fruit_Veg_Intake": "fruit_veg_intake",
    "Physical_Activity": "physical_activity",
    "Air_Pollution": "air_pollution",
    "Occupational_Hazards": "occupational_hazards",
    "BRCA_Mutation": "brca_mutation",
    "H_Pylori_Infection": "h_pylori_infection",
    "Calcium_Intake": "calcium_intake",
    "Overall_Risk_Score": "overall_risk_score",
    "BMI": "bmi",
    "Physical_Activity_Level": "physical_activity_level",
    "Risk_Level": "risk_level",
}

UNIFIED_TO_RAW_COLUMNS = {v: k for k, v in RAW_TO_UNIFIED_COLUMNS.items()}

FEATURE_COLUMNS = [
    "age",
    "bmi",
    "gender",
    "family_history",
    "brca_mutation",
    "h_pylori_infection",
    "smoking",
    "alcohol_use",
    "obesity",
    "diet_red_meat",
    "diet_salted_processed",
    "fruit_veg_intake",
    "physical_activity",
    "physical_activity_level",
    "air_pollution",
    "occupational_hazards",
    "calcium_intake",
]

NUMERIC_FEATURES = [
    "age",
    "smoking",
    "alcohol_use",
    "obesity",
    "diet_red_meat",
    "diet_salted_processed",
    "fruit_veg_intake",
    "physical_activity",
    "air_pollution",
    "occupational_hazards",
    "calcium_intake",
    "bmi",
    "physical_activity_level",
]

CATEGORICAL_FEATURES = [
    "gender",
    "family_history",
    "brca_mutation",
    "h_pylori_infection",
]

TARGET_COLUMN = "overall_risk_score"

CANCER_TYPE_MAP = OrderedDict(
    [
        ("Breast", "Рак молочной железы"),
        ("Prostate", "Рак простаты"),
        ("Skin", "Рак кожи"),
        ("Colon", "Рак толстой кишки"),
        ("Lung", "Рак лёгкого"),
    ]
)

CANCER_SLUGS = {
    "Breast": "breast",
    "Prostate": "prostate",
    "Skin": "skin",
    "Colon": "colon",
    "Lung": "lung",
}

MODEL_ARTIFACT_TEMPLATE = "models/{slug}_risk_model.joblib"

FORM_DEFAULTS = {
    "age": 45,
    "bmi": 24.0,
    "gender": 0,
    "family_history": 0,
    "brca_mutation": 0,
    "h_pylori_infection": 0,
    "smoking": 0,
    "alcohol_use": 0,
    "obesity": 0,
    "diet_red_meat": 5,
    "diet_salted_processed": 5,
    "fruit_veg_intake": 5,
    "physical_activity": 5,
    "physical_activity_level": 5,
    "air_pollution": 3,
    "occupational_hazards": 2,
    "calcium_intake": 5,
}

RUSSIAN_LABELS = {
    "age": "Возраст",
    "bmi": "ИМТ",
    "gender": "Пол",
    "family_history": "Семейная история онкозаболеваний",
    "brca_mutation": "Мутация BRCA",
    "h_pylori_infection": "Инфекция H. pylori",
    "smoking": "Курение",
    "alcohol_use": "Употребление алкоголя",
    "obesity": "Оценка ожирения",
    "diet_red_meat": "Употребление красного мяса",
    "diet_salted_processed": "Употребление солёной и обработанной пищи",
    "fruit_veg_intake": "Потребление фруктов и овощей",
    "physical_activity": "Частота физической активности",
    "physical_activity_level": "Интенсивность физической активности",
    "air_pollution": "Загрязнение воздуха",
    "occupational_hazards": "Профессиональные вредности",
    "calcium_intake": "Потребление кальция",
}

RUSSIAN_HELP = {
    "gender": "0 = женский, 1 = мужской в исходных данных. В приложении это скрыто за выбором пользователя.",
    "family_history": "Да/нет — наличие семейной истории онкологических заболеваний.",
    "brca_mutation": "Да/нет — наличие мутации BRCA.",
    "h_pylori_infection": "Да/нет — наличие инфекции H. pylori.",
    "smoking": "Шкала от 0 до 10.",
    "alcohol_use": "Шкала от 0 до 10.",
    "obesity": "Шкала от 0 до 10.",
    "diet_red_meat": "Шкала от 0 до 10.",
    "diet_salted_processed": "Шкала от 0 до 10.",
    "fruit_veg_intake": "Шкала от 0 до 10, где большие значения означают большее потребление.",
    "physical_activity": "Шкала от 0 до 10: как часто.",
    "physical_activity_level": "Шкала от 0 до 10: насколько интенсивно.",
    "air_pollution": "Шкала от 0 до 10.",
    "occupational_hazards": "Шкала от 0 до 10.",
    "calcium_intake": "Шкала от 0 до 10.",
}
