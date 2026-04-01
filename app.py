from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from cancer_risk_config import (
    CANCER_SLUGS,
    CANCER_TYPE_MAP,
    FEATURE_COLUMNS,
    FORM_DEFAULTS,
    MODEL_ARTIFACT_TEMPLATE,
)

st.set_page_config(
    page_title="Оценка риска развития рака",
    page_icon="🩺",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parent
AUTHOR_TEXT = "Автор проекта: Онучина Арина, школа МАОУ СОШ № 57, Калининград"

CANCER_RECOMMENDATIONS = {
    "Рак молочной железы": (
        "Поддерживайте регулярную физическую активность, следите за массой тела, "
        "обсуждайте с врачом семейный анамнез и не пропускайте профилактические осмотры."
    ),
    "Рак простаты": (
        "Полезны умеренная физическая активность, контроль веса, внимательное отношение "
        "к возрастным изменениям и плановые консультации у врача по мере взросления."
    ),
    "Рак кожи": (
        "Старайтесь защищать кожу от избыточного солнца, использовать средства с SPF, "
        "избегать ожогов и наблюдать за заметными изменениями родинок или кожи."
    ),
    "Рак толстой кишки": (
        "Обычно помогают более сбалансированное питание, достаточная физическая активность, "
        "умеренность в обработанной пище и своевременные профилактические обследования."
    ),
    "Рак лёгкого": (
        "Наиболее полезны отказ от курения, уменьшение контакта с загрязнённым воздухом и "
        "внимание к условиям труда и длительным респираторным жалобам."
    ),
}

FACTOR_HELP = {
    "age": "Возраст: с возрастом риск многих заболеваний, включая онкологические, обычно повышается.",
    "bmi": "ИМТ — индекс массы тела. Помогает оценить, есть ли избыток или дефицит массы тела.",
    "gender": "Пол: для некоторых видов рака риск и распространённость различаются у мужчин и женщин.",
    "family_history": "Семейная история: наличие онкологических заболеваний у близких родственников может повышать риск.",
    "brca_mutation": "Мутация BRCA: наследственное изменение в генах BRCA1/BRCA2, связанное с повышенным риском некоторых видов рака.",
    "h_pylori_infection": "H. pylori — бактерия, которая может жить в желудке; в некоторых случаях связана с повышенным риском заболеваний желудка.",
    "smoking": "Курение: один из самых известных факторов, повышающих риск ряда онкологических заболеваний, особенно рака лёгкого.",
    "alcohol_use": "Употребление алкоголя: при высоком уровне может повышать риск некоторых видов рака.",
    "obesity": "Ожирение: избыточная масса тела связана с повышением риска ряда заболеваний, включая некоторые виды рака.",
    "diet_red_meat": "Красное мясо: очень частое употребление может быть неблагоприятным фактором для некоторых видов рака.",
    "diet_salted_processed": "Солёная и обработанная пища: избыток такой еды считается неблагоприятным фактором для здоровья.",
    "fruit_veg_intake": "Фрукты и овощи: более регулярное потребление обычно считается защитным фактором.",
    "physical_activity": "Частота физической активности: как часто человек двигается, тренируется или занимается нагрузкой.",
    "physical_activity_level": "Интенсивность физической активности: насколько нагрузка сильная или энергозатратная.",
    "air_pollution": "Загрязнение воздуха: длительное воздействие загрязнённого воздуха может вредить организму.",
    "occupational_hazards": "Профессиональные вредности: воздействие пыли, химических веществ, дыма и других вредных факторов на работе.",
    "calcium_intake": "Потребление кальция: количество кальция в питании, например из молочных продуктов, рыбы, зелени или добавок.",
}


@st.cache_resource
def load_models() -> dict:
    bundles = {}
    missing = []

    for cancer_type in CANCER_TYPE_MAP:
        slug = CANCER_SLUGS[cancer_type]
        artifact_path = PROJECT_ROOT / MODEL_ARTIFACT_TEMPLATE.format(slug=slug)
        if not artifact_path.exists():
            missing.append(str(artifact_path))
            continue

        bundles[cancer_type] = joblib.load(artifact_path)

    if missing:
        missing_text = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Не найдены файлы моделей. Сначала запусти train_models.py и закоммить папку models/.\n"
            f"{missing_text}"
        )

    return bundles


def make_input_frame(user_values: dict) -> pd.DataFrame:
    df = pd.DataFrame([user_values])
    return df[FEATURE_COLUMNS]


def clip_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def score_to_percent(value: float) -> float:
    return round(clip_score(value) * 100, 2)


def interpret_score(score_0_1: float) -> tuple[str, str]:
    if score_0_1 < 0.33:
        return "Низкий", "🟢"
    if score_0_1 < 0.66:
        return "Умеренный", "🟠"
    return "Высокий", "🔴"


def build_sidebar() -> None:
    st.sidebar.title("О приложении")
    st.sidebar.write(
        "Это демонстрационное приложение на основе пяти отдельных ML-моделей, "
        "каждая из которых оценивает риск для своего типа рака."
    )
    st.sidebar.info(
        "Важно: результат не является диагнозом и не заменяет консультацию врача."
    )
    st.sidebar.markdown(
        """
        **Как интерпретировать шкалы 0–10**
        - 0: фактор практически отсутствует
        - 10: фактор выражен максимально
        """
    )


def main() -> None:
    build_sidebar()

    st.title("🩺 Оценка риска развития онкологических заболеваний")
    st.caption(AUTHOR_TEXT)
    st.write(
        "Заполните анкету, и приложение покажет прогнозируемую оценку риска "
        "по пяти типам рака из проекта."
    )

    with st.expander("Какие признаки используются в моделях?"):
        st.markdown(
            """
            - Возраст и ИМТ  
            - Пол  
            - Семейная история онкозаболеваний  
            - Мутация BRCA  
            - Инфекция *H. pylori*  
            - Курение, алкоголь, ожирение  
            - Пищевые привычки  
            - Физическая активность  
            - Загрязнение воздуха  
            - Профессиональные вредности  
            - Потребление кальция  
            """
        )
        st.caption("У каждого поля в анкете есть значок подсказки: наведите курсор, чтобы увидеть краткое объяснение.")

    try:
        model_bundles = load_models()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    with st.form("risk_form"):
        st.subheader("1. Базовые данные")
        c1, c2 = st.columns(2)

        with c1:
            age = st.number_input(
                "Возраст",
                min_value=18,
                max_value=100,
                value=FORM_DEFAULTS["age"],
                step=1,
                help=FACTOR_HELP["age"],
            )

            with st.expander("Калькулятор ИМТ"):
                height_cm = st.number_input("Рост, см", min_value=100, max_value=250, value=170, step=1)
                weight_kg = st.number_input("Вес, кг", min_value=25.0, max_value=300.0, value=70.0, step=0.1)
                calculated_bmi = round(weight_kg / ((height_cm / 100) ** 2), 2)
                st.write(f"Рассчитанный ИМТ: **{calculated_bmi}**")
                st.caption("ИМТ помогает примерно оценить соотношение роста и массы тела.")
                use_calculated_bmi = st.checkbox("Подставить рассчитанный ИМТ в анкету", value=True)

            bmi_manual = st.number_input(
                "ИМТ",
                min_value=10.0,
                max_value=60.0,
                value=float(FORM_DEFAULTS["bmi"]),
                step=0.1,
                help=FACTOR_HELP["bmi"],
            )
            gender_label = st.radio(
                "Пол",
                ["Женский", "Мужской"],
                horizontal=True,
                help=FACTOR_HELP["gender"],
            )
            family_history_label = st.radio(
                "Семейная история онкозаболеваний",
                ["Нет", "Да"],
                horizontal=True,
                help=FACTOR_HELP["family_history"],
            )

        with c2:
            brca_label = st.radio(
                "Мутация BRCA",
                ["Нет", "Да", "Не знаю"],
                horizontal=True,
                help=FACTOR_HELP["brca_mutation"],
            )
            h_pylori_label = st.radio(
                "Инфекция H. pylori",
                ["Нет", "Да", "Не знаю"],
                horizontal=True,
                help=FACTOR_HELP["h_pylori_infection"],
            )
            st.caption("Ответ «Не знаю» в расчёте будет интерпретирован как «Нет».")

        st.subheader("2. Факторы образа жизни и среды")
        c3, c4 = st.columns(2)

        with c3:
            smoking = st.slider(
                "Курение",
                0,
                10,
                FORM_DEFAULTS["smoking"],
                help=FACTOR_HELP["smoking"],
            )
            alcohol_use = st.slider(
                "Употребление алкоголя",
                0,
                10,
                FORM_DEFAULTS["alcohol_use"],
                help=FACTOR_HELP["alcohol_use"],
            )
            obesity = st.slider(
                "Оценка ожирения",
                0,
                10,
                FORM_DEFAULTS["obesity"],
                help=FACTOR_HELP["obesity"],
            )
            diet_red_meat = st.slider(
                "Употребление красного мяса",
                0,
                10,
                FORM_DEFAULTS["diet_red_meat"],
                help=FACTOR_HELP["diet_red_meat"],
            )
            diet_salted_processed = st.slider(
                "Употребление солёной и обработанной пищи",
                0,
                10,
                FORM_DEFAULTS["diet_salted_processed"],
                help=FACTOR_HELP["diet_salted_processed"],
            )
            fruit_veg_intake = st.slider(
                "Потребление фруктов и овощей",
                0,
                10,
                FORM_DEFAULTS["fruit_veg_intake"],
                help=FACTOR_HELP["fruit_veg_intake"],
            )

        with c4:
            physical_activity = st.slider(
                "Частота физической активности",
                0,
                10,
                FORM_DEFAULTS["physical_activity"],
                help=FACTOR_HELP["physical_activity"],
            )
            physical_activity_level = st.slider(
                "Интенсивность физической активности",
                0,
                10,
                FORM_DEFAULTS["physical_activity_level"],
                help=FACTOR_HELP["physical_activity_level"],
            )
            air_pollution = st.slider(
                "Загрязнение воздуха",
                0,
                10,
                FORM_DEFAULTS["air_pollution"],
                help=FACTOR_HELP["air_pollution"],
            )
            occupational_hazards = st.slider(
                "Профессиональные вредности",
                0,
                10,
                FORM_DEFAULTS["occupational_hazards"],
                help=FACTOR_HELP["occupational_hazards"],
            )
            calcium_intake = st.slider(
                "Потребление кальция",
                0,
                10,
                FORM_DEFAULTS["calcium_intake"],
                help=FACTOR_HELP["calcium_intake"],
            )

        submitted = st.form_submit_button("Рассчитать риск", use_container_width=True)

    if not submitted:
        return

    bmi_value = calculated_bmi if use_calculated_bmi else bmi_manual
    is_female = gender_label == "Женский"
    is_male = gender_label == "Мужской"

    user_values = {
        "age": age,
        "bmi": bmi_value,
        "gender": 1 if gender_label == "Мужской" else 0,
        "family_history": 1 if family_history_label == "Да" else 0,
        "brca_mutation": 1 if brca_label == "Да" else 0,
        "h_pylori_infection": 1 if h_pylori_label == "Да" else 0,
        "smoking": smoking,
        "alcohol_use": alcohol_use,
        "obesity": obesity,
        "diet_red_meat": diet_red_meat,
        "diet_salted_processed": diet_salted_processed,
        "fruit_veg_intake": fruit_veg_intake,
        "physical_activity": physical_activity,
        "physical_activity_level": physical_activity_level,
        "air_pollution": air_pollution,
        "occupational_hazards": occupational_hazards,
        "calcium_intake": calcium_intake,
    }

    input_df = make_input_frame(user_values)

    rows = []
    for cancer_type, cancer_name_ru in CANCER_TYPE_MAP.items():
        if is_female and cancer_name_ru == "Рак простаты":
            continue
        if is_male and cancer_name_ru == "Рак молочной железы":
            continue

        bundle = model_bundles[cancer_type]
        model = bundle["model"]
        score_0_1 = float(model.predict(input_df)[0])
        percent = score_to_percent(score_0_1)
        label, icon = interpret_score(score_0_1)

        rows.append(
            {
                "Тип рака": cancer_name_ru,
                "Оценка риска": score_0_1,
                "Риск, %": percent,
                "Категория": f"{icon} {label}",
                "Категория_текст": label,
                "Лучшая модель": bundle["metadata"]["selected_model"],
            }
        )

    results_df = pd.DataFrame(rows).sort_values("Риск, %", ascending=False).reset_index(drop=True)

    st.subheader("Результаты")
    st.dataframe(
        results_df[["Тип рака", "Риск, %", "Категория", "Лучшая модель"]],
        use_container_width=True,
        hide_index=True,
    )

    st.success(f"Для расчёта использован ИМТ: **{bmi_value}**")

    if is_female:
        st.info("Для женского пола оценка риска рака простаты не рассчитывается и не показывается в результатах.")
    if is_male:
        st.info("Для мужского пола оценка риска рака молочной железы не рассчитывается и не показывается.")

    top_row = results_df.iloc[0]
    st.warning(
        f"Наибольшая прогнозируемая оценка сейчас у категории "
        f"**{top_row['Тип рака']}**: **{top_row['Риск, %']}%**."
    )

    if results_df["Категория_текст"].isin(["Умеренный", "Высокий"]).any():
        st.warning(
            "По одной или нескольким категориям риск получился умеренным или высоким. "
            "Стоит спокойно обсудить результат с врачом и при необходимости пройти профилактическую консультацию."
        )

    st.subheader("Подробно по каждому типу рака")
    for _, row in results_df.iterrows():
        st.markdown(f"### {row['Тип рака']}")
        st.progress(int(round(row["Риск, %"])))
        st.write(f"**Прогнозируемая оценка риска:** {row['Риск, %']}%")
        st.write(f"**Категория:** {row['Категория']}")
        st.caption(f"Модель: {row['Лучшая модель']}")
        st.info(f"**Как можно снизить риск:** {CANCER_RECOMMENDATIONS[row['Тип рака']]}")
        if row["Категория_текст"] in {"Умеренный", "Высокий"}:
            st.warning(
                "Советуем обратиться к врачу, чтобы обсудить результат и решить, нужны ли дополнительные обследования."
            )

    st.info(
        "Модели прогнозируют числовую оценку риска из датасета проекта. "
        "Категории «низкий / умеренный / высокий» добавлены в интерфейсе только для удобства чтения."
    )
    st.caption(AUTHOR_TEXT)


if __name__ == "__main__":
    main()