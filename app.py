import streamlit as st
import pandas as pd
import os
from agent import get_agent

st.set_page_config(page_title="LLM Data Agent", layout="wide")
st.title("🤖 AI Аналитик Данных (Агентный подход)")

# Боковая панель для настроек и загрузки
with st.sidebar:
    st.header("Настройки")
    api_key = st.text_input("OpenAI API Key:", type="password")
    st.markdown("---")
    st.header("Загрузка данных")
    uploaded_file = st.file_uploader("Загрузите CSV или Excel", type=['csv', 'xlsx'])

    st.markdown("---")
    st.header("Контекст задачи")
    user_context = st.text_area("Опишите, на что обратить внимание:",
                                placeholder="Например: Проанализируй продажи, найди аномалии и построй график тренда.")

# Основная часть
if uploaded_file and api_key:
    # Читаем данные
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📊 Превью данных")
    st.dataframe(df.head())

    if st.button("🚀 Запустить анализ"):
        if not user_context:
            st.warning("Пожалуйста, укажите контекст задачи в боковой панели.")
        else:
            with st.spinner("Агент анализирует данные и пишет код..."):
                try:
                    # Создаем папку для графиков
                    os.makedirs('./plots', exist_ok=True)

                    agent = get_agent(df, api_key, user_context)

                    # Запуск агента
                    response = agent.invoke({"input": user_context})

                    st.subheader("📝 Отчет агента")
                    st.markdown(response['output'])

                    # Отображаем сохраненные графики
                    plots = [f for f in os.listdir('./plots') if f.endswith('.png')]
                    if plots:
                        st.subheader("📈 Построенные графики")
                        for plot in plots:
                            st.image(f'./plots/{plot}')

                except Exception as e:
                    st.error(f"Ошибка выполнения: {str(e)}")
else:
    st.info("Загрузите файл и введите API Key в боковой панели, чтобы начать.")