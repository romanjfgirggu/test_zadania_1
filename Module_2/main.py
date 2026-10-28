import sqlite3
from pathlib import Path

import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import librosa

st.set_page_config(
    page_title="Sales Dashboard",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data():
    with sqlite3.connect("../Module_1/datasets.sqlite") as conn:
        df = pd.read_sql(
            sql="SELECT * FROM initial_table",
            con=conn
        )
    folder = Path("../dataset/Soundwel Dataset - Audio and Spectrograms/")
    colum = []
    for file in folder.iterdir():
        if file.suffix == ".wav":
            audio, sr = librosa.load(file.absolute(), sr=None, mono=False)
            duration = audio.shape[-1] / sr  # Длительность аудио
            colum.append({"duration": duration})
    return pd.concat([df, pd.DataFrame(colum)], axis=1)

def sample_count():
    original_sample = 0
    augmentation_sample = 0
    folder = Path("../dataset/Soundwel Dataset - Audio and Spectrograms/")
    for file in folder.iterdir():
        if file.suffix == ".wav":
            original_sample += 1
    folder = Path("../dataset/Augmentation/")
    for file in folder.iterdir():
        if file.suffix == ".wav":
            augmentation_sample += 1
    return [{"Original_Sample": original_sample, "Augmentation_sample": augmentation_sample}]

def distribution_records(table: pd.DataFrame, column: str, description: str):
        return px.bar(
        x=pd.DataFrame(table[column].value_counts()).T.columns,
        y=pd.DataFrame(table[column].value_counts()).T.iloc[0],
        title=description,
        text_auto=True,
        labels={
            "x": "Тип выборки",
            "y": "Количество объектов"
        }
    )

df = load_data()

st.title("📊 Sales Dashboard")

st.subheader("Данные")
st.dataframe(df, width="stretch")


valance_counts = (
    df["Valence"]
    .value_counts()
    .sort_index()
    .reset_index()
)
valance_counts.columns = ["Класс", "Количество"]
fig = px.bar(
    valance_counts,
    x="Класс",
    y="Количество",
    title="Баланс класса Valance",
    text_auto=True,
)
st.plotly_chart(fig, width="stretch")

sample_count = pd.DataFrame(sample_count())
fig2 = px.bar(
    x=sample_count.columns,
    y=sample_count.iloc[0].values,
    title="Баланс исходной и аугментированной выборок",
    text_auto=True,
    labels={
        "x": "Тип выборки",
        "y": "Количество объектов"
    }
)
st.plotly_chart(fig2, width="stretch")

distribution1, distribution2, distribution3, distribution4 = st.tabs([
    "Распределение записей по Context Category",
    "Распределение записей по Age Category",
    "Распределение записей по Sex",
    "Распределение записей по Call Type (Freq)"
])

with distribution1:
    st.plotly_chart(distribution_records(df, "Context Category", "Распределение записей по Context Category"), width="stretch")
with distribution2:
    st.plotly_chart(distribution_records(df, "Age Category", "Распределение записей по Age Category"), width="stretch")
with distribution3:
    st.plotly_chart(distribution_records(df, "Sex", "Распределение записей по Sex"), width="stretch")
with distribution4:
    st.plotly_chart(distribution_records(df, "Call Type (Freq)", "Распределение записей по Call Type (Freq)"), width="stretch")

duration1, duration2, duration3 = st.tabs([
    "Средняя длительность звуков по Valence",
    "Медианная длительность звуков по Valence",
    "Суммарная длительность звуков по Valence"
])

with duration1:
    df[df["Valence"] == "Pos"]["duration"].mean()
    df[df["Valence"] == "Pos"]["duration"].median()
    df[df["Valence"] == "Pos"]["duration"].sum()