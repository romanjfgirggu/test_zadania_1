import sqlite3
from pathlib import Path

import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import librosa
from pandas.core.interchange.dataframe_protocol import DataFrame
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Dashboard",
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

def mms(df: pd.DataFrame, valence = "Pos"):
    return {"mean": df[df["Valence"] == valence]["duration"].mean(),
            "meadian": df[df["Valence"] == valence]["duration"].median(),
            "sum": df[df["Valence"] == valence]["duration"].sum()}

df = load_data()

st.title("Dashboard")

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

pos = mms(df, "Pos")
neg = mms(df, "Neg")

with duration1:
    st.plotly_chart(px.bar(
        x=["Pos", "Neg"],
        y=[pos["mean"], neg["mean"]],
        title="Средняя длительность звуков по Valence",
        text_auto=True,
        labels={
            "x": "Классы",
            "y": "Длительность"
        }
    ))
with duration2:
    st.plotly_chart(px.bar(
        x=["Pos", "Neg"],
        y=[pos["meadian"], neg["meadian"]],
        title="Средняя длительность звуков по Valence",
        text_auto=True,
        labels={
            "x": "Классы",
            "y": "Длительность"
        }
    ))
with duration3:
    st.plotly_chart(px.bar(
        x=["Pos", "Neg"],
        y=[pos["sum"], neg["sum"]],
        title="Средняя длительность звуков по Valence",
        text_auto=True,
        labels={
            "x": "Классы",
            "y": "Длительность"
        }
    ))






if "audio_index" not in st.session_state:
    st.session_state.audio_index = 0

# Защита при изменении количества записей
st.session_state.audio_index = min(
    st.session_state.audio_index,
    len(df) - 1
)

# Навигация
prev_col, counter_col, next_col = st.columns([1, 2, 1])

with prev_col:
    if st.button(
        "⬅️ Назад",
        disabled=st.session_state.audio_index == 0,
        use_container_width=True
    ):
        st.session_state.audio_index -= 1

with next_col:
    if st.button(
        "Вперёд ➡️",
        disabled=st.session_state.audio_index == len(df) - 1,
        use_container_width=True
    ):
        st.session_state.audio_index += 1

index = st.session_state.audio_index
record = df.iloc[index]

with counter_col:
    st.markdown(f"**Запись {index + 1} из {len(df)}**")

filename = record["Audio Filename"]
audio_path = Path("../dataset/Soundwel Dataset - Audio and Spectrograms/" + filename)


with st.container(border=True):
    st.subheader("Аудиозапись")
    st.caption(filename)

    if audio_path.is_file():
        st.audio(str(audio_path), format="audio/wav")
    else:
        st.warning(f"Аудиофайл не найден: {audio_path}")

    st.divider()

    st.subheader("Описание")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"**Возрастная категория:** {record['Age Category']}"
        )
        st.markdown(
            f"**Valence:** {record['Valence']}"
        )

    with col2:
        st.metric(
            "Длительность",
            f"{record['duration']:.2f} сек."
        )

y, sr = librosa.load(audio_path, sr=16000)

# STFT
S = librosa.stft(
    y,
    n_fft=1024,
    hop_length=256
)

# Амплитуда -> dB
S_db = librosa.amplitude_to_db(
    np.abs(S),
    ref=np.max
)

# Создаём график нужного размера
fig, ax = plt.subplots(figsize=(8, 3))

# Визуализация
img = librosa.display.specshow(
    S_db,
    sr=sr,
    hop_length=256,
    x_axis="time",
    y_axis="linear",
    cmap="magma",
    ax=ax
)

fig.colorbar(img, ax=ax, format="%+2.0f dB")

# Вывод в Streamlit
st.pyplot(fig, width="content")

if st.button("Сохранить в png"):
    fig.savefig(f"spec_save/spectrogram_{st.session_state.audio_index}.png")
    st.success(f"Файл успешно сохранён по пути spec_save/spectrogram_{st.session_state.audio_index}.png!")
plt.close(fig)