import pandas as pd
import numpy as np
import streamlit as st
from catboost import CatBoostClassifier
from io import BytesIO

st.title("⚽ 축구 승무패 AI 예측기")

# 모델 로드 (캐시 사용)
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model("trained_model.cbm")
    return model

model = load_model()

def make_features(df):
    df = df.copy()
    df["total_rp"] = df["win_환급률"] + df["draw_환급률"] + df["loss_환급률"]
    df["win_nipr"] = (1 / df["win_odds"]) / df["total_rp"]
    df["draw_nipr"] = (1 / df["draw_odds"]) / df["total_rp"]
    df["loss_nipr"] = (1 / df["loss_odds"]) / df["total_rp"]
    df["win_rpi"] = df["win_odds"] * (df["win_환급률"] / df["total_rp"])
    df["draw_rpi"] = df["draw_odds"] * (df["draw_환급률"] / df["total_rp"])
    df["loss_rpi"] = df["loss_odds"] * (df["loss_환급률"] / df["total_rp"])
    df["win_div"] = (1 / df["win_odds"]) - (df["win_환급률"] / df["total_rp"])
    df["draw_div"] = (1 / df["draw_odds"]) - (df["draw_환급률"] / df["total_rp"])
    df["loss_div"] = (1 / df["loss_odds"]) - (df["loss_환급률"] / df["total_rp"])
    df["rp_ratio"] = df[["win_환급률", "draw_환급률", "loss_환급률"]].max(axis=1) / \
                     df[["win_환급률", "draw_환급률", "loss_환급률"]].min(axis=1)
    return df

uploaded_file = st.file_uploader("📂 before.csv 업로드", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = make_features(df)

    feature_cols = [
        'total_rp', 'win_nipr', 'draw_nipr', 'loss_nipr',
        'win_rpi', 'draw_rpi', 'loss_rpi',
        'win_div', 'draw_div', 'loss_div',
        'rp_ratio', 'hand', 'league'
    ]

    X = df[feature_cols]
    proba = model.predict_proba(X)
    class_labels = model.classes_

    df["예측결과"] = [class_labels[i] for i in np.argmax(proba, axis=1)]
    df["승확률"] = proba[:, list(class_labels).index("승")]
    df["무확률"] = proba[:, list(class_labels).index("무")]
    df["패확률"] = proba[:, list(class_labels).index("패")]

    result = df[["name", "type", "예측결과", "승확률", "무확률", "패확률"]]
    st.dataframe(result)

    # 다운로드
    buffer = BytesIO()
    result.to_csv(buffer, index=False, encoding="utf-8-sig")
    st.download_button("📥 예측결과 다운로드", buffer.getvalue(), file_name="prediction_result.csv", mime="text/csv")
