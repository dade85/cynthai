# -*- coding: utf-8 -*-
"""
Streamlit version of Financial Forecast Interface (Extended Features)
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import openai
import streamlit as st
from io import BytesIO

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# -------------------------------
# 1. Data Upload or Synthetic Generation
# -------------------------------
def load_data(uploaded_file=None):
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df.columns = [col.lower() for col in df.columns]
        if 'ds' not in df.columns or 'y' not in df.columns:
            st.error("CSV must contain 'ds' and 'y' columns.")
            return None
        df['ds'] = pd.to_datetime(df['ds'])
    else:
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq='D')
        trend = np.linspace(0, 10, len(dates))
        seasonality = 5 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
        noise = np.random.normal(0, 1, len(dates))
        prices = 100 + trend + seasonality + noise
        df = pd.DataFrame({"ds": dates, "y": prices})
    return df

# -------------------------------
# 2. Forecasting
# -------------------------------
def get_forecast(df: pd.DataFrame, horizon: int):
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=horizon)
    forecast = model.predict(future)
    return forecast, model

def plot_forecast(model, forecast_df):
    fig = model.plot(forecast_df)
    st.pyplot(fig)
    return fig

def plot_components(model, forecast_df):
    fig = model.plot_components(forecast_df)
    st.pyplot(fig)

# -------------------------------
# 3. Intelligence Layer
# -------------------------------
def call_llm(prompt: str) -> str:
    try:
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a financial, forecast assistant and a member of the CynthAI agent family. Your job is to predict & forecast accurately based on inputted time series data"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM call failed: {str(e)}"

def get_forecast_accuracy(model):
    try:
        cv_results = cross_validation(model, initial='365 days', period='180 days', horizon='30 days')
        perf = performance_metrics(cv_results)
        return (f"RMSE: {perf['rmse'].mean():.2f}, MAE: {perf['mae'].mean():.2f}, MAPE: {perf['mape'].mean() * 100:.2f}%")
    except Exception as e:
        return f"Error: {str(e)}"

def forecast_summary(df: pd.DataFrame, forecast_df: pd.DataFrame) -> str:
    forecast_period = forecast_df[forecast_df['ds'] > df['ds'].max()]
    if forecast_period.empty:
        return "No forecast available."
    start = forecast_period.iloc[0]
    end = forecast_period.iloc[-1]
    avg = forecast_period['yhat'].mean()
    trend = "upward" if end['yhat'] > start['yhat'] else "downward" if end['yhat'] < start['yhat'] else "flat"
    return (
        f"Forecast period: {forecast_period['ds'].min().date()} to {forecast_period['ds'].max().date()}"
        f"Start: {start['yhat']:.2f}, End: {end['yhat']:.2f}, Avg: {avg:.2f} ({trend} trend)"
    )

def handle_query(query: str, df: pd.DataFrame, forecast_df: pd.DataFrame, model):
    query = query.lower().strip()

    if "summary" in query:
        return forecast_summary(df, forecast_df)
    elif "accuracy" in query:
        return get_forecast_accuracy(model)
    elif "trend" in query:
        future = forecast_df[forecast_df['ds'] > df['ds'].max()]
        if future.empty:
            return "No forecasted period."
        start, end = future.iloc[0]['yhat'], future.iloc[-1]['yhat']
        if end > start:
            return "Upward trend."
        elif end < start:
            return "Downward trend."
        return "Flat trend."
    elif "average" in query:
        future = forecast_df[forecast_df['ds'] > df['ds'].max()]
        return f"Average: {future['yhat'].mean():.2f}"
    elif "components" in query:
        plot_components(model, forecast_df)
        return "Components plotted."
    elif "value on" in query:
        try:
            date_str = query.split("value on")[-1].strip().split()[0]
            date = pd.to_datetime(date_str)
            row = forecast_df[forecast_df['ds'] == date]
            if row.empty:
                return f"No forecast for {date_str}."
            return f"{date_str}: {row['yhat'].values[0]:.2f} (±{row['yhat_upper'].values[0] - row['yhat_lower'].values[0]:.2f})"
        except:
            return "Invalid date format. Use YYYY-MM-DD."
    else:
        prompt = forecast_summary(df, forecast_df) + "User query: " + query
        return call_llm(prompt)

# -------------------------------
# 4. Streamlit UI
# -------------------------------
st.set_page_config(page_title="CynthAI© Financial Forecast Agent", layout="wide")
st.title("📈 CynthAI© Financial Forecast Agent Interface")

uploaded_file = st.file_uploader("Upload your CSV (columns: ds, y) or use default synthetic data:", type=["csv"])
df = load_data(uploaded_file)

if df is not None:
    horizon = st.slider("Forecast Horizon (days)", min_value=7, max_value=120, value=30)
    forecast, model = get_forecast(df, horizon)

    st.subheader("Forecast Plot")
    fig = plot_forecast(model, forecast)

    st.download_button("Download Forecast CSV", data=forecast.to_csv(index=False), file_name="forecast.csv", mime="text/csv")

    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.download_button("Download Forecast Chart (PNG)", data=buf.getvalue(), file_name="forecast.png", mime="image/png")

    query = st.text_input("Ask a question about the forecast:")
    if query:
        st.markdown("**Response:**")
        st.write(handle_query(query, df, forecast, model))
