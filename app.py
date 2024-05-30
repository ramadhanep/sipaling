import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import plotly.graph_objs as go
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Fetch stock data from Yahoo Finance
def fetch_stock_data(stock_code, start_date, end_date):
    data = yf.download(stock_code, start=start_date, end=end_date)
    return data

# Preprocess data for LSTM model
def preprocess_data(data, sequence_length=100):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    
    # Create sequences of data
    sequences = []
    for i in range(len(scaled_data) - sequence_length):
        sequences.append(scaled_data[i:i+sequence_length])
    
    sequences = np.array(sequences)
    
    return sequences, scaler

# Inverse transform predictions
def inverse_transform(predictions, scaler):
    return scaler.inverse_transform(predictions)

# Streamlit App
def main():
    st.set_page_config(page_title="Sipaling", page_icon="📈", layout="wide")
    st.title('Prediksi saham kamu yuk! 🚀')
    
    # Sidebar for user input
    st.sidebar.title('📈 Sipaling')
    st.sidebar.header('Input sahamnya dulu')
    
    # Load stock codes and company names from CSV file
    stock_codes_df = pd.read_csv('stock_codes.csv')
    stock_options = stock_codes_df.apply(lambda row: f"{row['Kode Saham']} - {row['Nama Perusahaan']}", axis=1).tolist()
    
    # Get user inputs
    selected_option = st.sidebar.selectbox("Pilih kode saham - nama perusahaan:", stock_options)
    selected_stock_code = selected_option.split(' - ')[0]
    selected_stock_name = selected_option.split(' - ')[1]
    start_date = st.sidebar.date_input("Start Date:", dt.date(2020, 1, 1))
    end_date = st.sidebar.date_input("End Date:", dt.date.today())
    forecast_days = st.sidebar.slider("Prediksi untuk berapa hari ke depan:", min_value=1, max_value=100, value=120)
    
    model = load_model('stock_dl_model.h5')
    
    # Button to trigger prediction
    if st.sidebar.button("Prediksi"):
        if model is not None:
            # Fetch stock data from Yahoo Finance
            stock_data = fetch_stock_data(selected_stock_code, start_date, end_date)
            
            # Preprocess data
            sequences, scaler = preprocess_data(stock_data['Close'])
            
            # Predict stock prices using the LSTM model
            predictions = model.predict(sequences)
            
            # Inverse transform predictions
            predictions = inverse_transform(predictions, scaler)
            
            # Plot actual prices and predictions using Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Harga Saham'))
            fig.add_trace(go.Scatter(x=stock_data.index[100:], y=predictions.flatten(), mode='lines', name='Prediksi Harga Historis'))
            
            # Forecast for the next N days
            forecast_dates = pd.date_range(start=end_date + dt.timedelta(days=1), periods=forecast_days)
            last_sequence = sequences[-1]
            forecast = []
            for _ in range(forecast_days):
                forecast.append(model.predict(last_sequence.reshape(1, -1, 1))[0, 0])
                last_sequence = np.roll(last_sequence, -1)
                last_sequence[-1] = forecast[-1]
            
            forecast = inverse_transform(np.array(forecast).reshape(-1, 1), scaler)
            fig.add_trace(go.Scatter(x=forecast_dates, y=forecast.flatten(), mode='lines', name=f'Prediksi Harga ({forecast_days} hari)'))
            
            fig.update_layout(title=f'Prediksi harga saham {selected_stock_code} - {selected_stock_name} selama {forecast_days} hari ke depan',
                              xaxis_title='Tanggal',
                              yaxis_title='Harga')
            st.plotly_chart(fig)

# Run the app
if __name__ == '__main__':
    main()
