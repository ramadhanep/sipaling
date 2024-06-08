import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import plotly.graph_objs as go
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

# Fetch stock data from Yahoo Finance
def fetch_stock_data(stock_code, start_date, end_date):
    data = yf.download(stock_code, start=start_date, end=end_date)
    return data

# Fetch news from Yahoo Finance
def fetch_news(stock_code, num_news):
    news_list = yf.Ticker(stock_code).news
    news_data = []
    for news_item in news_list[:num_news]:
        publisher = news_item.get('publisher', '')
        title = news_item.get('title', '')
        image = news_item.get('thumbnail', {}).get('resolutions', [{}])[0].get('url', '')
        link = news_item.get('link', '')
        news_data.append({'publisher': publisher, 'title': title, 'image': image, 'link': link})
    return pd.DataFrame(news_data)

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
    # Load stock codes and company names from CSV file
    stock_codes_df = pd.read_csv('stock_codes.csv')

    st.set_page_config(page_title="Sipaling", page_icon="📈", layout="wide")
    
    # Apply custom CSS for centering and max-width
    st.markdown("""
        <style>
        .main {
            max-width: 600px;
            margin: 0 auto;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title('Sipaling 📈')
    st.write("")
    
    # Menu for selecting the mode of the app
    mode = option_menu(None, ["Home", "News", "Info"], 
    icons=['house', 'newspaper', 'info-circle'], 
    menu_icon="cast", default_index=0, orientation="horizontal")
    
    if mode == "Home":
        stock_options = stock_codes_df.apply(lambda row: f"{row['Nama']} ~ {row['Kode']}", axis=1).tolist()
        
        # Get user inputs
        selected_option = st.selectbox("Pilih Asset", stock_options)
        selected_stock_code = selected_option.split(' ~ ')[1]
        selected_stock_name = selected_option.split(' ~ ')[0]
        start_date = st.date_input("Start Date", dt.date(2023, 1, 1))
        end_date = st.date_input("End Date", dt.date.today())
        forecast_days = st.slider("Prediksi untuk berapa hari ke depan", min_value=1, max_value=100, value=100)
        
        model = load_model('stock_dl_model.h5')
        
        # Button to trigger prediction
        if st.button("Prediksi"):
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
                fig.add_trace(go.Candlestick(x=stock_data.index, 
                                             open=stock_data['Open'], 
                                             high=stock_data['High'], 
                                             low=stock_data['Low'], 
                                             close=stock_data['Close'], 
                                             name='Harga Saham'))
                fig.add_trace(go.Scatter(x=stock_data.index[100:], 
                                         y=predictions.flatten(), 
                                         mode='lines', 
                                         name='Prediksi Harga', 
                                         line=dict(color='#FF4B4B')))
                
                # Forecast for the next N days
                forecast_dates = pd.date_range(start=end_date + dt.timedelta(days=1), periods=forecast_days)
                last_sequence = sequences[-1]
                forecast = []
                for _ in range(forecast_days):
                    forecast.append(model.predict(last_sequence.reshape(1, -1, 1))[0, 0])
                    last_sequence = np.roll(last_sequence, -1)
                    last_sequence[-1] = forecast[-1]
                
                forecast = inverse_transform(np.array(forecast).reshape(-1, 1), scaler)
                fig.add_trace(go.Scatter(x=forecast_dates, 
                                         y=forecast.flatten(), 
                                         mode='lines', 
                                         name=f'Prediksi Harga', 
                                         line=dict(color='#FF4B4B')))
                
                st.write("")
                st.write(f'**Prediksi harga {selected_stock_name} ({selected_stock_code}) selama {forecast_days} hari ke depan**')
                
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig)

    elif mode == "News":
        stock_options = stock_codes_df.apply(lambda row: f"{row['Nama']} ~ {row['Kode']}", axis=1).tolist()
        
        # Get user inputs
        selected_option = st.selectbox("Pilih Asset", stock_options)
        selected_stock_code = selected_option.split(' ~ ')[1]
        selected_stock_name = selected_option.split(' ~ ')[0]
        news = fetch_news(selected_stock_code, 10)
        
        if not news.empty:
            st.write(f"### Berita Terbaru tentang {selected_stock_name} ({selected_stock_code})")
            for index, row in news.iterrows():
                st.write("---")
                st.write(f"**{row['publisher']}**")
                if row['image']:  # Check if the image URL is not empty
                    st.image(row['image'], width=200)
                st.write(f"#### {row['title']}")
                st.write(f"[Baca sekarang]({row['link']})")
        else:
            st.write("Tidak ada berita terbaru untuk aset ini.")

    elif mode == "Info":
        st.write("Sipaling adalah aplikasi prediksi harga aset finansial yang inovatif dan canggih, mencakup baik saham maupun cryptocurrency. Dengan menggunakan teknologi machine learning dan analisis data terkini, Sipaling memberikan kamu kemampuan untuk meramalkan pergerakan harga aset dengan akurasi tinggi. Aplikasi ini dirancang untuk para investor, trader, dan semua orang yang tertarik dalam dunia keuangan.")
        st.write("Dengan Sipaling, kamu dapat memilih berbagai aset saham dari berbagai perusahaan terkemuka, serta cryptocurrency populer seperti Bitcoin, Ethereum, dan lainnya. Aplikasi ini memberikan kamu fleksibilitas untuk memilih rentang waktu tertentu, mulai dari beberapa hari hingga beberapa bulan, untuk melakukan prediksi harga aset.")
        st.write("Fitur unggulan dari Sipaling adalah kemampuannya untuk menghasilkan prediksi harga aset untuk periode yang akan datang. Dengan menggunakan teknik deep learning dan analisis data historis, Sipaling memberikan kamu gambaran yang jelas tentang kemungkinan pergerakan harga aset ke depan.")
        st.write("Tidak hanya itu, Sipaling juga dilengkapi dengan antarmuka yang ramah pengguna dan visualisasi data interaktif, sehingga kamu dapat dengan mudah memahami dan menganalisis tren pasar saham dan cryptocurrency.")
        st.write("---")
        st.write("Thanks to Ramadhan 🙏🏻")
        st.write("its free for everyone")

# Run the app
if __name__ == '__main__':
    main()
