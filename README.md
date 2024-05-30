# Sipaling 📈 

## Overview
Sipaling is a web application that predicts stock prices using an LSTM deep learning model. It allows users to select a stock, specify a date range for historical data, and predict future prices for a specified number of days.

## Features
- Select stock codes and company names from a predefined list.
- Fetch historical stock data from Yahoo Finance.
- Predict stock prices using a pre-trained LSTM model.
- Visualize actual and predicted stock prices.
- Customizable forecast duration (1-100 days).

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/ramadhanep/sipaling.git
    cd sipaling
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Download or place the pre-trained LSTM model (`stock_dl_model.h5`) in the project directory.

4. Ensure the `stock_codes.csv` file with stock codes and company names is present in the project directory.

## Usage

1. Run the Streamlit app:
    ```sh
    streamlit run app.py
    ```

2. Open your web browser and go to `http://localhost:8501`.

3. Use the sidebar to:
    - Select a stock code and company name.
    - Choose the start and end dates for historical data.
    - Specify the number of days to forecast.

4. Click the "Prediksi" button to view the predictions.

## File Structure
- `app.py`: The main application code.
- `stock_dl_model.h5`: Pre-trained LSTM model for stock price prediction.
- `stock_codes.csv`: CSV file containing stock codes and company names.
- `requirements.txt`: List of Python dependencies.

## Dependencies
- pandas
- numpy
- datetime
- plotly
- yfinance
- tensorflow
- scikit-learn
- matplotlib
- streamlit

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- [Yahoo Finance](https://finance.yahoo.com) for providing historical stock data.
- [Streamlit](https://streamlit.io) for providing an easy-to-use web application framework.
- [TensorFlow](https://www.tensorflow.org) for the deep learning framework used to build the LSTM model.

## Contributing
Contributions are welcome! Please create a pull request or submit an issue if you have any suggestions or improvements.