import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

def predict_stock(stock_symbol, target_date):
    data = yf.download(stock_symbol, period='1y')
    data = data[['Close']].dropna()

    data['Target'] = data['Close'].shift(-1)
    data.dropna(inplace=True)

    X = np.array(data['Close']).reshape(-1, 1)
    y = np.array(data['Target'])

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    model = SVR(kernel='rbf', C=100, gamma=0.1)
    model.fit(X_scaled, y)

   
    today = datetime.today().date()
    target = datetime.strptime(target_date, '%Y-%m-%d').date()
    days_ahead = (target - today).days
    days_ahead = max(1, min(days_ahead, 30))  

    current_price = data['Close'].iloc[-1]
    future_dates = []
    predicted_prices = []

    for i in range(days_ahead):
        input_scaled = scaler.transform(np.array(current_price).reshape(1, -1))
        predicted_price = model.predict(input_scaled)[0]
        predicted_prices.append(predicted_price)
        future_dates.append(today + timedelta(days=i + 1))
        current_price = predicted_price

    target_prediction = predicted_prices[-1]

  
    test_X = X_scaled[-10:]
    test_y = y[-10:]
    pred_y = model.predict(test_X)
    mae = round(mean_absolute_error(test_y, pred_y), 2)
    rmse = round(np.sqrt(mean_squared_error(test_y, pred_y)), 2)
    r2 = round(r2_score(test_y, pred_y), 2)

    next_days = []
    for i, price in enumerate(predicted_prices):
        next_days.append({
            'date': future_dates[i].strftime('%Y-%m-%d'),
            'prediction': round(price, 2)
        })

 
    plt.figure(figsize=(10, 5))
    plt.plot(data.index[-60:], data['Close'][-60:], label='Historical Prices', color='blue')
    plt.plot(future_dates, predicted_prices, 'o--', label='Predicted Prices', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.title(f'{stock_symbol} Stock Price Forecast')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    buf1 = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf1, format='png')
    buf1.seek(0)
    plot1 = base64.b64encode(buf1.read()).decode('utf-8')
    plt.close()

  
    plt.figure(figsize=(10, 4))
    plt.plot(future_dates, predicted_prices, 'ro-', label='Forecast')
    plt.title('Zoomed Forecast')
    plt.xlabel('Date')
    plt.ylabel('Predicted Price ($)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend()
    buf2 = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf2, format='png')
    buf2.seek(0)
    plot2 = base64.b64encode(buf2.read()).decode('utf-8')
    plt.close()

    combined_plots = {
        'plot1': plot1,
        'plot2': plot2
    }

    return combined_plots, round(target_prediction, 2), next_days, round(100 - mean_absolute_error(test_y, pred_y) / np.mean(test_y) * 100, 2), mae, rmse, r2
