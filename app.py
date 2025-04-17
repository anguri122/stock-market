from flask import Flask, render_template, request, redirect, url_for, flash
from model.stock_predictor import predict_stock
import yfinance as yf
import pandas as pd
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'secret_key'

@app.route('/')
def cover():
    return render_template('cover.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == "chaitanya" and password == "12345":
            return redirect(url_for('input_stock'))
        else:
            flash("Invalid username or password", "danger")
            return render_template('login.html')
    return render_template('login.html')

@app.route('/input', methods=['GET', 'POST'])
def input_stock():
    stocks = {
        'AAPL': 'Apple Inc.',
        'GOOGL': 'Alphabet Inc.',
        'MSFT': 'Microsoft Corp.',
        'AMZN': 'Amazon.com, Inc.',
        'TSLA': 'Tesla, Inc.',
        'META': 'Meta Platforms',
        'NFLX': 'Netflix',
        'NVDA': 'NVIDIA Corp.',
        'AMD': 'Advanced Micro Devices',
        'IBM': 'IBM'
    }

    if request.method == 'POST':
        stock = request.form['stock']
        date = request.form['date']
        return redirect(url_for('dashboard', stock=stock, date=date))

    current_date = datetime.now().strftime('%Y-%m-%d')
    return render_template('input.html', current_date=current_date, stocks=stocks)

@app.route('/dashboard')
def dashboard():
    stock = request.args.get('stock')
    date = request.args.get('date')

    plot_urls, predicted_value, next_days, accuracy, mae, rmse, r2 = predict_stock(stock, date)

    ticker = yf.Ticker(stock)
    data = ticker.history(period='1y')

    closing_price = round(data['Close'].iloc[-1], 2)
    volume = "{:,}".format(int(data['Volume'].iloc[-1])) if not pd.isna(data['Volume'].iloc[-1]) else "N/A"
    market_cap = "{:,}".format(int(ticker.info.get("marketCap", 0))) if ticker.info.get("marketCap") else "N/A"
    high_52 = round(data['Close'].max(), 2)
    low_52 = round(data['Close'].min(), 2)

    return render_template('dashboard.html',
                           stock=stock,
                           date=date,
                           predicted_value=predicted_value,
                           plot1_url=plot_urls['plot1'],
                           plot2_url=plot_urls['plot2'],
                           next_days=next_days,
                           accuracy=accuracy,
                           mae=mae,
                           rmse=rmse,
                           r2=r2,
                           closing_price=closing_price,
                           volume=volume,
                           market_cap=market_cap,
                           high_52=high_52,
                           low_52=low_52)

if __name__ == '__main__':
    app.run(debug=True)
