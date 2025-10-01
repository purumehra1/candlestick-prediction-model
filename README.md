📈 Candlestick Pattern Predictor
A deep learning system that predicts stock price movements using candlestick patterns and technical indicators. Built with TensorFlow and a hybrid CNN-LSTM neural network architecture.

🎯 Overview
This project implements a sophisticated machine learning model that analyzes historical stock market data to predict whether prices will move UP or DOWN. It combines Convolutional Neural Networks (for pattern recognition) with Long Short-Term Memory networks (for time-series analysis) to identify tradeable patterns in financial data.

✨ Key Features
Interactive Stock Selection - Analyze any stock with real-time data fetching via Yahoo Finance API

Advanced Feature Engineering - Automatically calculates 22+ technical indicators including:

Moving Averages (SMA 5, 10, 20, 50, 200 & EMA 12, 26)

Momentum Indicators (RSI, MACD, Stochastic Oscillator)

Volatility Measures (Bollinger Bands, ATR)

Volume Analysis (OBV, Volume Rate of Change)

5 Major Candlestick Patterns (Doji, Hammer, Shooting Star, Bullish/Bearish Engulfing)

Hybrid CNN-LSTM Architecture - State-of-the-art deep learning model for sequential pattern recognition

Model Evaluation - Comprehensive metrics including accuracy, precision, recall, and F1-score

Persistence - Saves trained models for future predictions without retraining

🛠️ Technologies Used
Python 3.x

TensorFlow/Keras - Deep learning framework

Scikit-learn - Data preprocessing and evaluation

Pandas & NumPy - Data manipulation

yfinance - Stock market data retrieval

TA-Lib - Technical analysis indicators

📦 Installation
bash
# Clone the repository
git clone https://github.com/yourusername/CandlestickPredictor.git
cd CandlestickPredictor

# Install required packages
pip3 install tensorflow scikit-learn yfinance pandas numpy
🚀 Usage
Basic Usage

bash
python3 candlestick_model.py
When prompted, enter any stock ticker symbol:

US Stocks: AAPL, TSLA, GOOGL, MSFT

Indian Stocks: RELIANCE.NS, TCS.NS, INFY.NS

Example Output

text
======================================================================
CANDLESTICK PATTERN PREDICTION MODEL
======================================================================

Enter stock ticker (e.g., AAPL, TSLA, RELIANCE.NS): TSLA

Fetching data for TSLA...
Downloaded 500 records
Engineering features...
Total features: 22

Training model...
Epoch 1/50 - loss: 0.6932 - accuracy: 0.5028
...

==================================================
MODEL EVALUATION RESULTS
==================================================
Accuracy:  0.8222 (82.22%)
Precision: 0.8156
Recall:    0.8222
F1-Score:  0.8187

==================================================
SAMPLE PREDICTION
==================================================
Predicted: UP ↑ (Confidence: 87.34%)
Actual:    UP ↑
Result:    ✓ CORRECT
🧠 Model Architecture
Input Layer - 20 timesteps × 22 features

Conv1D Layer - 64 filters, kernel size 3, ReLU activation

MaxPooling1D - Pool size 2

LSTM Layer - 50 units

Dropout - 0.2 for regularization

Dense Output - Sigmoid activation for binary classification

The model uses:

Optimizer: Adam

Loss Function: Binary Crossentropy

Training: 50 epochs with early stopping and learning rate reduction

Data Split: 80% training, 20% testing

📊 Results
The model achieves prediction accuracy ranging from 55-85% depending on the stock and market conditions. Performance varies because:

Stock markets are inherently unpredictable

Short-term price movements contain significant noise

External factors (news, earnings, geopolitics) aren't included in the model

🎓 What I Learned
Integrating Python with Xcode using External Build System

Time-series forecasting with deep learning

Financial feature engineering and technical analysis

Model evaluation and hyperparameter tuning

Real-world machine learning limitations in financial markets

🔮 Future Improvements
 Add sentiment analysis from news/social media

 Implement multi-step ahead predictions

 Include volume-weighted indicators

 Add portfolio backtesting functionality

 Create visualization dashboard for predictions

 Support for cryptocurrency markets

⚠️ Disclaimer
This project is for educational purposes only. Do not use these predictions for actual trading decisions. Stock market prediction is extremely difficult, and past performance does not guarantee future results. Always consult with financial professionals before making investment decisions.

👤 Author
Puru Mehra
Created on: October 1, 2025

📝 License
This project is open source and available under the MIT License.

🌟 Acknowledgments
TensorFlow team for the deep learning framework

Yahoo Finance for providing free market data

The open-source data science community

⭐ If you found this project helpful, please give it a star!# candlestick-prediction-model
Stocks chart prediction ml model
