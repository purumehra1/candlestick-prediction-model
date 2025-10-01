"""
Candlestick Pattern Prediction Model - Complete Implementation
Run in Xcode: File > New > Project > Other > External Build System
Set Build Tool to: /usr/bin/python3
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Data Collection
def fetch_stock_data(symbol='AAPL', period='2y'):
    """Fetch historical stock data using yfinance"""
    try:
        import yfinance as yf
        print(f"Fetching data for {symbol}...")
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        df.reset_index(inplace=True)
        print(f"Downloaded {len(df)} records")
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        # Generate synthetic data for testing
        return generate_synthetic_data()

def generate_synthetic_data(n_samples=500):
    """Generate synthetic OHLC data for testing"""
    dates = pd.date_range(end=datetime.now(), periods=n_samples, freq='D')
    base_price = 100
    data = []
    
    for i in range(n_samples):
        open_price = base_price + np.random.randn() * 2
        close_price = open_price + np.random.randn() * 3
        high_price = max(open_price, close_price) + abs(np.random.randn())
        low_price = min(open_price, close_price) - abs(np.random.randn())
        volume = np.random.randint(1000000, 10000000)
        
        data.append({
            'Date': dates[i],
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': close_price,
            'Volume': volume
        })
        base_price = close_price
    
    return pd.DataFrame(data)

# Feature Engineering
def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    # Moving Averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    
    return df

def detect_candlestick_patterns(df):
    """Detect key candlestick patterns"""
    # Candle body and shadows
    df['body'] = abs(df['Close'] - df['Open'])
    df['upper_shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['lower_shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['range'] = df['High'] - df['Low']
    
    # Doji: small body relative to range
    df['Doji'] = (df['body'] <= df['range'] * 0.1).astype(int)
    
    # Hammer: small body, long lower shadow
    df['Hammer'] = ((df['lower_shadow'] >= df['body'] * 2) &
                    (df['upper_shadow'] <= df['body'] * 0.3)).astype(int)
    
    # Shooting Star: small body, long upper shadow
    df['Shooting_Star'] = ((df['upper_shadow'] >= df['body'] * 2) &
                           (df['lower_shadow'] <= df['body'] * 0.3)).astype(int)
    
    # Engulfing patterns
    df['Bullish_Engulfing'] = 0
    df['Bearish_Engulfing'] = 0
    
    for i in range(1, len(df)):
        prev_body = abs(df.iloc[i-1]['Close'] - df.iloc[i-1]['Open'])
        curr_body = abs(df.iloc[i]['Close'] - df.iloc[i]['Open'])
        
        if (df.iloc[i-1]['Close'] < df.iloc[i-1]['Open'] and
            df.iloc[i]['Close'] > df.iloc[i]['Open'] and
            curr_body > prev_body):
            df.at[i, 'Bullish_Engulfing'] = 1
            
        if (df.iloc[i-1]['Close'] > df.iloc[i-1]['Open'] and
            df.iloc[i]['Close'] < df.iloc[i]['Open'] and
            curr_body > prev_body):
            df.at[i, 'Bearish_Engulfing'] = 1
    
    return df

def prepare_features(df):
    """Prepare feature matrix"""
    df = calculate_technical_indicators(df)
    df = detect_candlestick_patterns(df)
    
    # Create target variable (1 if price goes up next day, 0 otherwise)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
                    'SMA_10', 'SMA_30', 'EMA_10', 'RSI', 'MACD', 'Signal',
                    'BB_middle', 'BB_upper', 'BB_lower', 'body', 'upper_shadow',
                    'lower_shadow', 'Doji', 'Hammer', 'Shooting_Star',
                    'Bullish_Engulfing', 'Bearish_Engulfing']
    
    return df, feature_cols

# Data Preprocessing
def create_sequences(data, target, sequence_length=20):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(target[i+sequence_length])
    return np.array(X), np.array(y)

def normalize_data(train_data, test_data):
    """Normalize features using Min-Max scaling"""
    train_min = train_data.min(axis=0)
    train_max = train_data.max(axis=0)
    
    train_normalized = (train_data - train_min) / (train_max - train_min + 1e-8)
    test_normalized = (test_data - train_min) / (train_max - train_min + 1e-8)
    
    return train_normalized, test_normalized

# Model Architecture
def build_cnn_lstm_model(input_shape, num_classes=2):
    """Build hybrid CNN-LSTM model using TensorFlow/Keras"""
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
        
        model = keras.Sequential([
            # CNN layers for feature extraction
            layers.Conv1D(filters=64, kernel_size=3, activation='relu',
                         input_shape=input_shape),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            
            # LSTM layers for temporal patterns
            layers.LSTM(100, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(50),
            layers.Dropout(0.3),
            
            # Dense layers
            layers.Dense(50, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    except ImportError:
        print("TensorFlow not installed. Please install: pip install tensorflow")
        return None

# Training and Evaluation
def train_model(X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    """Train the model"""
    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    model = build_cnn_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    if model is None:
        return None
    
    print("\nModel Architecture:")
    model.summary()
    
    try:
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=10,
                                   restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                      patience=5, min_lr=1e-7)
        
        print("\nTraining model...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        return model, history
    except Exception as e:
        print(f"Training error: {e}")
        return None, None

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.metrics import confusion_matrix, classification_report
        
        # Predictions
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred,
                                    target_names=['Down', 'Up']))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred
        }
    except ImportError:
        print("scikit-learn not installed. Please install: pip install scikit-learn")
        return None

# Main Execution
def main():
    print("="*70)
    print("CANDLESTICK PATTERN PREDICTION MODEL")
    print("="*70)
    
    # Ask user for stock ticker
    stock_symbol = input("\nEnter stock ticker (e.g., AAPL, TSLA, RELIANCE.NS): ").upper()
    
    # 1. Data Collection
    print(f"\nFetching data for {stock_symbol}...")
    df = fetch_stock_data(stock_symbol, period='5y')

    print(f"\nDataset shape: {df.shape}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # 2. Feature Engineering
    print("\nEngineering features...")
    df, feature_cols = prepare_features(df)
    print(f"Total features: {len(feature_cols)}")
    
    # 3. Prepare sequences
    sequence_length = 20
    X_data = df[feature_cols].values
    y_data = df['Target'].values
    
    # Split data (80-20)
    split_idx = int(len(X_data) * 0.8)
    X_train_raw = X_data[:split_idx]
    X_test_raw = X_data[split_idx:]
    y_train_raw = y_data[:split_idx]
    y_test_raw = y_data[split_idx:]
    
    # Normalize
    X_train_norm, X_test_norm = normalize_data(X_train_raw, X_test_raw)
    
    # Create sequences
    X_train, y_train = create_sequences(X_train_norm, y_train_raw, sequence_length)
    X_test, y_test = create_sequences(X_test_norm, y_test_raw, sequence_length)
    
    print(f"\nSequence length: {sequence_length}")
    print(f"Training sequences: {len(X_train)}")
    print(f"Testing sequences: {len(X_test)}")
    
    # 4. Train Model
    model, history = train_model(X_train, y_train, X_test, y_test,
                                 epochs=50, batch_size=32)
    
    if model is None:
        print("\nModel training failed. Please install required packages:")
        print("pip install tensorflow scikit-learn yfinance pandas numpy")
        return
    
    # 5. Evaluate Model
    results = evaluate_model(model, X_test, y_test)
    
    # 6. Save Model
    if results:
        try:
            model.save('candlestick_model.h5')
            print("\n✓ Model saved as 'candlestick_model.h5'")
            print("\nTo load model:")
            print("from tensorflow import keras")
            print("model = keras.models.load_model('candlestick_model.h5')")
        except Exception as e:
            print(f"\nError saving model: {e}")
    
    # 7. Sample Prediction
    print("\n" + "="*50)
    print("SAMPLE PREDICTION")
    print("="*50)
    sample_idx = np.random.randint(0, len(X_test))
    sample = X_test[sample_idx:sample_idx+1]
    prediction = model.predict(sample)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]
    actual_class = y_test[sample_idx]
    
    print(f"Predicted: {'UP ↑' if predicted_class == 1 else 'DOWN ↓'} "
          f"(Confidence: {confidence*100:.2f}%)")
    print(f"Actual:    {'UP ↑' if actual_class == 1 else 'DOWN ↓'}")
    print(f"Result:    {'✓ CORRECT' if predicted_class == actual_class else '✗ INCORRECT'}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()

