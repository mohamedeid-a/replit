from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime
import requests
from transformers import pipeline
import threading
import logging
import os

app = Flask(__name__)

class SmartAIXServer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.sentiment_analyzer = None
        self.news_cache = {}
        self.lock = threading.Lock()
        self.setup_logging()
        self.load_model()

    def setup_logging(self):
        logging.basicConfig(
            filename='smart_aix_server.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def load_model(self):
        try:
            if os.path.exists('model.joblib'):
                self.model = joblib.load('model.joblib')
                self.scaler = joblib.load('scaler.joblib')
                logging.info("Model loaded successfully")
            else:
                self.initialize_model()
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            self.initialize_model()
            
    def initialize_model(self):
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        logging.info("New model initialized")
        
    def preprocess_data(self, data):
        try:
            df = pd.DataFrame(data)
            # Technical features
            df['SMA_fast'] = df['close'].rolling(window=12).mean()
            df['SMA_slow'] = df['close'].rolling(window=26).mean()
            df['RSI'] = self.calculate_rsi(df['close'])
            df['ATR'] = self.calculate_atr(df)
            df['Volatility'] = df['high'] - df['low']
            
            # Clean and fill NaN values
            df = df.fillna(method='bfill')
            
            features = ['SMA_fast', 'SMA_slow', 'RSI', 'ATR', 'Volatility']
            X = df[features].values
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            return X_scaled
            
        except Exception as e:
            logging.error(f"Error preprocessing data: {str(e)}")
            return None
            
    def calculate_rsi(self, prices, period=14):
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum()/period
        down = -seed[seed < 0].sum()/period
        rs = up/down
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100./(1. + rs)
        
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
                
            up = (up*(period-1) + upval)/period
            down = (down*(period-1) + downval)/period
            rs = up/down
            rsi[i] = 100. - 100./(1. + rs)
            
        return rsi
        
    def calculate_atr(self, df, period=14):
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
        
    def predict(self, data):
        try:
            X = self.preprocess_data(data)
            if X is None:
                return {'signal': 0, 'confidence': 0, 'volatility': 0}
                
            if self.model is None:
                self.load_model()
                
            with self.lock:
                prediction = self.model.predict(X)
                probabilities = self.model.predict_proba(X)
                
            signal = prediction[-1]  # Use last prediction
            confidence = np.max(probabilities[-1])
            volatility = data['ATR'][-1] if 'ATR' in data else 0
            
            return {
                'signal': int(signal),
                'confidence': float(confidence),
                'volatility': float(volatility),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error making prediction: {str(e)}")
            return {'signal': 0, 'confidence': 0, 'volatility': 0}
            
    def update_model(self, training_data):
        try:
            df = pd.DataFrame(training_data)
            X = self.preprocess_data(df)
            y = df['result'].values
            
            if X is None or len(X) != len(y):
                return False
                
            with self.lock:
                self.model.fit(X, y)
                joblib.dump(self.model, 'model.joblib')
                joblib.dump(self.scaler, 'scaler.joblib')
                
            logging.info(f"Model updated with {len(X)} samples")
            return True
            
        except Exception as e:
            logging.error(f"Error updating model: {str(e)}")
            return False
            
    def analyze_news_sentiment(self, text):
        if self.sentiment_analyzer is None:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert"
            )
        
        result = self.sentiment_analyzer(text)
        sentiment = result[0]
        return {
            'label': sentiment['label'],
            'score': float(sentiment['score'])
        }
        
    def fetch_economic_calendar(self):
        # Replace with your preferred economic calendar API
        # This is a placeholder using a mock response
        return {
            'events': [
                {
                    'datetime': datetime.now().isoformat(),
                    'currency': 'USD',
                    'event': 'NFP',
                    'impact': 'HIGH',
                    'forecast': '200K',
                    'previous': '190K'
                }
            ]
        }
        
    def start(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.host, self.port))
        server_socket.listen(5)
        
        logging.info(f"Server started on {self.host}:{self.port}")
        print(f"SmartAIX Server listening on {self.host}:{self.port}")
        
        while True:
            try:
                client_socket, addr = server_socket.accept()
                logging.info(f"Connection from {addr}")
                
                data = b""
                while True:
                    chunk = client_socket.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                
                if data:
                    request = json.loads(data.decode())
                    response = self.handle_request(request)
                    client_socket.send(json.dumps(response).encode())
                
                client_socket.close()
                
            except Exception as e:
                logging.error(f"Error handling connection: {str(e)}")
                continue
                
    def handle_request(self, request):
        command = request.get('command', '')
        
        if command == 'predict':
            return self.predict(request.get('data', {}))
            
        elif command == 'update_model':
            success = self.update_model(request.get('data', {}))
            return {'success': success}
            
        elif command == 'news_sentiment':
            sentiment = self.analyze_news_sentiment(request.get('text', ''))
            return sentiment
            
        elif command == 'economic_calendar':
            calendar = self.fetch_economic_calendar()
            return calendar
            
        return {'error': 'Unknown command'}

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'market_data' not in data:
            return jsonify({'error': 'Invalid input data'}), 400
        
        server = SmartAIXServer()
        result = server.predict(data['market_data'])
        
        if result is None:
            return jsonify({'error': 'Prediction failed'}), 500
            
        return jsonify(result)
    except Exception as e:
        logging.error(f"API error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/update_model', methods=['POST'])
def update_model():
    try:
        data = request.get_json()
        if not data or 'training_data' not in data:
            return jsonify({'error': 'Invalid training data'}), 400
        
        server = SmartAIXServer()
        success = server.update_model(data['training_data'])
        
        return jsonify({'success': success})
    except Exception as e:
        logging.error(f"Model update error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/news_sentiment', methods=['POST'])
def analyze_news():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        server = SmartAIXServer()
        sentiment = server.analyze_news_sentiment(data['text'])
        
        return jsonify(sentiment)
    except Exception as e:
        logging.error(f"Sentiment analysis error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)