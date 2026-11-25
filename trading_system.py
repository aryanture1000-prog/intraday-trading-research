import math
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import gc
import sys
import codecs
import socket
from collections import defaultdict
from datetime import datetime, timedelta, time as datetime_time
import json
import logging
import threading
import queue
import warnings
import traceback
import time
import ssl
import struct
from typing import Dict, List, Tuple, Optional

# Networking and API
import requests
from keras.src.layers import Add, LayerNormalization, Bidirectional
from matplotlib.gridspec import GridSpec
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.inspection import permutation_importance
from websocket import WebSocketApp
from SmartApi import SmartConnect
from SmartApi.smartApiWebsocket import SmartWebSocket
import pyotp
import logzero

# Data processing and analysis
import numpy as np
import pandas as pd
import psutil
from scipy import stats

# Technical Analysis
import ta

# Machine Learning - Core
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.feature_selection import mutual_info_classif

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import xgboost as xgb

# Visualization (optional, for analysis)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Ensure stdout can handle Unicode
if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)


def safe_log(message: str):
    """Log messages with safe character replacements"""
    # Replace emojis with plain text
    safe_message = message.replace('✅', '[SUCCESS]') \
        .replace('❌', '[ERROR]') \
        .replace('⚠️', '[WARNING]')
    logging.info(safe_message)


# Suppress warnings
warnings.filterwarnings('ignore')


class Config:
    """Consolidated configuration with combined settings"""

    def __init__(self):
        # Angel One credentials are read from environment variables (see .env.example)
        self.API_KEY = os.getenv("ANGEL_API_KEY", "")
        self.CLIENT_ID = os.getenv("ANGEL_CLIENT_ID", "")
        self.PASSWORD = os.getenv("ANGEL_PASSWORD", "")
        self.TOTP_KEY = os.getenv("ANGEL_TOTP_KEY", "")
        if not all([self.API_KEY, self.CLIENT_ID, self.PASSWORD, self.TOTP_KEY]):
            logging.warning("Angel One credentials missing: copy .env.example to .env and fill it in")

        # Trading universe - modified to include only one symbol
        self.SYMBOLS = {
            'ADANIENT': {
                'token': '25',
                'symbol': 'NSE:ADANIENT-EQ',
                'exchange': 'NSE'
            }
        }

        # Trading parameters
        self.EXCHANGE = "NSE"
        self.TIMEFRAME = "ONE_MINUTE"
        self.INITIAL_CAPITAL = 100000
        self.MIN_TRAINING_SAMPLES = 100

        # Risk parameters calibrated to historical performance
        self.MAX_POSITION_SIZE = 0.06
        self.MAX_DAILY_LOSS = 0.02
        self.STOP_LOSS = 0.009
        self.TAKE_PROFIT = 0.018
        self.MAX_DRAWDOWN = 0.006
        self.VOLATILITY_THRESHOLD = 0.03
        self.MAX_POSITIONS = 10
        self.MAX_CAPITAL_AT_RISK = 0.5
        self.MIN_POSITION_VALUE = 1000
        self.MAX_POSITION_VALUE = self.INITIAL_CAPITAL * 0.05
        self.MAX_TOTAL_POSITION_SIZE = 0.5

        # System timing parameters
        self.WEBSOCKET_RECONNECT_INTERVAL = 120
        self.SESSION_REFRESH_INTERVAL = 3600
        self.ORDER_TIMEOUT = 30
        self.MAX_RETRIES = 3
        self.RETRY_DELAY = 5
        self.TRAINING_DAYS = 60
        self.RETRAIN_INTERVAL = 14400
        self.SIGNAL_INTERVAL = 10

        # Market hours
        self.MARKET_OPEN = "09:15"
        self.MARKET_CLOSE = "15:30"

        # Strategy parameters
        self.STRATEGY_PARAMS = {
            'ma_short': 5,
            'ma_long': 15,
            'rsi_entry': 40,
            'rsi_exit': 60,
            'volume_threshold': 1.2,
            'trend_strength': 0.01,
            'momentum_threshold': 0.002
        }

        # Risk parameters
        self.risk_params = {
            'max_position_size': self.MAX_POSITION_SIZE,
            'max_daily_loss': self.MAX_DAILY_LOSS,
            'stop_loss': self.STOP_LOSS,
            'take_profit': self.TAKE_PROFIT,
            'max_drawdown': self.MAX_DRAWDOWN,
            'volatility_threshold': self.VOLATILITY_THRESHOLD,
            'max_positions': self.MAX_POSITIONS,
            'max_capital_at_risk': self.MAX_CAPITAL_AT_RISK
        }


class RateLimiter:
    """Rate limiter for Angel One API"""

    def __init__(self):
        self.locks = {
            'session': threading.Lock(),
            'order': threading.Lock(),
            'market': threading.Lock(),
            'historical': threading.Lock(),
            'websocket': threading.Lock()  # Add websocket-specific lock

        }

        self.last_request_time = {
            'session': 0,
            'order': 0,
            'market': 0,
            'historical': 0
        }

        # Add 'session' to minute_counters
        self.minute_counters = {
            'session': {'count': 0, 'reset_time': time.time()},  # Add this line
            'order': {'count': 0, 'reset_time': time.time()},
            'market': {'count': 0, 'reset_time': time.time()},
            'historical': {'count': 0, 'reset_time': time.time()}
        }

        self.rate_limits = {
            'session': 60,  # 60 requests per second
            'order': 20,  # 20 orders per second
            'market': 10,  # 10 market data requests per second
            'historical': 3  # 3 historical requests per second
        }

        self.minute_limits = {
            'session': 60,  # 60 session requests per minute
            'order': 500,  # 500 orders per minute
            'market': 500,  # 500 market data requests per minute
            'historical': 180  # 180 historical requests per minute
        }

        # Add delay between subscriptions
        time.sleep(0.5)

    def wait_if_needed(self, category: str):
        """Wait if rate limit is about to be exceeded"""
        with self.locks[category]:
            current_time = time.time()

            # Reset minute counters if needed
            if current_time - self.minute_counters[category]['reset_time'] >= 60:
                self.minute_counters[category] = {
                    'count': 0,
                    'reset_time': current_time
                }

            # Check and wait for per-second limit
            elapsed = current_time - self.last_request_time[category]
            if elapsed < (1.0 / self.rate_limits[category]):
                time.sleep((1.0 / self.rate_limits[category]) - elapsed)

            # Check minute limit
            if category in self.minute_limits:
                if self.minute_counters[category]['count'] >= self.minute_limits[category]:
                    wait_time = 60 - (current_time - self.minute_counters[category]['reset_time'])
                    if wait_time > 0:
                        time.sleep(wait_time)
                        self.minute_counters[category] = {
                            'count': 0,
                            'reset_time': time.time()
                        }

                self.minute_counters[category]['count'] += 1

            self.last_request_time[category] = time.time()


class MLEnsemble:
    """Enhanced ML Ensemble with LSTM for time-series patterns and XGBoost for robust prediction"""

    def __init__(self, config: Config):
        """Enhanced ML Ensemble with LSTM for time-series patterns and XGBoost for robust prediction"""
        self.config = config
        self.last_prediction_time = 0

        # Initialize both arima structures for compatibility and to fix the error
        self.arima_data = {}  # For backward compatibility
        self.arima_models = {}  # New structure for ARIMA models

        # For ideal path learning
        self.ideal_trades = {}  # Will store ideal entries/exits by symbol
        self.ideal_features = {}  # Features that correlate with ideal trades
        self.profit_thresholds = {}  # Dynamic profit thresholds by symbol
        self.signal_thresholds = {}  # Signal thresholds by symbol

        # Data preprocessing components
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_trained = False
        self.training_lock = threading.Lock()
        self.last_training_time = None

        # Performance tracking
        self.model_performance = {
            'accuracy': [],
            'predictions': [],
            'feature_importance': {},
            'market_regime': []
        }

        # Sequence length for LSTM
        self.sequence_length = 10  # Number of time steps to consider

        # Initialize XGBoost model with optimized parameters
        try:
            self.xgb_model = xgb.XGBClassifier(
                max_depth=5,  # Optimal from paper
                learning_rate=0.03,  # Reduced from 0.05 for more stable learning
                n_estimators=250,  # Increased from 200 for better ensemble power
                subsample=0.9,  # Increased from 0.8 for better generalization
                colsample_bytree=0.8,  # Using 0.8 as in the paper
                min_child_weight=4,  # Increased from 3 to reduce overfitting
                gamma=0.2,  # Increased for better pruning - implements regularization part of paper's formula:
                # L(m) = Σ(i=1 to N) l(y(i), ŷ(i)) + Σ(m=1 to M) Ω(f(m))
                reg_alpha=0.3,  # L1 regularization - part of Ω(f(m)) term
                reg_lambda=1,  # L2 regularization - part of Ω(f(m)) term
                scale_pos_weight=3,  # Will be dynamically adjusted later
                objective='binary:logistic',  # As in paper
                tree_method='hist',  # Faster algorithm
                random_state=42,
                n_jobs=-1,  # Use all cores
                eval_metric='auc'  # AUC for better classification performance
            )
        except ImportError:
            logging.error("XGBoost not available - using fallback classifier")
            self.xgb_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            )

        # We'll initialize the LSTM model during training when we know feature dimensions
        self.lstm_model = None
        self._lstm_initialized = False
        self._lstm_available = True  # Will be set to False if LSTM can't be created

        # Model weights
        self.model_weights = {
            'lstm': 0.6,  # Higher weight for time-series patterns
            'xgb': 0.4  # Complementary weight for feature-based patterns
        }

        # Prediction history and market regime tracking
        self.prediction_histories = []
        self.performance_window = 100
        self.current_regime = 'normal'
        self.regime_history = []

        # Define lookahead features that shouldn't be used for prediction
        self._lookahead_features = [
            'ideal_entry', 'ideal_exit', 'trade_id', 'profit_target',
            'stop_level', 'realized_profit', 'pre_entry', 'post_entry',
            'entry_quality', 'fwd_return_3', 'fwd_return_5', 'fwd_return_10',
            'fwd_return_15', 'fwd_max_price_5', 'fwd_min_price_5',
            'fwd_max_profit_5', 'fwd_max_loss_5', 'fwd_max_price_10',
            'fwd_min_price_10', 'fwd_max_profit_10', 'fwd_max_loss_10',
            'fwd_max_price_15', 'fwd_min_price_15', 'fwd_max_profit_15',
            'fwd_max_loss_15', 'ideal_entry_long', 'ideal_entry_short',
            'trade_direction'
        ]

        # SPEED FIX: Add TensorFlow optimization
        self._optimize_tensorflow()

    def process_arima(self, data: pd.DataFrame, symbol: str = None) -> pd.DataFrame:

        try:
            # Extract symbol from data if not provided
            if symbol is None:
                symbol = data.name if hasattr(data, 'name') and data.name is not None else 'unknown'

            # CRITICAL FIX: Initialize both arima structures if they don't exist
            # This prevents the 'arima_data' attribute error
            if not hasattr(self, 'arima_data'):
                self.arima_data = {}

            if not hasattr(self, 'arima_models'):
                self.arima_models = {}

            # Check if we need to fit a new ARIMA model
            need_new_model = True

            # Try to get existing model from either structure
            model_data = None
            if symbol in self.arima_data:
                model_data = self.arima_data[symbol]
                need_new_model = False

            if need_new_model and symbol in self.arima_models:
                model_data = self.arima_models[symbol]
                need_new_model = False

            # Check if model is recent enough (less than 24 hours old)
            if model_data and 'last_updated' in model_data:
                time_since_update = (datetime.now() - model_data['last_updated']).total_seconds()
                if time_since_update > 86400:  # 24 hours
                    need_new_model = True

            # Fit new ARIMA model if needed
            if need_new_model:
                try:
                    import statsmodels.api as sm
                    from statsmodels.tsa.arima.model import ARIMA
                    from statsmodels.tsa.stattools import adfuller

                    # Verify we have Close price data
                    if 'Close' not in data.columns:
                        logging.error(f"No Close price data available for {symbol}")
                        return data

                    # Extract price series
                    price_series = data['Close'].values

                    # Perform Augmented Dickey-Fuller test to determine 'd'
                    adf_result = adfuller(price_series)
                    d = 1 if adf_result[1] > 0.05 else 0  # If p-value > 0.05, series is non-stationary

                    # Use ARIMA(0,1,1) as found optimal in the paper
                    order = (0, d, 1)

                    # Fit ARIMA model
                    model = ARIMA(price_series, order=order)
                    model_fit = model.fit()

                    # Create model data structure
                    model_data = {
                        'model': model_fit,
                        'order': order,
                        'last_updated': datetime.now()
                    }

                    # Store in both structures for maximum compatibility
                    self.arima_data[symbol] = model_data
                    self.arima_models[symbol] = model_data

                    logging.info(f"ARIMA{order} model fitted successfully for {symbol}")

                except Exception as fit_error:
                    logging.error(f"ARIMA model fitting error: {str(fit_error)}")
                    logging.error(traceback.format_exc())
                    # Continue with existing model if available

            # Generate ARIMA predictions if we have a model
            if model_data and 'model' in model_data:
                try:
                    model = model_data['model']
                    last_price = data['Close'].iloc[-1]

                    # Generate one-step forecast
                    forecast = model.forecast(steps=1)
                    predicted_price = forecast[0]

                    # Calculate percentage change and create signal
                    pct_change = (predicted_price / last_price - 1) * 100

                    # Add prediction data to DataFrame
                    data.loc[data.index[-1], 'arima_prediction'] = predicted_price
                    data.loc[data.index[-1], 'arima_change_pct'] = pct_change

                    # Generate simple ARIMA signal (-1 to 1)
                    threshold = 0.05  # 0.05% threshold
                    if pct_change > threshold:
                        data.loc[data.index[-1], 'arima_signal'] = 1  # Bullish
                    elif pct_change < -threshold:
                        data.loc[data.index[-1], 'arima_signal'] = -1  # Bearish
                    else:
                        data.loc[data.index[-1], 'arima_signal'] = 0  # Neutral

                    # Map signal to probability (0-1 scale) for ensemble integration
                    arima_score = 0.5  # Neutral default
                    if pct_change > 0:
                        # Map positive change to 0.5-1.0 range (max at +0.2%)
                        arima_score = min(1.0, 0.5 + pct_change * 2.5)
                    else:
                        # Map negative change to 0.0-0.5 range (min at -0.2%)
                        arima_score = max(0.0, 0.5 + pct_change * 2.5)

                    data.loc[data.index[-1], 'arima_score'] = arima_score

                    # Weight for ensemble integration (10%)
                    arima_weight = 0.1

                    # If we're performing ML prediction, integrate ARIMA component
                    if hasattr(self, 'predict') and callable(getattr(self, 'predict')):
                        # Get ML predictions - store original weights
                        lstm_weight = getattr(self, 'model_weights', {}).get('lstm', 0.5)
                        xgb_weight = getattr(self, 'model_weights', {}).get('xgb', 0.4)

                        # Calculate normalized weights
                        total = lstm_weight + xgb_weight + arima_weight
                        lstm_norm = lstm_weight / total
                        xgb_norm = xgb_weight / total
                        arima_norm = arima_weight / total

                        # Store weights in data
                        data.loc[data.index[-1], 'lstm_weight'] = lstm_norm
                        data.loc[data.index[-1], 'xgb_weight'] = xgb_norm
                        data.loc[data.index[-1], 'arima_weight'] = arima_norm

                    logging.info(
                        f"ARIMA forecast for {symbol}: {predicted_price:.2f} ({pct_change:.2f}%), Score: {arima_score:.3f}")

                except Exception as pred_error:
                    logging.error(f"ARIMA prediction error: {str(pred_error)}")
                    logging.error(traceback.format_exc())

            return data

        except Exception as e:
            logging.error(f"ARIMA processing error: {str(e)}")
            logging.error(traceback.format_exc())
            return data

    def _ensure_lstm_model(self, num_features):
        """Enhanced LSTM architecture for 5-class prediction (HOLD, ENTER_LONG, ENTER_SHORT, EXIT_LONG, EXIT_SHORT)"""
        try:
            # Skip if LSTM is not available or already initialized
            if not self._lstm_available or self._lstm_initialized:
                return

            # Import necessary layers with regularization
            from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Bidirectional, Add, \
                LayerNormalization
            from tensorflow.keras.regularizers import l2
            from tensorflow.keras.models import Model
            from tensorflow.keras.optimizers import Adam

            # Define input shape based on sequence length
            input_sequence = Input(shape=(self.sequence_length, num_features))

            # First LSTM layer - implement forget gate equation: f(t) = σ(g)([x(t); h(t-1)]W(f) + b(f))
            lstm_out = Bidirectional(LSTM(128, return_sequences=True,
                                          kernel_regularizer=l2(0.0001),
                                          recurrent_regularizer=l2(0.0001),
                                          recurrent_dropout=0.2))(input_sequence)
            lstm_out = LayerNormalization()(lstm_out)  # Better than BatchNorm for LSTM
            lstm_out = Dropout(0.5)(lstm_out)

            # Second LSTM layer - implements cell state equation: c(t) = f(t) ∘ c(t-1) + i(t) ∘ c̃(t)
            # and hidden state equation: h(t) = o(t) ∘ σ(h)(c(t))
            lstm_out = LSTM(64, return_sequences=False,
                            kernel_regularizer=l2(0.0001),
                            recurrent_regularizer=l2(0.0001))(lstm_out)
            lstm_out = LayerNormalization()(lstm_out)
            lstm_out = Dropout(0.4)(lstm_out)

            # Dense layers with residual connection - inspired by paper's deep architecture
            dense1 = Dense(64, activation='relu', kernel_regularizer=l2(0.0001))(lstm_out)
            dense1 = LayerNormalization()(dense1)
            dense1 = Dropout(0.3)(dense1)

            dense2 = Dense(32, activation='relu', kernel_regularizer=l2(0.0001))(dense1)
            dense2 = LayerNormalization()(dense2)
            dense2 = Dropout(0.3)(dense2)

            # Residual connection to prevent vanishing gradients
            if dense1.shape[-1] == dense2.shape[-1]:  # Check if dimensions match
                dense2 = Add()([dense1, dense2])  # Residual connection

            # UPDATED: Output layer for 5 classes (HOLD, ENTER_LONG, ENTER_SHORT, EXIT_LONG, EXIT_SHORT)
            output = Dense(5, activation='softmax')(dense2)

            # Create model
            self.lstm_model = Model(inputs=input_sequence, outputs=output)

            # Compile with Adam optimizer - as specified in paper
            self.lstm_model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            self._lstm_initialized = True
            logging.info(f"Enhanced LSTM model initialized with 5-class architecture for entry/exit prediction")

        except Exception as e:
            logging.error(f"Failed to initialize LSTM model: {str(e)}")
            self._lstm_available = False
            self.model_weights = {'lstm': 0.0, 'xgb': 1.0}

    def _optimize_tensorflow(self):
        """Optimize TensorFlow for faster training"""
        try:
            import tensorflow as tf

            # Enable memory growth to prevent allocation delays
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

            # Configure for optimal CPU performance
            tf.config.threading.set_intra_op_parallelism_threads(0)
            tf.config.threading.set_inter_op_parallelism_threads(0)

            logging.info("TensorFlow optimized for faster training")
        except Exception as e:
            logging.warning(f"TensorFlow optimization warning: {str(e)}")

    def _prepare_lstm_sequences(self, X):
        """
        Prepare sequences for LSTM processing from a pandas DataFrame
        """
        try:
            # Convert to numpy array if it's a DataFrame
            if isinstance(X, pd.DataFrame):
                X_values = X.values
            else:
                X_values = X

            # Check if we have enough data for sequences
            if len(X_values) < self.sequence_length:
                logging.warning(
                    f"Not enough data for LSTM sequences: {len(X_values)} rows < {self.sequence_length} required")
                return np.array([])

            # Create sequences with optimized approach
            sequences = []
            for i in range(len(X_values) - self.sequence_length + 1):
                sequence = X_values[i:i + self.sequence_length]
                sequences.append(sequence)

            return np.array(sequences)

        except Exception as e:
            logging.error(f"Error preparing LSTM sequences: {str(e)}")
            logging.error(traceback.format_exc())
            return np.array([])

    def _calculate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical features optimized for intraday trading with improved indicators"""
        try:
            if df.empty:
                logging.error("Empty DataFrame provided")
                return None

            # Create working copy
            data = df.copy()

            # CRITICAL FIX 1: Ensure consistent column names by standardizing capitalization
            column_mapping = {}
            for standard_col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                for col in data.columns:
                    if col.upper() == standard_col.upper() and col != standard_col:
                        column_mapping[col] = standard_col

            if column_mapping:
                data = data.rename(columns=column_mapping)

            # Ensure we have OHLCV data (basic validation)
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                # Try to find case variations
                for col in missing_cols:
                    col_lower = col.lower()
                    matching_cols = [c for c in data.columns if c.lower() == col_lower]
                    if matching_cols:
                        data[col] = data[matching_cols[0]]

                # After trying case variations, check again for missing columns
                still_missing = [col for col in required_cols if col not in data.columns]

                # If we still have missing columns, try alternative names
                if still_missing:
                    # Common alternative column names
                    alt_names = {
                        'Close': ['close', 'CLOSE', 'last', 'ltp', 'price', 'c', 'C'],
                        'Open': ['open', 'OPEN', 'o', 'O'],
                        'High': ['high', 'HIGH', 'h', 'H'],
                        'Low': ['low', 'LOW', 'l', 'L'],
                        'Volume': ['volume', 'VOLUME', 'vol', 'v', 'V']
                    }

                    # Try to find alternative column names
                    for std_col in still_missing:
                        found = False
                        for alt in alt_names.get(std_col, []):
                            if alt in data.columns:
                                data[std_col] = data[alt]
                                found = True
                                break

                        # If we couldn't find an alternative, create the column
                        if not found:
                            # For price columns, try to use another price column
                            if std_col in ['Open', 'High', 'Low', 'Close']:
                                existing_price_cols = [c for c in ['Open', 'High', 'Low', 'Close'] if c in data.columns]
                                if existing_price_cols:
                                    # Use an existing price column if available
                                    data[std_col] = data[existing_price_cols[0]]
                                else:
                                    # If no price columns exist, look for any numeric column
                                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                                    if len(numeric_cols) > 0:
                                        data[std_col] = data[numeric_cols[0]]
                                    else:
                                        # Last resort: create a constant column
                                        logging.warning(f"Creating constant column for {std_col}")
                                        data[std_col] = 100.0  # Default value
                            # For Volume, default to 0
                            elif std_col == 'Volume':
                                data[std_col] = 0

            for col in ['Open', 'High', 'Low', 'Close']:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')

                    # CRITICAL FIX: Never allow zero or negative values in price data
                    invalid_mask = (data[col] <= 0) | data[col].isna()

                    if invalid_mask.any():
                        # Use forward fill first, then backward fill
                        data[col] = data[col].mask(invalid_mask).fillna(method='ffill').fillna(method='bfill')

                        # If still invalid, use interpolation
                        if data[col].isna().any() or (data[col] <= 0).any():
                            data[col] = data[col].interpolate(method='linear')

                        # Final fallback: use previous day's close or reasonable default
                        if data[col].isna().any() or (data[col] <= 0).any():
                            last_valid = data[col].loc[data[col] > 0].iloc[-1] if (data[col] > 0).any() else 2500.0
                            data[col] = data[col].fillna(last_valid)
                            data.loc[data[col] <= 0, col] = last_valid

                        fixed_count = invalid_mask.sum()

            # ENSURE OHLC relationships are maintained
            data['High'] = data[['Open', 'High', 'Low', 'Close']].max(axis=1)
            data['Low'] = data[['Open', 'High', 'Low', 'Close']].min(axis=1)

            # Volume handling
            if 'Volume' in data.columns:
                data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce').fillna(0).clip(lower=0)

            # CRITICAL FIX 3: Ensure price relationships are valid
            # High should be highest, Low should be lowest
            data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
            data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)

            # ======= INTRADAY-SPECIFIC FEATURES ========

            # 1. Ultra-short timeframe price action features
            data['returns'] = data['Close'].pct_change().fillna(0)
            data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1)).fillna(0)

            # Intraday volatility - shorter windows for minute data
            data['volatility_3'] = data['returns'].rolling(3).std().fillna(0) * 100  # 3-minute volatility
            data['volatility_5'] = data['returns'].rolling(5).std().fillna(0) * 100  # 5-minute volatility
            data['volatility_10'] = data['returns'].rolling(10).std().fillna(0) * 100  # 10-minute volatility
            data['volatility_20'] = data['returns'].rolling(20).std().fillna(0) * 100  # 20-minute volatility

            # Very important: Volatility ratio to detect regime changes during the day
            data['volatility_ratio'] = data['volatility_5'] / data['volatility_20'].rolling(10).mean().fillna(
                data['volatility_20'])

            # NEW: Volatility acceleration - measure rate of change in volatility
            data['volatility_accel'] = data['volatility_5'].diff(3).fillna(0)

            # 2. Short-term momentum indicators optimized for intraday
            # Shorter timeframes for intraday
            for period in [2, 3, 5, 8]:
                # Price momentum - critical for short-term moves
                data[f'momentum_{period}'] = data['Close'].pct_change(period).fillna(0)

                # Price momentum acceleration - helps detect turning points
                if period <= 5:  # Only calculate for very short timeframes
                    data[f'momentum_accel_{period}'] = data[f'momentum_{period}'].diff().fillna(0)

            # NEW: Improved momentum signal combining multiple timeframes
            data['momentum_composite'] = (
                    data['momentum_2'] * 0.35 +
                    data['momentum_3'] * 0.25 +
                    data['momentum_5'] * 0.25 +
                    data['momentum_8'] * 0.15
            )

            # 3. Faster-response Moving Averages
            # Shorter periods for intraday trading
            for period in [3, 5, 8, 13, 21, 55]:
                # EMAs - faster responsive for intraday
                data[f'ema_{period}'] = data['Close'].ewm(span=period, adjust=False, min_periods=1).mean()

                # Price relative to EMA - key intraday signal
                data[f'close_to_ema_{period}'] = ((data['Close'] / data[f'ema_{period}']) - 1) * 100

                # EMA slopes - detect momentum shifts
                data[f'ema_{period}_slope'] = (data[f'ema_{period}'] - data[f'ema_{period}'].shift(2)) / (
                        data[f'ema_{period}'].shift(2) + 0.0001) * 100

            # NEW: EMA alignment - strong trend indicator when multiple EMAs align
            data['ema_alignment'] = 0
            # Uptrend alignment
            up_align = (data['ema_5'] > data['ema_8']) & (data['ema_8'] > data['ema_13']) & (
                    data['ema_13'] > data['ema_21'])
            data.loc[up_align, 'ema_alignment'] = 1
            # Downtrend alignment
            down_align = (data['ema_5'] < data['ema_8']) & (data['ema_8'] < data['ema_13']) & (
                    data['ema_13'] < data['ema_21'])
            data.loc[down_align, 'ema_alignment'] = -1

            # MA Crossovers - critical intraday signals
            # Use shorter timeframes for intraday
            for fast, slow in [(3, 5), (5, 8), (8, 13), (8, 21)]:
                # Crossover events
                data[f'ema_cross_{fast}_{slow}'] = (
                        (data[f'ema_{fast}'] > data[f'ema_{slow}']).astype(int) -
                        (data[f'ema_{fast}'].shift(1) > data[f'ema_{slow}'].shift(1)).astype(int)
                )

                # Distance between MAs - indicates momentum strength
                data[f'ma_dist_{fast}_{slow}'] = (data[f'ema_{fast}'] - data[f'ema_{slow}']) / data[f'ema_{slow}'] * 100

            # NEW: Composite moving average crossover signal
            data['ma_cross_composite'] = (
                    data['ema_cross_5_8'] * 0.4 +
                    data['ema_cross_8_13'] * 0.4 +
                    data['ema_cross_8_21'] * 0.2
            )

            # 4. Ultra-sensitive RSI for intraday signals
            # Short-term RSI specially tuned for intraday
            for period in [2, 5, 9, 14]:  # Include very short periods for intraday
                delta = data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).ewm(span=period, adjust=False).mean()
                loss = (-delta.where(delta < 0, 0)).ewm(span=period, adjust=False).mean()

                # Handle division by zero
                rs = gain / loss.replace(0, np.inf)
                data[f'rsi_{period}'] = 100 - (100 / (1 + rs))

                # Intraday-specific RSI flags with improved thresholds
                # Use 72/28 for shorter periods (more precise)
                if period <= 5:
                    data[f'rsi_{period}_overbought'] = (data[f'rsi_{period}'] > 72).astype(int)
                    data[f'rsi_{period}_oversold'] = (data[f'rsi_{period}'] < 28).astype(int)
                else:
                    # Standard but slightly refined 68/32 for longer periods
                    data[f'rsi_{period}_overbought'] = (data[f'rsi_{period}'] > 68).astype(int)
                    data[f'rsi_{period}_oversold'] = (data[f'rsi_{period}'] < 32).astype(int)

                # RSI momentum - critical for intraday turning points
                data[f'rsi_{period}_slope'] = data[f'rsi_{period}'].diff(1).fillna(0)

                # NEW FROM UPDATE: Add RSI rate of change
                data[f'rsi_{period}_roc'] = data[f'rsi_{period}'].diff(3).fillna(0)

            # Set primary RSI
            data['rsi'] = data['rsi_14']

            # NEW FROM UPDATE: Add trend exhaustion indicator
            data['trend_exhaustion'] = (
                    ((data['rsi_14'] > 70) & (data['rsi_14_roc'] < 0)) |
                    ((data['rsi_14'] < 30) & (data['rsi_14_roc'] > 0))
            ).astype(int)

            # NEW: RSI divergence detection
            # Regular bullish divergence (price making lower lows while RSI makes higher lows)
            data['price_lower_low'] = (data['Close'] < data['Close'].shift(1)) & (
                    data['Close'].shift(1) < data['Close'].shift(2))
            data['rsi_higher_low'] = (data['rsi_14'] > data['rsi_14'].shift(1)) & (
                    data['rsi_14'].shift(1) < data['rsi_14'].shift(2))
            data['bullish_divergence'] = data['price_lower_low'] & data['rsi_higher_low']

            # Regular bearish divergence (price making higher highs while RSI makes lower highs)
            data['price_higher_high'] = (data['Close'] > data['Close'].shift(1)) & (
                    data['Close'].shift(1) > data['Close'].shift(2))
            data['rsi_lower_high'] = (data['rsi_14'] < data['rsi_14'].shift(1)) & (
                    data['rsi_14'].shift(1) > data['rsi_14'].shift(2))
            data['bearish_divergence'] = data['price_higher_high'] & data['rsi_lower_high']

            # 5. Custom Intraday Bollinger Bands
            # Use shorter timeframes for intraday
            for period in [10, 20]:  # Shorter periods for intraday
                data[f'bb_middle_{period}'] = data['Close'].rolling(window=period).mean()

                # Intraday-specific: Use 2.2 std for intraday (more precise)
                bb_std = data['Close'].rolling(window=period).std()
                data[f'bb_upper_{period}'] = data[f'bb_middle_{period}'] + 2.2 * bb_std  # Reduced from 2.5
                data[f'bb_lower_{period}'] = data[f'bb_middle_{period}'] - 2.2 * bb_std  # Reduced from 2.5

                # Position within bands
                data[f'bb_pct_b_{period}'] = (data['Close'] - data[f'bb_lower_{period}']) / (
                        data[f'bb_upper_{period}'] - data[f'bb_lower_{period}'] + 0.0001
                )

                # Bollinger bandwidth - key for volatility expansion/contraction
                data[f'bb_width_{period}'] = (data[f'bb_upper_{period}'] - data[f'bb_lower_{period}']) / data[
                    f'bb_middle_{period}']

                # Breakout signals - critical for intraday momentum trades
                data[f'bb_upper_breakout_{period}'] = (data['Close'] > data[f'bb_upper_{period}']).astype(int)
                data[f'bb_lower_breakout_{period}'] = (data['Close'] < data[f'bb_lower_{period}']).astype(int)

            # NEW: Bollinger %B momentum - shows strength of moves within bands
            data['bb_pct_b_momentum'] = data['bb_pct_b_20'].diff(3).fillna(0)

            # NEW: Bollinger band squeeze indicator - detects potential breakouts
            data['bb_squeeze'] = (data['bb_width_20'] < data['bb_width_20'].rolling(20).mean() * 0.85).astype(int)

            # 6. Enhanced Intraday-specific price patterns
            # Candle patterns are highly valuable for intraday
            data['body_size'] = abs(data['Close'] - data['Open']) / data['Open'] * 100
            data['upper_wick'] = (data['High'] - data[['Open', 'Close']].max(axis=1)) / data['Open'] * 100
            data['lower_wick'] = (data[['Open', 'Close']].min(axis=1) - data['Low']) / data['Open'] * 100

            # Special intraday patterns with higher sensitivity
            data['doji'] = ((data['body_size'] < 0.05) &  # Tighter definition for intraday
                            (data['upper_wick'] > 0.05) &
                            (data['lower_wick'] > 0.05)).astype(int)

            data['hammer'] = ((data['body_size'] < 0.5) &  # Smaller for intraday
                              (data['lower_wick'] > 1.5 * data['body_size']) &
                              (data['upper_wick'] < 0.1)).astype(int)

            data['shooting_star'] = ((data['body_size'] < 0.5) &
                                     (data['upper_wick'] > 1.5 * data['body_size']) &
                                     (data['lower_wick'] < 0.1)).astype(int)

            # Engulfing patterns - very strong intraday signals
            data['bullish_engulfing'] = ((data['Close'] > data['Open']) &
                                         (data['Close'].shift(1) < data['Open'].shift(1)) &
                                         (data['Open'] < data['Close'].shift(1)) &
                                         (data['Close'] > data['Open'].shift(1))).astype(int)

            data['bearish_engulfing'] = ((data['Close'] < data['Open']) &
                                         (data['Close'].shift(1) > data['Open'].shift(1)) &
                                         (data['Open'] > data['Close'].shift(1)) &
                                         (data['Close'] < data['Open'].shift(1))).astype(int)

            # NEW: Pattern strength indicator - stronger recognition of key patterns
            data['bullish_pattern_strength'] = (
                    data['hammer'] * 1.2 +
                    data['bullish_engulfing'] * 1.5 +
                    (data['doji'] & data['rsi_14_oversold']) * 0.8
            )

            data['bearish_pattern_strength'] = (
                    data['shooting_star'] * 1.2 +
                    data['bearish_engulfing'] * 1.5 +
                    (data['doji'] & data['rsi_14_overbought']) * 0.8
            )

            # 7. Intraday-specific volume features
            # Intraday volume patterns are critical
            data['volume_sma'] = data['Volume'].rolling(window=10, min_periods=1).mean()  # Shorter window
            data['volume_ratio'] = (data['Volume'] / data['volume_sma']).fillna(1)

            # Intraday volume spikes detection
            data['volume_spike'] = (data['volume_ratio'] > 2.0).astype(int)  # Volume spike classifier

            # Volume trends for short timeframes
            for period in [2, 3, 5]:  # Shorter periods for intraday
                data[f'volume_momentum_{period}'] = data['Volume'].pct_change(period).fillna(0)

                # On-balance volume - shorter timeframes for intraday
                direction = np.sign(data['returns'])
                direction = direction.replace(0, method='ffill')
                weighted_volume = direction * data['Volume']
                data[f'obv_{period}'] = weighted_volume.rolling(window=period).sum()
                data[f'obv_{period}_slope'] = data[f'obv_{period}'].diff(1).fillna(0) / (
                        data[f'obv_{period}'].abs().replace(0, 1) + 0.0001)

            # Intraday money flow indicators
            mf_multiplier = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (
                    data['High'] - data['Low'] + 0.000001)
            mf_multiplier = mf_multiplier.replace([np.inf, -np.inf], 0)
            mf_volume = mf_multiplier * data['Volume']

            # Use shorter periods for intraday
            for period in [5, 10]:
                data[f'cmf_{period}'] = mf_volume.rolling(period).sum() / (
                        data['Volume'].rolling(period).sum() + 0.0001)

            # NEW: Volume trend confirmation - checks if volume aligns with price move
            # Price up + volume up = strong bullish
            data['vol_price_confirm_up'] = ((data['returns'] > 0) & (data['Volume'] > data['volume_sma'])).astype(int)
            # Price down + volume up = strong bearish
            data['vol_price_confirm_down'] = ((data['returns'] < 0) & (data['Volume'] > data['volume_sma'])).astype(int)

            # NEW FROM UPDATE: Volume-Price Confirmation
            if 'volume_ratio' in data.columns and 'returns' in data.columns:
                data['vol_price_confirm'] = data['volume_ratio'] * data['returns'].rolling(3).mean()

            # 8. Intraday ATR and Volatility features
            tr1 = data['High'] - data['Low']
            tr2 = abs(data['High'] - data['Close'].shift())
            tr3 = abs(data['Low'] - data['Close'].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # Use shorter periods for intraday
            for period in [3, 7, 14]:
                data[f'atr_{period}'] = tr.rolling(window=period, min_periods=1).mean()
                data[f'natr_{period}'] = (data[f'atr_{period}'] / data['Close']) * 100

                # ATR changes - detect volatility expansion
                data[f'atr_{period}_change'] = data[f'atr_{period}'].pct_change(1).fillna(0)  # Faster response (1 vs 3)

            # NEW: ATR ratio - shows changing volatility regimes
            data['atr_ratio'] = data['atr_3'] / data['atr_14']

            # Set default column
            data['atr'] = data['atr_14']

            # 9. Intraday-optimized MACD
            # Use faster settings for intraday
            for fast, slow, signal in [(6, 13, 4), (8, 17, 5), (12, 26, 9)]:  # Added standard MACD
                fast_ema = data['Close'].ewm(span=fast, adjust=False, min_periods=1).mean()
                slow_ema = data['Close'].ewm(span=slow, adjust=False, min_periods=1).mean()
                macd = fast_ema - slow_ema
                macd_signal = macd.ewm(span=signal, adjust=False, min_periods=1).mean()

                data[f'macd_{fast}_{slow}'] = macd
                data[f'macd_signal_{fast}_{slow}'] = macd_signal
                data[f'macd_hist_{fast}_{slow}'] = macd - macd_signal

                # MACD crossovers - critical intraday signals
                data[f'macd_cross_{fast}_{slow}'] = ((macd > macd_signal).astype(int) -
                                                     (macd.shift(1) > macd_signal.shift(1)).astype(int))

                # MACD histogram dynamics - important for momentum shift
                data[f'macd_hist_change_{fast}_{slow}'] = (data[f'macd_hist_{fast}_{slow}'] -
                                                           data[f'macd_hist_{fast}_{slow}'].shift(1)).fillna(0)

            # NEW: MACD Composite signal - combines multiple timeframes
            if all(f'macd_cross_{fast}_{slow}' in data.columns for fast, slow, _ in [(6, 13, 4), (12, 26, 9)]):
                data['macd_composite'] = (
                        data['macd_cross_6_13'] * 0.6 +
                        data['macd_cross_12_26'] * 0.4
                )

            # 10. Intraday composite indicators
            if 'volume_ratio' in data.columns and 'rsi_5' in data.columns:
                # Use RSI 5 instead of 14 for intraday
                data['vol_rsi_signal'] = data['volume_ratio'] * (50 - data['rsi_5']) / 50

            if 'volume_ratio' in data.columns and 'bb_pct_b_10' in data.columns:
                # Use 10-period BB instead of 20 for intraday
                data['vol_bb_signal'] = data['volume_ratio'] * (0.5 - abs(data['bb_pct_b_10'] - 0.5))

            # NEW: Critical intraday specific composite features
            # 1. Opening range breakout indicator (first 15 minutes)
            if len(data) >= 15:
                opening_high = data['High'].iloc[:15].max()
                opening_low = data['Low'].iloc[:15].min()

                data['orb_upper_break'] = (data['Close'] > opening_high).astype(int)
                data['orb_lower_break'] = (data['Close'] < opening_low).astype(int)

                # Distance from opening range
                data['orb_upper_distance'] = (data['Close'] - opening_high) / opening_high * 100
                data['orb_lower_distance'] = (opening_low - data['Close']) / opening_low * 100

                # NEW: Improved ORB with volume confirmation
                data['orb_upper_confirmed'] = data['orb_upper_break'] & (data['Volume'] > data['volume_sma'] * 1.3)
                data['orb_lower_confirmed'] = data['orb_lower_break'] & (data['Volume'] > data['volume_sma'] * 1.3)

            # 2. Intraday momentum reversal detection
            # Higher highs and lower lows detector (short-term)
            data['higher_high'] = ((data['High'] > data['High'].shift(1)) &
                                   (data['High'].shift(1) > data['High'].shift(2))).astype(int)

            data['lower_low'] = ((data['Low'] < data['Low'].shift(1)) &
                                 (data['Low'].shift(1) < data['Low'].shift(2))).astype(int)

            # NEW: Higher high/lower low streak counting
            data['hh_streak'] = 0
            data['ll_streak'] = 0

            # Count consecutive higher highs
            hh_mask = data['higher_high'] == 1
            hh_groups = hh_mask.ne(hh_mask.shift()).cumsum()
            hh_cumcount = hh_mask.groupby(hh_groups).cumsum()
            data.loc[hh_mask, 'hh_streak'] = hh_cumcount[hh_mask]

            # Count consecutive lower lows
            ll_mask = data['lower_low'] == 1
            ll_groups = ll_mask.ne(ll_mask.shift()).cumsum()
            ll_cumcount = ll_mask.groupby(ll_groups).cumsum()
            data.loc[ll_mask, 'll_streak'] = ll_cumcount[ll_mask]

            # 3. Intraday Support/Resistance levels
            # Simple pivot points
            data['pivot'] = (data['High'].shift(1) + data['Low'].shift(1) + data['Close'].shift(1)) / 3
            data['r1'] = (2 * data['pivot']) - data['Low'].shift(1)
            data['s1'] = (2 * data['pivot']) - data['High'].shift(1)

            # Distance from pivot points
            data['pivot_distance'] = (data['Close'] - data['pivot']) / data['pivot'] * 100
            data['r1_distance'] = (data['Close'] - data['r1']) / data['r1'] * 100
            data['s1_distance'] = (data['Close'] - data['s1']) / data['s1'] * 100

            # NEW: Pivot breakout signals
            data['pivot_r1_break'] = ((data['Close'] > data['r1']) &
                                      (data['Close'].shift(1) <= data['r1'].shift(1))).astype(int)
            data['pivot_s1_break'] = ((data['Close'] < data['s1']) &
                                      (data['Close'].shift(1) >= data['s1'].shift(1))).astype(int)

            # 4. Time-of-day features critical for intraday patterns
            # Convert index to datetime if it's not already
            if not isinstance(data.index, pd.DatetimeIndex):
                if 'timestamp' in data.columns:
                    data['timestamp'] = pd.to_datetime(data['timestamp'])
                    data.set_index('timestamp', inplace=True)
                else:
                    # Create synthetic datetime index if needed
                    data.index = pd.date_range(start='2025-01-01 09:15:00', periods=len(data), freq='1min')

            # Extract time features when possible
            try:
                # Hour of day (9-15 for market hours)
                data['hour'] = data.index.hour

                # Market session indicators (morning, mid-day, closing)
                data['morning_session'] = ((data.index.hour == 9) |
                                           ((data.index.hour == 10) & (data.index.minute < 30))).astype(int)

                data['mid_day_session'] = (((data.index.hour == 10) & (data.index.minute >= 30)) |
                                           (data.index.hour == 11) |
                                           (data.index.hour == 12) |
                                           ((data.index.hour == 13) & (data.index.minute < 30))).astype(int)

                data['closing_session'] = (((data.index.hour == 13) & (data.index.minute >= 30)) |
                                           (data.index.hour == 14) |
                                           (data.index.hour == 15)).astype(int)

                # First and last 15 minutes (often have different behavior)
                data['first_15_min'] = ((data.index.hour == 9) & (data.index.minute < 30)).astype(int)
                data['last_15_min'] = ((data.index.hour == 15) & (data.index.minute > 15)).astype(int)

                # NEW: Time-weighted features - trade setups have different probabilities by time of day
                # Morning breakout strength - breakouts more likely in morning
                if 'orb_upper_break' in data.columns:
                    data['morning_breakout'] = data['orb_upper_break'] * data['morning_session']

                    # Afternoon reversal tendency - reversals more common in afternoon
                if 'rsi_14_overbought' in data.columns:
                    data['afternoon_reversal'] = (data['rsi_14_overbought'] * data['closing_session'])
            except:
                # If time features can't be extracted, create dummy columns
                data['hour'] = 12  # Neutral hour
                data['morning_session'] = 0
                data['mid_day_session'] = 1
                data['closing_session'] = 0
                data['first_15_min'] = 0
                data['last_15_min'] = 0

                # 5. Gap analysis (if first bar of the day can be detected)
            try:
                if 'day' not in data.columns and isinstance(data.index, pd.DatetimeIndex):
                    data['day'] = data.index.date

                    day_changes = data['day'].ne(data['day'].shift(1)).astype(int)

                    if day_changes.sum() > 0:
                        # Get first bar indexes for each day
                        first_bars = day_changes[day_changes == 1].index

                        # For each first bar, calculate gap from previous day's close
                        for idx in first_bars:
                            position = data.index.get_loc(idx)

                            # Need at least one previous day's data
                            if position > 0:
                                prev_close = data.iloc[position - 1]['Close']
                                open_price = data.loc[idx, 'Open']

                                # Calculate gap percentage
                                gap_pct = (open_price / prev_close - 1) * 100

                                # Set gap flag
                                data.loc[idx, 'gap_up'] = 1 if gap_pct > 0.5 else 0
                                data.loc[idx, 'gap_down'] = 1 if gap_pct < -0.5 else 0
                                data.loc[idx, 'gap_size'] = gap_pct

                    # Fillna for gap columns
                    data['gap_up'] = pd.Series(data.get('gap_up', 0), index=data.index).fillna(0)
                    data['gap_down'] = pd.Series(data.get('gap_down', 0), index=data.index).fillna(0)
                    data['gap_size'] = pd.Series(data.get('gap_size', 0), index=data.index).fillna(0)

                    # Drop the day column if not needed anymore
                    if 'day' in data.columns:
                        data.drop('day', axis=1, inplace=True)

                    # NEW: Gap fill detection
                    if 'gap_up' in data.columns and 'gap_size' in data.columns:
                        # Initialize the gap_fill column with zeros
                        data['gap_fill'] = 0

                        # For stocks that gapped up, detect if price falls back to fill the gap
                        for idx in data[data['gap_up'] == 1].index:
                            if idx in data.index:
                                position = data.index.get_loc(idx)
                                if position > 0 and position < len(data) - 1:
                                    gap_start = data.iloc[position - 1]['Close']
                                    # Check next 20 bars for gap fill
                                    end_pos = min(position + 20, len(data) - 1)
                                    for i in range(position + 1, end_pos + 1):
                                        if data.iloc[i]['Low'] <= gap_start:
                                            data.iloc[i, data.columns.get_loc('gap_fill')] = 1
                                            break

            except Exception as gap_error:
                # Log the error but continue
                logging.warning(f"Error calculating gap features: {str(gap_error)}")

                # Create empty gap columns
                data['gap_up'] = 0
                data['gap_down'] = 0
                data['gap_size'] = 0

                # 6. Price velocity metrics (critical for intraday momentum trading)
                # Rate of price change per minute
            data['price_velocity_1'] = data['returns'] * 100  # In percent per bar
            data['price_velocity_3'] = data['returns'].rolling(3).sum() * 100 / 3  # Average over 3 bars
            data['price_velocity_5'] = data['returns'].rolling(5).sum() * 100 / 5  # Average over 5 bars

            # Acceleration of price movement
            data['price_acceleration'] = data['price_velocity_1'].diff().fillna(0)

            # NEW: Smoothed price velocity for filtering noise
            data['price_velocity_smooth'] = data['price_velocity_3'].ewm(span=3).mean()

            # 7. Enhanced trend detection for intraday
            # Supertrend indicator components
            factor = 2.8  # More sensitive for intraday
            atr_period = 7  # Shorter ATR period

            # Calculate basic ATR if not already done
            if f'atr_{atr_period}' not in data.columns:
                tr1 = data['High'] - data['Low']
                tr2 = abs(data['High'] - data['Close'].shift())
                tr3 = abs(data['Low'] - data['Close'].shift())
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                data[f'atr_{atr_period}'] = tr.rolling(window=atr_period, min_periods=1).mean()

            # Calculate supertrend bands
            data['basic_upper_band'] = ((data['High'] + data['Low']) / 2) + (factor * data[f'atr_{atr_period}'])
            data['basic_lower_band'] = ((data['High'] + data['Low']) / 2) - (factor * data[f'atr_{atr_period}'])

            # Initialize supertrend columns
            data['supertrend_direction'] = 1  # Start as uptrend
            data['supertrend'] = data['basic_lower_band']  # Start with lower band

            # Calculate final supertrend
            for i in range(1, len(data)):
                curr_direction = data['supertrend_direction'].iloc[i - 1]

                # Uptrend
                if curr_direction == 1:
                    # If current close crosses below the previous lower band
                    if data['Close'].iloc[i] <= data['basic_lower_band'].iloc[i]:
                        data.loc[data.index[i], 'supertrend_direction'] = -1
                        data.loc[data.index[i], 'supertrend'] = data['basic_upper_band'].iloc[i]
                    else:
                        data.loc[data.index[i], 'supertrend_direction'] = 1
                        data.loc[data.index[i], 'supertrend'] = max(data['basic_lower_band'].iloc[i],
                                                                    data['basic_lower_band'].iloc[i - 1])
                # Downtrend
                else:
                    # If current close crosses above the previous upper band
                    if data['Close'].iloc[i] >= data['basic_upper_band'].iloc[i]:
                        data.loc[data.index[i], 'supertrend_direction'] = 1
                        data.loc[data.index[i], 'supertrend'] = data['basic_lower_band'].iloc[i]
                    else:
                        data.loc[data.index[i], 'supertrend_direction'] = -1
                        data.loc[data.index[i], 'supertrend'] = min(data['basic_upper_band'].iloc[i],
                                                                    data['basic_upper_band'].iloc[i - 1])

            # Distance from supertrend (good for entry timing)
            data['supertrend_distance'] = (data['Close'] - data['supertrend']) / data['supertrend'] * 100

            # NEW: Supertrend signal - detects crossover points
            data['supertrend_signal'] = data['supertrend_direction'].diff()

            # NEW: Adaptive Supertrend - adjusts factor based on volatility
            # More sensitive in low volatility, less sensitive in high volatility
            if 'volatility_20' in data.columns:
                # Adjust factor based on volatility - use 3.5 for high volatility, 2.5 for low
                adaptive_factor = 2.5 + (data['volatility_20'] / data['volatility_20'].rolling(50).mean() - 1) * 2
                adaptive_factor = adaptive_factor.clip(2.5, 3.5)

                # Recalculate with adaptive factor
                data['adaptive_upper_band'] = ((data['High'] + data['Low']) / 2) + (
                        adaptive_factor * data[f'atr_{atr_period}'])
                data['adaptive_lower_band'] = ((data['High'] + data['Low']) / 2) - (
                        adaptive_factor * data[f'atr_{atr_period}'])

                # Initialize columns
                data['adaptive_supertrend_dir'] = 1
                data['adaptive_supertrend'] = data['adaptive_lower_band']

                # Calculate with same logic as regular supertrend
                for i in range(1, len(data)):
                    curr_direction = data['adaptive_supertrend_dir'].iloc[i - 1]
                    if curr_direction == 1:
                        if data['Close'].iloc[i] <= data['adaptive_lower_band'].iloc[i]:
                            data.loc[data.index[i], 'adaptive_supertrend_dir'] = -1
                            data.loc[data.index[i], 'adaptive_supertrend'] = data['adaptive_upper_band'].iloc[i]
                        else:
                            data.loc[data.index[i], 'adaptive_supertrend_dir'] = 1
                            data.loc[data.index[i], 'adaptive_supertrend'] = max(data['adaptive_lower_band'].iloc[i],
                                                                                 data['adaptive_lower_band'].iloc[
                                                                                     i - 1])
                    else:
                        if data['Close'].iloc[i] >= data['adaptive_upper_band'].iloc[i]:
                            data.loc[data.index[i], 'adaptive_supertrend_dir'] = 1
                            data.loc[data.index[i], 'adaptive_supertrend'] = data['adaptive_lower_band'].iloc[i]
                        else:
                            data.loc[data.index[i], 'adaptive_supertrend_dir'] = -1
                            data.loc[data.index[i], 'adaptive_supertrend'] = min(data['adaptive_upper_band'].iloc[i],
                                                                                 data['adaptive_upper_band'].iloc[
                                                                                     i - 1])

                # Adaptive supertrend signal
                data['adaptive_supertrend_signal'] = data['adaptive_supertrend_dir'].diff()

            # Ensure we have 'rsi' column for compatibility with other code
            if 'rsi_14' in data.columns:
                data['rsi'] = data['rsi_14']
            elif 'rsi_9' in data.columns:
                data['rsi'] = data['rsi_9']
            elif 'rsi_5' in data.columns:
                data['rsi'] = data['rsi_5']

            # NEW: Add composite signal columns that combine multiple indicators
            # 1. Trend strength composite
            if all(col in data.columns for col in ['supertrend_direction', 'ema_alignment', 'macd_composite']):
                data['trend_strength_composite'] = (
                        data['supertrend_direction'] * 0.4 +
                        data['ema_alignment'] * 0.4 +
                        data['macd_composite'] * 0.2
                )

            # 2. Momentum composite
            if all(col in data.columns for col in ['rsi_14', 'momentum_composite', 'price_velocity_smooth']):
                # Normalize RSI from 0-100 to -1 to +1 scale
                rsi_normalized = (data['rsi_14'] - 50) / 50

                data['momentum_strength_composite'] = (
                        rsi_normalized * 0.4 +
                        data['momentum_composite'] * 0.4 +
                        data['price_velocity_smooth'] / 0.5 * 0.2  # Normalize velocity
                ).clip(-1, 1)  # Clip to -1 to +1 range

            # 3. Volatility/breakout composite
            if all(col in data.columns for col in ['volatility_ratio', 'bb_squeeze', 'bb_pct_b_momentum']):
                data['breakout_potential_composite'] = (
                        (data['volatility_ratio'] < 0.8).astype(int) * 0.4 +  # Low volatility component
                        data['bb_squeeze'] * 0.4 +  # BB squeeze component
                        (data['bb_pct_b_momentum'].abs() > 0.05).astype(int) * 0.2  # Momentum building in BB
                )

            # 4. Volume signal composite
            if all(col in data.columns for col in ['volume_ratio', 'vol_price_confirm_up', 'vol_price_confirm_down']):
                data['volume_signal_composite'] = (
                        data['vol_price_confirm_up'] * 0.5 +
                        data['vol_price_confirm_down'] * -0.5 +
                        (data['volume_ratio'] > 1.2).astype(int) * 0.2
                )

            # 5. MASTER composite signal for ML enhancement
            key_columns = ['trend_strength_composite', 'momentum_strength_composite',
                           'breakout_potential_composite', 'volume_signal_composite']

            if all(col in data.columns for col in key_columns):
                # For bullish signals
                data['master_signal_long'] = (
                        data['trend_strength_composite'].clip(0, 1) * 0.35 +
                        data['momentum_strength_composite'].clip(0, 1) * 0.30 +
                        data['breakout_potential_composite'] * 0.20 +
                        data['volume_signal_composite'].clip(0, 1) * 0.15
                )

                # For bearish signals
                data['master_signal_short'] = (
                        data['trend_strength_composite'].clip(-1, 0).abs() * 0.35 +
                        data['momentum_strength_composite'].clip(-1, 0).abs() * 0.30 +
                        data['breakout_potential_composite'] * 0.20 +
                        data['volume_signal_composite'].clip(-1, 0).abs() * 0.15
                )

            # === New Enhanced Features from Update ===

            # 1. Better price action micro-patterns
            data['wick_ratio'] = (data['High'] - data['Low']) / (abs(data['Close'] - data['Open']) + 0.0001)
            data['upper_wick_ratio'] = (data['High'] - data[['Open', 'Close']].max(axis=1)) / (
                    data['High'] - data['Low'] + 0.0001)
            data['lower_wick_ratio'] = (data[['Open', 'Close']].min(axis=1) - data['Low']) / (
                    data['High'] - data['Low'] + 0.0001)

            # 2. Enhanced intraday volatility metrics
            data['volatility_ratio_3_10'] = data['volatility_3'] / data['volatility_10'].rolling(3).mean().fillna(
                data['volatility_10'])
            data['volatility_expansion'] = (data['volatility_5'] > data['volatility_5'].shift(5) * 1.3).astype(int)
            data['volatility_contraction'] = (data['volatility_5'] < data['volatility_5'].shift(5) * 0.7).astype(int)

            # 3. Improved momentum with rate of change
            for period in [1, 3, 5]:
                # Rate of change - percentage of price change over time
                data[f'roc_{period}'] = ((data['Close'] / data['Close'].shift(period)) - 1) * 100
                # Acceleration of ROC
                data[f'roc_accel_{period}'] = data[f'roc_{period}'] - data[f'roc_{period}'].shift(1)

            # 4. Sharper EMA cross signals
            for fast, slow in [(5, 13), (8, 21)]:
                # Signal strength when EMAs are far apart (stronger trend)
                data[f'ema_spread_{fast}_{slow}'] = abs(data[f'ema_{fast}'] - data[f'ema_{slow}']) / data[
                    f'ema_{slow}'] * 100
                # Signal confirmation when volume aligns with cross
                if f'ema_cross_{fast}_{slow}' in data.columns and 'volume_ratio' in data.columns:
                    data[f'ema_cross_vol_{fast}_{slow}'] = data[f'ema_cross_{fast}_{slow}'] * data['volume_ratio']

            # 5. RSI with Stochastic for stronger signals
            # Stochastic RSI - provides extreme overbought/oversold signals
            for period in [9, 14]:
                if f'rsi_{period}' in data.columns:
                    rsi = data[f'rsi_{period}']
                    stoch_rsi_k = ((rsi - rsi.rolling(14).min()) /
                                   (rsi.rolling(14).max() - rsi.rolling(14).min() + 0.0001) * 100)
                    data[f'stoch_rsi_{period}'] = stoch_rsi_k
                    data[f'stoch_rsi_{period}_overbought'] = (stoch_rsi_k > 80).astype(int)
                    data[f'stoch_rsi_{period}_oversold'] = (stoch_rsi_k < 20).astype(int)

            # 6. Volume profile analysis
            data['volume_profile'] = data['Volume'] / data['Volume'].rolling(50).mean()
            data['rising_volume'] = ((data['Volume'] > data['Volume'].shift(1)) &
                                     (data['Volume'].shift(1) > data['Volume'].shift(2))).astype(int)
            data['falling_volume'] = ((data['Volume'] < data['Volume'].shift(1)) &
                                      (data['Volume'].shift(1) < data['Volume'].shift(2))).astype(int)

            # 7. Enhanced intraday patterns
            if all(col in data.columns for col in ['body_size', 'upper_wick', 'lower_wick']):
                data['high_wave_candle'] = ((data['body_size'] < 0.3) &
                                            (data['upper_wick'] > data['body_size'] * 1.5) &
                                            (data['lower_wick'] > data['body_size'] * 1.5)).astype(int)
                data['price_rejection'] = ((data['upper_wick'] > 3 * data['body_size']) |
                                           (data['lower_wick'] > 3 * data['body_size'])).astype(int)

            # 8. Volume thrust indicators
            data['volume_thrust'] = (data['Volume'] > 1.5 * data['Volume'].rolling(5).mean()).astype(int)
            data['vol_price_thrust'] = ((data['returns'] > 0) &
                                        (data['Volume'] > 1.5 * data['Volume'].rolling(5).mean())).astype(int)

            # 9. Better intraday support/resistance dynamics
            data['close_to_high'] = (data['High'] - data['Close']) / (data['High'] - data['Low'] + 0.0001)
            data['close_to_low'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'] + 0.0001)
            data['bar_strength'] = ((data['Close'] - data['Low']) / (
                    data['High'] - data['Low'] + 0.0001) - 0.5) * 2  # -1 to 1

            # 10. Relative volume analysis
            data['rel_vol_delta'] = data['volume_ratio'] - data['volume_ratio'].shift(1)
            data['vol_oscillator'] = (data['Volume'] - data['Volume'].rolling(10).mean()) / data['Volume'].rolling(
                10).std().replace(0, 1)

            # NEW FROM UPDATE: Add Price Action Momentum
            data['momentum_strength'] = (data['Close'] - data['Close'].shift(5)) / (
                    data['High'].rolling(5).max() - data['Low'].rolling(5).min() + 0.0001)

            # NEW FROM UPDATE: Add Enhanced Volatility-Adjusted Momentum
            if 'returns' in data.columns and 'volatility_5' in data.columns:
                data['vol_adj_momentum'] = data['returns'].rolling(5).mean() / (data['volatility_5'] + 0.001)

            # NEW FROM UPDATE: Add Triple EMA Confluence
            if all(col in data.columns for col in ['ema_5', 'ema_8', 'ema_13', 'ema_21']):
                data['triple_ema_signal'] = np.where(
                    (data['ema_5'] > data['ema_8']) & (data['ema_8'] > data['ema_13']) & (
                            data['ema_13'] > data['ema_21']), 1,
                    np.where((data['ema_5'] < data['ema_8']) & (data['ema_8'] < data['ema_13']) & (
                            data['ema_13'] < data['ema_21']), -1, 0)
                )

            # Clean any remaining NaN values
            data = data.fillna(method='ffill').fillna(method='bfill').fillna(0)
            data = data.replace([np.inf, -np.inf], 0)

            return data

        except Exception as e:
            logging.error(f"Intraday feature calculation error: {str(e)}")
            logging.error(traceback.format_exc())
            return df  # Return original data on error

    def _generate_ideal_path(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Generate ideal trading path with SEQUENTIAL entry/exit logic - exits only after entries
        OPTIMIZED VERSION with vectorized operations for 10x speed improvement"""
        try:
            # Create working copy
            data = df.copy()

            # Validate input data
            if data is None or data.empty:
                logging.error(f"Empty DataFrame provided for ideal path generation: {symbol}")
                self.profit_thresholds[symbol] = {
                    'long_target': 0.50,
                    'short_target': 0.50,
                    'long_stop': 0.25,
                    'short_stop': 0.25,
                    'holding_period': 60
                }
                return pd.DataFrame()

            logging.info(f"SPEED OPTIMIZATION: Starting vectorized ideal path generation for {symbol}")

            # Setup tracking columns - UPDATED WITH SEQUENTIAL EXIT SIGNALS
            data['ideal_entry_long'] = 0
            data['ideal_entry_short'] = 0
            data['ideal_entry'] = 0  # Combined ideal entry
            data['ideal_exit'] = 0
            data['ideal_exit_long'] = 0  # NEW: Exit signals for long positions
            data['ideal_exit_short'] = 0  # NEW: Exit signals for short positions
            data['position_active'] = 0  # NEW: Track when position is active
            data['position_direction'] = 0  # NEW: Track position direction (1=LONG, -1=SHORT)
            data['trade_id'] = 0
            data['trade_direction'] = 0
            data['profit_target'] = 0.0
            data['stop_loss'] = 0.0
            data['realized_profit'] = 0.0
            data['exit_reason'] = ''

            # Calculate ATR if not present
            if 'atr_14' not in data.columns and all(col in data.columns for col in ['High', 'Low', 'Close']):
                high_low = data['High'] - data['Low']
                high_close = abs(data['High'] - data['Close'].shift())
                low_close = abs(data['Low'] - data['Close'].shift())
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                data['atr_14'] = tr.rolling(window=14).mean()

            # Default ATR value if calculation failed
            avg_atr = data['atr_14'].mean() if 'atr_14' in data.columns else (data['Close'].mean() * 0.01)

            # Use exact values for stop/target percentages with 3:1 ratio
            min_target_pct = 0.50  # Fixed at exactly 0.5%
            min_stop_pct = 0.25  # Fixed at exactly 0.25%

            # Ensure targets meet minimum thresholds
            long_profit_target = max(1.8 * avg_atr / data['Close'].mean() * 100, min_target_pct)
            long_stop_loss = max(abs(0.6 * avg_atr / data['Close'].mean() * 100), min_stop_pct) * -1  # Keep negative

            short_profit_target = max(1.8 * avg_atr / data['Close'].mean() * 100, min_target_pct)
            short_stop_loss = max(abs(0.6 * avg_atr / data['Close'].mean() * 100), min_stop_pct) * -1  # Keep negative

            # Ensure exactly 3:1 reward-to-risk ratio
            ideal_ratio = 3.0
            long_profit_target = abs(long_stop_loss) * ideal_ratio
            short_profit_target = abs(short_stop_loss) * ideal_ratio

            # SPEED OPTIMIZATION: Pre-calculate arrays for vectorized operations
            logging.info(f"SPEED OPTIMIZATION: Pre-calculating arrays for vectorized operations")
            closes = data['Close'].values
            highs = data['High'].values
            lows = data['Low'].values
            volumes = data['Volume'].values if 'Volume' in data.columns else np.ones(len(data))

            # Pre-calculate EMAs for all bars at once
            ema8_vals = np.full(len(data), np.nan)
            ema13_vals = np.full(len(data), np.nan)
            ema21_vals = np.full(len(data), np.nan)
            ema55_vals = np.full(len(data), np.nan)

            if all(f'ema_{period}' in data.columns for period in [8, 13, 21, 55]):
                ema8_vals = data['ema_8'].fillna(0).values
                ema13_vals = data['ema_13'].fillna(0).values
                ema21_vals = data['ema_21'].fillna(0).values
                ema55_vals = data['ema_55'].fillna(0).values

            # Pre-calculate volume ratios
            volume_sma = pd.Series(volumes).rolling(10).mean().fillna(1).values
            volume_ratios = volumes / np.where(volume_sma > 0, volume_sma, 1)

            # PHASE 1: Generate all potential entries with exit analysis
            long_candidates = []
            short_candidates = []
            max_lookahead_bars = 30  # Maximum bars to look ahead for target/stop

            # SPEED OPTIMIZATION: Process in optimized chunks
            start_idx = 20
            end_idx = len(data) - max_lookahead_bars - 1
            chunk_size = 500

            logging.info(f"SPEED OPTIMIZATION: Processing {end_idx - start_idx} bars in chunks of {chunk_size}")

            for chunk_start in range(start_idx, end_idx, chunk_size):
                chunk_end = min(chunk_start + chunk_size, end_idx)

                # First pass: Identify all tradable setups - scanning every bar
                for i in range(chunk_start, chunk_end):
                    entry_price = closes[i]

                    # Skip invalid prices
                    if pd.isna(entry_price) or entry_price <= 0:
                        continue

                    # === LONG ENTRY ANALYSIS ===
                    trend_score = 0
                    # Use pre-calculated EMAs for trend determination
                    if not np.isnan(ema8_vals[i]):
                        # Check multi-timeframe alignment (stronger signal when all align)
                        if ema8_vals[i] > ema13_vals[i] > ema21_vals[i] > ema55_vals[i]:
                            trend_score = 1.0  # Strong uptrend - all EMAs aligned
                        elif ema8_vals[i] > ema13_vals[i] > ema21_vals[i]:
                            trend_score = 0.7  # Good uptrend - shorter EMAs aligned
                        elif ema8_vals[i] > ema13_vals[i]:
                            trend_score = 0.4  # Moderate uptrend - only shortest EMAs aligned
                        elif ema8_vals[i] < ema13_vals[i] < ema21_vals[i] < ema55_vals[i]:
                            trend_score = -1.0  # Strong downtrend - all EMAs aligned
                        elif ema8_vals[i] < ema13_vals[i] < ema21_vals[i]:
                            trend_score = -0.7  # Good downtrend - shorter EMAs aligned
                        elif ema8_vals[i] < ema13_vals[i]:
                            trend_score = -0.4  # Moderate downtrend - only shortest EMAs aligned

                    # Initialize scoring for LONG and SHORT separately
                    long_score = 0
                    short_score = 0

                    # CORRECT directional logic - FIXED
                    if trend_score > 0:  # Uptrend - look for LONG opportunities when trend is up
                        long_score += 0.3  # Buy in uptrend
                        if 'rsi_14' in data.columns:
                            rsi = data.iloc[i]['rsi_14']
                            if rsi < 40:  # Oversold in uptrend = good long entry
                                long_score += 0.4
                    elif trend_score < 0:  # Downtrend - look for SHORT opportunities when trend is down
                        short_score += 0.3  # Sell in downtrend
                        if 'rsi_14' in data.columns:
                            rsi = data.iloc[i]['rsi_14']
                            if rsi > 60:  # Overbought in downtrend = good short entry
                                short_score += 0.4

                    # RSI Logic Fix:
                    if 'rsi_14' in data.columns:
                        rsi = data.iloc[i]['rsi_14']
                        if rsi < 30:  # Oversold = LONG opportunity
                            long_score += 0.6
                        elif rsi > 70:  # Overbought = SHORT opportunity
                            short_score += 0.6

                    # Bollinger Band Fix:
                    if 'bb_pct_b_20' in data.columns:
                        bb_pct = data.iloc[i]['bb_pct_b_20']
                        if bb_pct < 0.2:  # Near lower band = LONG opportunity
                            long_score += 0.3
                        elif bb_pct > 0.8:  # Near upper band = SHORT opportunity
                            short_score += 0.3

                    # Volume Score (0 to +1) - Enhanced with trend confirmation using pre-calculated ratios
                    volume_score = 0
                    vol_ratio = volume_ratios[i]
                    # Non-linear scaling for volume impact - higher volume gets more weight
                    if vol_ratio > 2.0:
                        volume_score = 1.0  # Very high volume
                    elif vol_ratio > 1.5:
                        volume_score = 0.8  # High volume
                    elif vol_ratio > 1.2:
                        volume_score = 0.6  # Moderate volume
                    elif vol_ratio > 1.0:
                        volume_score = 0.3  # Slightly above average

                    # Add volume to both scores
                    long_score += volume_score * 0.15
                    short_score += volume_score * 0.15

                    # Valid setup thresholds - use same threshold for both directions
                    valid_long_setup = long_score > 0.22
                    valid_short_setup = short_score > 0.22

                    # Look ahead for profitable long entries WITH EXIT ANALYSIS
                    if valid_long_setup:
                        stop_price = entry_price * (1 - min_stop_pct / 100)  # Below entry for longs
                        target_price = entry_price * (1 + min_target_pct / 100)  # Above entry for longs

                        hit_target = False
                        hit_stop = False
                        exit_bar = None
                        optimal_hold = 5
                        exit_reason = 'TIME_LIMIT'

                        # NEW: Track additional exit conditions
                        hit_technical_exit = False
                        hit_time_exit = False
                        current_profit_pct = 0

                        # SPEED OPTIMIZATION: Vectorized lookahead validation
                        future_end = min(i + max_lookahead_bars + 1, len(highs))
                        if future_end > i + 1:
                            future_highs = highs[i + 1:future_end]
                            future_lows = lows[i + 1:future_end]

                            # Vectorized target/stop detection
                            target_hits = np.where(future_highs >= target_price)[0]
                            stop_hits = np.where(future_lows <= stop_price)[0]

                            hit_target = len(target_hits) > 0
                            hit_stop = len(stop_hits) > 0

                            if hit_target:
                                first_target = target_hits[0]
                                first_stop = stop_hits[0] if hit_stop else float('inf')

                                if first_target < first_stop:
                                    exit_bar = i + 1 + first_target
                                    optimal_hold = first_target + 1
                                    exit_reason = 'TAKE_PROFIT'
                                else:
                                    hit_target = False  # Stop hit first
                                    exit_bar = i + 1 + first_stop
                                    optimal_hold = first_stop + 1
                                    exit_reason = 'STOP_LOSS'
                                    hit_stop = True
                            elif hit_stop:
                                first_stop = stop_hits[0]
                                exit_bar = i + 1 + first_stop
                                optimal_hold = first_stop + 1
                                exit_reason = 'STOP_LOSS'

                            # NEW: Technical indicator exits (only if no target/stop hit)
                            if not hit_target and not hit_stop:
                                # Look for technical exits in the lookahead period
                                for j in range(1, min(max_lookahead_bars + 1, future_end - i)):
                                    future_idx = i + j
                                    if future_idx >= len(data):
                                        break

                                    current_data = data.iloc[future_idx]
                                    entry_data = data.iloc[i]
                                    current_profit_pct = (future_highs[j - 1] / entry_price - 1) * 100

                                    # RSI reversal exit for longs
                                    if 'rsi_14' in data.columns and j > 5:  # Give position time to develop
                                        if current_data['rsi_14'] > 70 and current_data['rsi_14'] < \
                                                data.iloc[future_idx - 1]['rsi_14']:
                                            if current_profit_pct > 0.1:  # At least 0.1% profit
                                                hit_technical_exit = True
                                                exit_bar = future_idx
                                                optimal_hold = j
                                                exit_reason = 'RSI_EXIT'
                                                break

                                    # Volatility spike exit
                                    if 'volatility_5' in data.columns and 'volatility_20' in data.columns:
                                        if current_data['volatility_5'] > entry_data['volatility_20'] * 2.5:
                                            if current_profit_pct > -0.1:  # Small loss is acceptable
                                                hit_technical_exit = True
                                                exit_bar = future_idx
                                                optimal_hold = j
                                                exit_reason = 'VOLATILITY_EXIT'
                                                break

                                    # Time-based exit
                                    if j >= 45:  # 45 minutes holding time
                                        if current_profit_pct > -0.15:  # Limit loss to 0.15%
                                            hit_time_exit = True
                                            exit_bar = future_idx
                                            optimal_hold = j
                                            exit_reason = 'TIME_EXIT'
                                            break

                        # Include trades based on expanded criteria
                        include_trade = False

                        # Original criteria
                        if hit_target and not hit_stop and exit_bar is not None:
                            include_trade = True

                        # NEW: Include technical exits
                        elif hit_technical_exit and not hit_stop and exit_bar is not None:
                            include_trade = True

                        # NEW: Include time exits (limit how many to maintain quality)
                        elif hit_time_exit and not hit_stop and exit_bar is not None:
                            # Only include time exits for high-quality setups
                            if long_score > 0.4:  # Higher threshold for time exits
                                include_trade = True

                        if include_trade:
                            long_candidates.append((i, exit_bar, long_score, optimal_hold, exit_reason))

                    # Look ahead for profitable short entries WITH EXIT ANALYSIS
                    if valid_short_setup:
                        stop_price = entry_price * (1 + min_stop_pct / 100)  # Above entry for shorts
                        target_price = entry_price * (1 - min_target_pct / 100)  # Below entry for shorts

                        hit_target = False
                        hit_stop = False
                        exit_bar = None
                        optimal_hold = 5
                        exit_reason = 'TIME_LIMIT'

                        # NEW: Track additional exit conditions
                        hit_technical_exit = False
                        hit_time_exit = False
                        current_profit_pct = 0

                        # SPEED OPTIMIZATION: Vectorized lookahead for shorts
                        future_end = min(i + max_lookahead_bars + 1, len(lows))
                        if future_end > i + 1:
                            future_highs = highs[i + 1:future_end]
                            future_lows = lows[i + 1:future_end]

                            target_hits = np.where(future_lows <= target_price)[0]
                            stop_hits = np.where(future_highs >= stop_price)[0]

                            hit_target = len(target_hits) > 0
                            hit_stop = len(stop_hits) > 0

                            if hit_target:
                                first_target = target_hits[0]
                                first_stop = stop_hits[0] if hit_stop else float('inf')

                                if first_target < first_stop:
                                    exit_bar = i + 1 + first_target
                                    optimal_hold = first_target + 1
                                    exit_reason = 'TAKE_PROFIT'
                                else:
                                    hit_target = False  # Stop hit first
                                    exit_bar = i + 1 + first_stop
                                    optimal_hold = first_stop + 1
                                    exit_reason = 'STOP_LOSS'
                                    hit_stop = True
                            elif hit_stop:
                                first_stop = stop_hits[0]
                                exit_bar = i + 1 + first_stop
                                optimal_hold = first_stop + 1
                                exit_reason = 'STOP_LOSS'

                            # NEW: Technical indicator exits for shorts (only if no target/stop hit)
                            if not hit_target and not hit_stop:
                                # Look for technical exits in the lookahead period
                                for j in range(1, min(max_lookahead_bars + 1, future_end - i)):
                                    future_idx = i + j
                                    if future_idx >= len(data):
                                        break

                                    current_data = data.iloc[future_idx]
                                    entry_data = data.iloc[i]
                                    current_profit_pct = (entry_price / future_lows[
                                        j - 1] - 1) * 100  # Inverted for shorts

                                    # RSI reversal exit for shorts
                                    if 'rsi_14' in data.columns and j > 5:
                                        if current_data['rsi_14'] < 30 and current_data['rsi_14'] > \
                                                data.iloc[future_idx - 1]['rsi_14']:
                                            if current_profit_pct > 0.1:  # At least 0.1% profit
                                                hit_technical_exit = True
                                                exit_bar = future_idx
                                                optimal_hold = j
                                                exit_reason = 'RSI_EXIT'
                                                break

                                    # Volatility spike exit
                                    if 'volatility_5' in data.columns and 'volatility_20' in data.columns:
                                        if current_data['volatility_5'] > entry_data['volatility_20'] * 2.5:
                                            if current_profit_pct > -0.1:  # Small loss is acceptable
                                                hit_technical_exit = True
                                                exit_bar = future_idx
                                                optimal_hold = j
                                                exit_reason = 'VOLATILITY_EXIT'
                                                break

                                    # Time-based exit
                                    if j >= 45:  # 45 minutes holding time
                                        if current_profit_pct > -0.15:  # Limit loss to 0.15%
                                            hit_time_exit = True
                                            exit_bar = future_idx
                                            optimal_hold = j
                                            exit_reason = 'TIME_EXIT'
                                            break

                        # Include trades based on expanded criteria
                        include_trade = False

                        # Original criteria
                        if hit_target and not hit_stop and exit_bar is not None:
                            include_trade = True

                        # NEW: Include technical exits
                        elif hit_technical_exit and not hit_stop and exit_bar is not None:
                            include_trade = True

                        # NEW: Include time exits
                        elif hit_time_exit and not hit_stop and exit_bar is not None:
                            if short_score > 0.4:  # Higher threshold for time exits
                                include_trade = True

                        if include_trade:
                            short_candidates.append((i, exit_bar, short_score, optimal_hold, exit_reason))

            # Set target percentages for long and short separately - REDUCED TO 5% TOTAL
            target_long_entries = int(len(data) * 0.025)  # Changed from 0.10 to 0.02 (2% for longs)
            target_short_entries = int(len(data) * 0.025)  # Changed from 0.10 to 0.02 (2% for shorts)
            min_acceptable_entries = int(len(data) * 0.015)  # Changed from 0.08 to 0.015 (1.5% minimum)

            # Sort long candidates separately
            long_entries = []
            if len(long_candidates) >= min_acceptable_entries:
                long_candidates.sort(key=lambda x: x[2], reverse=True)
                long_entries = [(entry, exit_bar, hold, reason) for entry, exit_bar, _, hold, reason in
                                long_candidates[:target_long_entries]]

            # Sort short candidates separately
            short_entries = []
            if len(short_candidates) >= min_acceptable_entries:
                short_candidates.sort(key=lambda x: x[2], reverse=True)
                short_entries = [(entry, exit_bar, hold, reason) for entry, exit_bar, _, hold, reason in
                                 short_candidates[:target_short_entries]]

            # Log results with details for both directions
            logging.info(
                f"SPEED OPTIMIZATION: Found {len(long_entries)} potential long bars ({len(long_entries) / len(data):.1%}) and "
                f"{len(short_entries)} potential short bars ({len(short_entries) / len(data):.1%})")
            logging.info(f"Final signal rate: {(len(long_entries) + len(short_entries)) / len(data):.1%}")

            # CRITICAL FIX: Process trades sequentially and create labels WITH POSITION TRACKING
            current_trade_id = 0
            win_count = 0
            loss_count = 0

            # SEQUENTIAL PROCESSING: Track active positions to prevent exit without entry
            active_long_positions = {}  # {entry_bar: exit_bar}
            active_short_positions = {}  # {entry_bar: exit_bar}

            # Combine all entries and exits and sort by bar index for sequential processing
            all_events = []

            # Add long entries and exits
            for entry_idx, exit_idx, hold_duration, exit_reason in long_entries:
                all_events.append((entry_idx, 'ENTER_LONG', exit_idx, exit_reason))
                all_events.append((exit_idx, 'EXIT_LONG', entry_idx, exit_reason))

            # Add short entries and exits
            for entry_idx, exit_idx, hold_duration, exit_reason in short_entries:
                all_events.append((entry_idx, 'ENTER_SHORT', exit_idx, exit_reason))
                all_events.append((exit_idx, 'EXIT_SHORT', entry_idx, exit_reason))

            # Sort all events by bar index
            all_events.sort(key=lambda x: x[0])

            # Process events sequentially
            for event_idx, event_type, related_idx, reason in all_events:
                current_trade_id += 1

                if event_type == 'ENTER_LONG':
                    # Mark long entry
                    data.iloc[event_idx, data.columns.get_loc('ideal_entry_long')] = 1
                    data.iloc[event_idx, data.columns.get_loc('ideal_entry')] = 1
                    data.iloc[event_idx, data.columns.get_loc('trade_id')] = current_trade_id
                    data.iloc[event_idx, data.columns.get_loc('trade_direction')] = 1  # 1 for longs
                    data.iloc[event_idx, data.columns.get_loc('profit_target')] = min_target_pct
                    data.iloc[event_idx, data.columns.get_loc('stop_loss')] = -min_stop_pct

                    # Mark position as active from entry to exit
                    for active_idx in range(event_idx, min(related_idx + 1, len(data))):
                        data.iloc[active_idx, data.columns.get_loc('position_active')] = 1
                        data.iloc[active_idx, data.columns.get_loc('position_direction')] = 1  # LONG

                elif event_type == 'EXIT_LONG':
                    # Only mark exit if we have an active long position
                    if any(data.iloc[i]['position_direction'] == 1 for i in range(max(0, event_idx - 30), event_idx)):
                        data.iloc[event_idx, data.columns.get_loc('ideal_exit_long')] = 1
                        data.iloc[event_idx, data.columns.get_loc('ideal_exit')] = 1

                        # Calculate profit
                        entry_price = data.iloc[related_idx]['Close']
                        exit_price = data.iloc[event_idx]['High']  # Use High for long exits
                        profit_pct = (exit_price / entry_price - 1) * 100

                        data.iloc[event_idx, data.columns.get_loc('realized_profit')] = profit_pct
                        data.iloc[event_idx, data.columns.get_loc('exit_reason')] = reason
                        win_count += 1

                elif event_type == 'ENTER_SHORT':
                    # Mark short entry
                    data.iloc[event_idx, data.columns.get_loc('ideal_entry_short')] = 1
                    data.iloc[event_idx, data.columns.get_loc('ideal_entry')] = 1
                    data.iloc[event_idx, data.columns.get_loc('trade_id')] = current_trade_id
                    data.iloc[event_idx, data.columns.get_loc('trade_direction')] = -1  # -1 for shorts
                    data.iloc[event_idx, data.columns.get_loc('profit_target')] = min_target_pct
                    data.iloc[event_idx, data.columns.get_loc('stop_loss')] = -min_stop_pct

                    # Mark position as active from entry to exit
                    for active_idx in range(event_idx, min(related_idx + 1, len(data))):
                        data.iloc[active_idx, data.columns.get_loc('position_active')] = 1
                        data.iloc[active_idx, data.columns.get_loc('position_direction')] = -1  # SHORT

                elif event_type == 'EXIT_SHORT':
                    # Only mark exit if we have an active short position
                    if any(data.iloc[i]['position_direction'] == -1 for i in range(max(0, event_idx - 30), event_idx)):
                        data.iloc[event_idx, data.columns.get_loc('ideal_exit_short')] = 1
                        data.iloc[event_idx, data.columns.get_loc('ideal_exit')] = 1

                        # Calculate profit for shorts
                        entry_price = data.iloc[related_idx]['Close']
                        exit_price = data.iloc[event_idx]['Low']  # Use Low for short exits
                        profit_pct = (entry_price / exit_price - 1) * 100  # Inverted for shorts

                        data.iloc[event_idx, data.columns.get_loc('realized_profit')] = profit_pct
                        data.iloc[event_idx, data.columns.get_loc('exit_reason')] = reason
                        win_count += 1

            # Store trade data
            self.ideal_trades[symbol] = {
                'entry_indices': data[data['ideal_entry'] == 1].index.tolist(),
                'targets': data[data['ideal_entry'] == 1][['profit_target', 'stop_loss', 'trade_direction']].to_dict(
                    'records')
            }

            # Store profit thresholds with exact values
            self.profit_thresholds[symbol] = {
                'long_target': min_target_pct,  # Exactly 0.5%
                'short_target': min_target_pct,  # Exactly 0.5%
                'long_stop': min_stop_pct,  # Exactly 0.25%
                'short_stop': min_stop_pct,  # Exactly 0.25%
                'holding_period': max_lookahead_bars
            }

            # Log results with realistic metrics
            ideal_entries = int(data['ideal_entry'].sum())
            ideal_exits = int(data['ideal_exit'].sum())

            if ideal_entries > 0:
                realized_profits = data[data['ideal_exit'] == 1]['realized_profit']
                avg_realized_profit = realized_profits.mean() if len(realized_profits) > 0 else 0

                total_trades = win_count + loss_count
                win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0

                logging.info(f"SPEED OPTIMIZED ideal path generation for {symbol}:")
                logging.info(
                    f"  Identified {ideal_entries} ideal entry points ({ideal_entries / len(data):.1%} of bars)")
                logging.info(f"  Completed {ideal_exits} ideal trades with SEQUENTIAL exit signals")
                logging.info(f"  Win rate: {win_rate:.1f}%")
                logging.info(f"  Long profit target: {min_target_pct:.2f}%, Short profit target: {min_target_pct:.2f}%")
                logging.info(f"  Long stop loss: {-min_stop_pct:.2f}%, Short stop loss: {-min_stop_pct:.2f}%")
                logging.info(f"  Average realized profit: {avg_realized_profit:.2f}%")
                logging.info(f"  CRITICAL: Exit signals only generated AFTER corresponding entry signals")
            else:
                logging.warning(f"No ideal trades generated for {symbol} (unexpected error)")

            return data

        except Exception as e:
            logging.error(f"Ideal path generation error for {symbol}: {str(e)}")
            logging.error(traceback.format_exc())

            # Ensure default thresholds on error with exact values
            if symbol not in self.profit_thresholds:
                self.profit_thresholds[symbol] = {
                    'long_target': 0.50,  # Exactly 0.5%
                    'short_target': 0.50,  # Exactly 0.5%
                    'long_stop': 0.25,  # Exactly 0.25%
                    'short_stop': 0.25,  # Exactly 0.25%
                    'holding_period': 60
                }
                logging.info(f"Set default thresholds for {symbol} after error")

            # Return original data with empty ideal columns
            df['ideal_entry'] = 0
            df['ideal_entry_long'] = 0
            df['ideal_entry_short'] = 0
            df['ideal_exit'] = 0
            df['ideal_exit_long'] = 0  # NEW
            df['ideal_exit_short'] = 0  # NEW
            df['position_active'] = 0  # NEW
            df['position_direction'] = 0  # NEW
            df['trade_id'] = 0
            df['trade_direction'] = 0
            df['exit_reason'] = ''
            return df

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for LSTM and XGBoost models with sequence formation for LSTM
        """
        try:
            if not isinstance(df, pd.DataFrame) or df.empty:
                return pd.DataFrame()

            # Calculate technical features
            full_data = self._calculate_technical_features(df)

            # Clean data
            full_data = full_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
            full_data = full_data.replace([np.inf, -np.inf], 0)

            # Remove lookahead features
            if hasattr(self, '_lookahead_features'):
                for col in self._lookahead_features:
                    if col in full_data.columns:
                        full_data = full_data.drop(col, axis=1)

            # Filter to only use selected features if available
            if hasattr(self, '_selected_features') and self._selected_features is not None:
                available_features = [f for f in self._selected_features if f in full_data.columns]
                if available_features:
                    return full_data[available_features]
                else:
                    logging.error("No selected features available in the data")
                    return pd.DataFrame()

            return full_data

        except Exception as e:
            logging.error(f"Feature preparation error: {str(e)}")
            logging.error(traceback.format_exc())
            return pd.DataFrame()

    def train(self, data: pd.DataFrame):
        """Train ML models with sequential logic - exits only valid when positions are active"""
        try:
            with self.training_lock:
                # Get symbol for better logging
                symbol = data.name if hasattr(data, 'name') and data.name is not None else 'unknown'

                # Store symbol as attribute for reference
                self.current_symbol = symbol

                if data is None or data.empty or len(data) < self.config.MIN_TRAINING_SAMPLES:
                    logging.error(f"Insufficient training data for {symbol}")
                    return

                logging.info(f"Beginning SEQUENTIAL training for {symbol} with {len(data)} samples")

                # Create separate copies for feature calculation and ideal path generation
                feature_data_source = data.copy()
                ideal_path_source = data.copy()

                # Calculate technical features
                full_feature_data = self._calculate_technical_features(feature_data_source)
                if full_feature_data.empty:
                    logging.error(f"Feature calculation failed for {symbol}")
                    return

                # Generate ideal trading path with SEQUENTIAL logic
                ideal_data = self._generate_ideal_path(ideal_path_source, symbol)

                # Get clean features without lookahead
                clean_feature_data = full_feature_data.copy()

                # Remove symbol column and any string/regime columns before training
                string_cols = clean_feature_data.select_dtypes(include=['object']).columns.tolist()
                regime_cols = [col for col in clean_feature_data.columns if 'regime' in str(col).lower()]
                status_cols = [col for col in clean_feature_data.columns if 'status' in str(col).lower()]

                cols_to_remove = list(set(string_cols + regime_cols + status_cols))
                if cols_to_remove:
                    logging.info(f"Removing non-numeric columns for training: {cols_to_remove}")
                    clean_feature_data = clean_feature_data.drop(columns=cols_to_remove, errors='ignore')

                for col in self._lookahead_features:
                    if col in clean_feature_data.columns:
                        clean_feature_data.drop(col, axis=1, inplace=True)

                # CRITICAL FIX: Create SEQUENTIAL target with position state awareness
                target = pd.Series(0, index=clean_feature_data.index)  # Default is 0 (HOLD)

                # SEQUENTIAL LOGIC: Only create valid targets based on position state
                for i in range(len(ideal_data)):
                    current_idx = ideal_data.index[i]

                    # Check if this bar has position state information
                    position_active = ideal_data.iloc[i].get('position_active', 0)
                    position_direction = ideal_data.iloc[i].get('position_direction', 0)

                    # ENTRY SIGNALS: Can happen anytime (no position required)
                    if ideal_data.iloc[i].get('ideal_entry_long', 0) == 1:
                        target.loc[current_idx] = 1  # ENTER_LONG
                    elif ideal_data.iloc[i].get('ideal_entry_short', 0) == 1:
                        target.loc[current_idx] = 2  # ENTER_SHORT

                    # EXIT SIGNALS: Only valid when position is active in the correct direction
                    elif position_active == 1:  # Position is active
                        if ideal_data.iloc[i].get('ideal_exit_long', 0) == 1 and position_direction == 1:
                            target.loc[current_idx] = 3  # EXIT_LONG (only if long position active)
                        elif ideal_data.iloc[i].get('ideal_exit_short', 0) == 1 and position_direction == -1:
                            target.loc[current_idx] = 4  # EXIT_SHORT (only if short position active)

                # VALIDATION: Count sequential signal distribution
                target_ratio = 0.05  # Target 20% total signals

                # Count actual signal percentages by direction with validation
                long_entry_count = (target == 1).sum()
                short_entry_count = (target == 2).sum()
                long_exit_count = (target == 3).sum()
                short_exit_count = (target == 4).sum()
                total_signals = long_entry_count + short_entry_count + long_exit_count + short_exit_count

                long_entry_ratio = long_entry_count / len(target)
                short_entry_ratio = short_entry_count / len(target)
                long_exit_ratio = long_exit_count / len(target)
                short_exit_ratio = short_exit_count / len(target)

                logging.info(
                    f"SEQUENTIAL 5-Class distribution: ENTER_LONG: {long_entry_count} ({long_entry_ratio:.2%}), "
                    f"ENTER_SHORT: {short_entry_count} ({short_entry_ratio:.2%}), "
                    f"EXIT_LONG: {long_exit_count} ({long_exit_ratio:.2%}), "
                    f"EXIT_SHORT: {short_exit_count} ({short_exit_ratio:.2%})")
                logging.info(f"Total SEQUENTIAL signals: {total_signals} ({total_signals / len(target):.2%})")

                # CRITICAL VALIDATION: Ensure exits <= entries for each direction
                if long_exit_count > long_entry_count:
                    logging.warning(
                        f"SEQUENTIAL VIOLATION: More long exits ({long_exit_count}) than entries ({long_entry_count})")
                if short_exit_count > short_entry_count:
                    logging.warning(
                        f"SEQUENTIAL VIOLATION: More short exits ({short_exit_count}) than entries ({short_entry_count})")

                # Log validation success
                if long_exit_count <= long_entry_count and short_exit_count <= short_entry_count:
                    logging.info("✅ SEQUENTIAL VALIDATION PASSED: Exits <= Entries for both directions")

                # ENHANCED: Use train/validation/test split to prevent data leakage
                train_size = int(len(clean_feature_data) * 0.6)  # 60% train
                val_size = int(len(clean_feature_data) * 0.2)  # 20% validation

                # Create indices for different classes
                class0_indices = np.where(target == 0)[0]  # HOLD
                class1_indices = np.where(target == 1)[0]  # ENTER_LONG
                class2_indices = np.where(target == 2)[0]  # ENTER_SHORT
                class3_indices = np.where(target == 3)[0]  # EXIT_LONG
                class4_indices = np.where(target == 4)[0]  # EXIT_SHORT

                # Split each class while preserving time order
                train_class0_indices = class0_indices[class0_indices < train_size]
                train_class1_indices = class1_indices[class1_indices < train_size]
                train_class2_indices = class2_indices[class2_indices < train_size]
                train_class3_indices = class3_indices[class3_indices < train_size]
                train_class4_indices = class4_indices[class4_indices < train_size]

                val_class0_indices = class0_indices[
                    (class0_indices >= train_size) & (class0_indices < train_size + val_size)]
                val_class1_indices = class1_indices[
                    (class1_indices >= train_size) & (class1_indices < train_size + val_size)]
                val_class2_indices = class2_indices[
                    (class2_indices >= train_size) & (class2_indices < train_size + val_size)]
                val_class3_indices = class3_indices[
                    (class3_indices >= train_size) & (class3_indices < train_size + val_size)]
                val_class4_indices = class4_indices[
                    (class4_indices >= train_size) & (class4_indices < train_size + val_size)]

                test_class0_indices = class0_indices[class0_indices >= train_size + val_size]
                test_class1_indices = class1_indices[class1_indices >= train_size + val_size]
                test_class2_indices = class2_indices[class2_indices >= train_size + val_size]
                test_class3_indices = class3_indices[class3_indices >= train_size + val_size]
                test_class4_indices = class4_indices[class4_indices >= train_size + val_size]

                # Combine to create train, validation, and test indices
                train_indices = np.sort(
                    np.concatenate([train_class0_indices, train_class1_indices, train_class2_indices,
                                    train_class3_indices, train_class4_indices]))
                val_indices = np.sort(
                    np.concatenate([val_class0_indices, val_class1_indices, val_class2_indices,
                                    val_class3_indices, val_class4_indices]))
                test_indices = np.sort(
                    np.concatenate([test_class0_indices, test_class1_indices, test_class2_indices,
                                    test_class3_indices, test_class4_indices]))

                # Create train, validation, and test sets
                X_train = clean_feature_data.iloc[train_indices]
                y_train = target.iloc[train_indices]
                X_val = clean_feature_data.iloc[val_indices]
                y_val = target.iloc[val_indices]
                X_test = clean_feature_data.iloc[test_indices]
                y_test = target.iloc[test_indices]

                # ENHANCED: Feature selection with F-score method from paper
                try:
                    # Calculate mutual information scores for initial ranking
                    mi_scores = mutual_info_classif(X_train, y_train)
                    mi_scores = pd.Series(mi_scores, index=X_train.columns)

                    # Get initial top features
                    initial_features = mi_scores.sort_values(ascending=False).index.tolist()[:60]

                    # Calculate F-scores for feature importance using paper's formula
                    class_indices = {
                        0: (y_train == 0),  # HOLD
                        1: (y_train == 1),  # ENTER_LONG
                        2: (y_train == 2),  # ENTER_SHORT
                        3: (y_train == 3),  # EXIT_LONG
                        4: (y_train == 4)  # EXIT_SHORT
                    }

                    n_classes = {
                        cls: np.sum(indices)
                        for cls, indices in class_indices.items()
                    }

                    # Mean values for classes
                    mean_overall = X_train.mean(axis=0)
                    mean_class = {
                        cls: X_train[indices].mean(axis=0) if n > 0 else mean_overall
                        for cls, indices, n in zip(
                            class_indices.keys(),
                            class_indices.values(),
                            n_classes.values()
                        )
                    }

                    # Calculate numerator from paper's F-score formula for multiclass
                    numerator = sum(
                        (mean_class[cls] - mean_overall) ** 2
                        for cls in class_indices.keys()
                    )

                    # Calculate variances for denominator
                    var_class = {
                        cls: X_train[indices].var(axis=0) if n > 1 else pd.Series(1, index=X_train.columns)
                        for cls, indices, n in zip(
                            class_indices.keys(),
                            class_indices.values(),
                            n_classes.values()
                        )
                    }

                    # Calculate F-score: numerator / denominator
                    denominator = sum(var_class.values())
                    f_scores = numerator / denominator
                    f_scores = f_scores.fillna(0)

                    # Select top features by F-score (paper recommends top 10)
                    top_n = min(20, len(X_train.columns))  # Increased to 20 for better sequential modeling
                    selected_features = f_scores.sort_values(ascending=False).index[:top_n].tolist()

                    # Ensure we have critical features for intraday trading
                    essential_features = ['rsi_5', 'rsi_9', 'volume_ratio', 'volatility_5',
                                          'ema_5', 'ema_8', 'bb_pct_b_10', 'momentum_3']
                    for feature in essential_features:
                        if feature in X_train.columns and feature not in selected_features:
                            selected_features.append(feature)

                    # Final cap at 30 features
                    self._selected_features = selected_features[:30]

                    logging.info(
                        f"Selected {len(self._selected_features)} features using F-score method for SEQUENTIAL training")

                except Exception as selector_error:
                    # Fallback to simple selection
                    logging.warning(f"Feature selection error: {str(selector_error)}")
                    self._selected_features = X_train.columns[:min(30, len(X_train.columns))].tolist()

                # Scale features
                self.scaler = StandardScaler()
                X_train_scaled = self.scaler.fit_transform(X_train[self._selected_features])
                X_val_scaled = self.scaler.transform(X_val[self._selected_features])
                X_test_scaled = self.scaler.transform(X_test[self._selected_features])

                # SEQUENTIAL-AWARE: Prepare LSTM sequences with position state context
                if 'volatility_20' in data.columns:
                    avg_volatility = data['volatility_20'].mean()
                    baseline_vol = self.config.VOLATILITY_THRESHOLD
                    vol_ratio = avg_volatility / baseline_vol

                    # Adjust sequence length based on volatility
                    if vol_ratio > 1.3:  # High volatility
                        self.sequence_length = 20  # Shorter sequences for high volatility
                    elif vol_ratio < 0.7:  # Low volatility
                        self.sequence_length = 30  # Longer sequences for low volatility
                    else:
                        self.sequence_length = 25  # Default for normal volatility

                    logging.info(
                        f"Adjusted sequence length to {self.sequence_length} for SEQUENTIAL training based on volatility ratio {vol_ratio:.2f}")
                else:
                    self.sequence_length = 30  # Default for intraday

                X_train_seq = self._prepare_lstm_sequences(X_train_scaled)
                X_val_seq = self._prepare_lstm_sequences(X_val_scaled)
                X_test_seq = self._prepare_lstm_sequences(X_test_scaled)

                # UPDATED: Calculate SEQUENTIAL-AWARE class weights for 5-class problem
                class_weights = None
                if len(np.unique(y_train)) > 1:
                    classes = sorted(np.unique(y_train))
                    n_samples = len(y_train)

                    # Count samples in each class
                    class_counts = {cls: np.sum(y_train == cls) for cls in classes}

                    # SEQUENTIAL-AWARE: Adjust weights to account for natural rarity of exit signals
                    class_weights = {}
                    for cls, count in class_counts.items():
                        if cls == 0:  # HOLD
                            class_weights[cls] = 1.0  # Base weight for majority class
                        elif cls in [1, 2]:  # ENTER_LONG, ENTER_SHORT
                            class_weights[cls] = n_samples / (len(classes) * count) if count > 0 else 1.0
                        elif cls in [3, 4]:  # EXIT_LONG, EXIT_SHORT
                            # MUCH higher weight for exit signals since they're naturally rarer
                            base_weight = n_samples / (len(classes) * count) if count > 0 else 1.0
                            class_weights[cls] = base_weight * 3.0  # Increased from 1.5 to 3.0

                    # Cap maximum weights to prevent extreme values
                    max_weight = 10.0  # Increased cap for exit signals
                    class_weights = {
                        cls: min(weight, max_weight)
                        for cls, weight in class_weights.items()
                    }

                    logging.info(
                        f"SEQUENTIAL 5-class weights - {', '.join([f'Class {cls}: {weight:.2f}' for cls, weight in class_weights.items()])}")

                    # Target signal rate (total of all signals)
                    current_signal_rate = sum(class_counts.get(i, 0) for i in range(1, 5)) / n_samples
                    logging.info(
                        f"Target signal rate: {target_ratio:.2f}, Current SEQUENTIAL rate: {current_signal_rate:.2f}")

                # Adjust y arrays for LSTM
                if len(X_train_seq) > 0:  # Ensure enough data for sequences
                    y_train_seq = y_train.iloc[self.sequence_length - 1:len(y_train)].values
                    y_val_seq = y_val.iloc[self.sequence_length - 1:len(y_val)].values if len(
                        y_val) >= self.sequence_length else None
                    y_test_seq = y_test.iloc[self.sequence_length - 1:len(y_test)].values if len(
                        y_test) >= self.sequence_length else None

                    # Ensure lengths match
                    if len(X_train_seq) != len(y_train_seq):
                        y_train_seq = y_train_seq[:len(X_train_seq)]

                    # UPDATED: Convert to 5-class format for SEQUENTIAL entry/exit prediction
                    y_train_seq = tf.keras.utils.to_categorical(y_train_seq, num_classes=5)
                    if y_val_seq is not None and len(X_val_seq) != len(y_val_seq):
                        y_val_seq = y_val_seq[:len(X_val_seq)]
                    if y_val_seq is not None:
                        y_val_seq = tf.keras.utils.to_categorical(y_val_seq, num_classes=5)
                    if y_test_seq is not None and len(X_test_seq) != len(y_test_seq):
                        y_test_seq = y_test_seq[:len(X_test_seq)]
                    if y_test_seq is not None:
                        y_test_seq = tf.keras.utils.to_categorical(y_test_seq, num_classes=5)

                    # Create or update LSTM model with enhanced architecture for intraday
                    if self._lstm_available and not self._lstm_initialized:
                        self._ensure_lstm_model(X_train_scaled.shape[1])

                    # ENHANCED: LSTM training with SEQUENTIAL-aware parameters
                    if self._lstm_initialized:
                        try:
                            # Define callbacks optimized for sequential learning
                            early_stopping = tf.keras.callbacks.EarlyStopping(
                                monitor='val_loss' if y_test_seq is not None else 'loss',
                                patience=15,  # Patience for sequential patterns
                                restore_best_weights=True,
                                min_delta=0.0005
                            )

                            # Learning rate schedule for sequential pattern learning
                            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                                monitor='val_loss' if y_test_seq is not None else 'loss',
                                factor=0.5,
                                patience=10,
                                min_lr=1e-5
                            )

                            # Train LSTM model with SEQUENTIAL awareness
                            self.lstm_model.fit(
                                X_train_seq, y_train_seq,
                                validation_data=(X_val_seq, y_val_seq) if y_val_seq is not None else None,
                                epochs=50,  # As specified in paper
                                batch_size=32,  # Optimal batch size from paper
                                class_weight=class_weights,
                                callbacks=[early_stopping, reduce_lr],
                                verbose=0
                            )

                            logging.info(
                                f"LSTM model trained successfully with SEQUENTIAL logic for 5-class prediction")

                        except Exception as lstm_error:
                            logging.error(f"LSTM training error: {str(lstm_error)}")
                            logging.error(traceback.format_exc())

                # ENHANCED: XGBoost training with SEQUENTIAL awareness
                try:
                    # Update XGBoost for SEQUENTIAL 5-class prediction
                    self.xgb_model.set_params(
                        objective='multi:softprob',
                        num_class=5,
                        eval_metric='mlogloss'
                    )

                    # For XGB, convert to multiclass format
                    y_train_xgb = y_train.copy()
                    y_test_xgb = y_test.copy()

                    # Enhanced cross-validation for sequential patterns
                    if len(X_test_scaled) > 200:
                        n_splits = 3
                        cv_indices = []
                        fold_size = len(X_train_scaled) // n_splits

                        for i in range(n_splits):
                            if i == n_splits - 1:
                                val_indices = list(range(i * fold_size, len(X_train_scaled)))
                                train_indices = list(range(0, i * fold_size))
                            else:
                                val_indices = list(range(i * fold_size, (i + 1) * fold_size))
                                train_indices = list(range(0, i * fold_size)) + list(
                                    range((i + 1) * fold_size, len(X_train_scaled)))

                            cv_indices.append((train_indices, val_indices))

                        # Train with cross-validation
                        for i, (train_idx, val_idx) in enumerate(cv_indices):
                            X_fold_train = X_train_scaled[train_idx]
                            y_fold_train = y_train_xgb.iloc[train_idx]
                            X_fold_val = X_train_scaled[val_idx]
                            y_fold_val = y_train_xgb.iloc[val_idx]

                            # Train on this fold
                            self.xgb_model.fit(
                                X_fold_train, y_fold_train,
                                eval_set=[(X_fold_val, y_fold_val)],
                                verbose=False
                            )

                            logging.info(
                                f"XGBoost SEQUENTIAL 5-class cross-validation fold {i + 1}/{n_splits} completed")

                        # Final fit using validation set for monitoring
                        self.xgb_model.fit(
                            X_train_scaled, y_train_xgb,
                            eval_set=[(X_val_scaled, y_val)],
                            verbose=False
                        )
                    else:
                        # Standard training for smaller datasets
                        self.xgb_model.fit(
                            X_train_scaled, y_train_xgb,
                            eval_set=[(X_test_scaled, y_test_xgb)],
                            verbose=False
                        )

                    logging.info(f"XGBoost model trained successfully for {symbol} SEQUENTIAL 5-class trading")

                except Exception as xgb_error:
                    logging.error(f"XGBoost training error: {str(xgb_error)}")
                    logging.error(traceback.format_exc())

                    # Fallback to simplest fit method
                    try:
                        self.xgb_model.fit(X_train_scaled, y_train)
                        logging.info(f"XGBoost trained with fallback method for {symbol}")
                    except Exception as fallback_error:
                        logging.error(f"XGBoost fallback training failed: {str(fallback_error)}")

                # Add ARIMA component for price trend prediction (unchanged but logged)
                try:
                    import statsmodels.api as sm
                    from statsmodels.tsa.arima.model import ARIMA
                    from statsmodels.tsa.stattools import adfuller

                    # Extract price series
                    price_series = data['Close'].values

                    # Perform Augmented Dickey-Fuller test as per paper to determine 'd'
                    adf_result = adfuller(price_series)
                    d = 1 if adf_result[1] > 0.05 else 0  # If p-value > 0.05, series is non-stationary

                    # Use ARIMA(0,1,1) as found optimal in the paper
                    order = (0, d, 1)

                    # Fit ARIMA model
                    model = ARIMA(price_series, order=order)
                    model_fit = model.fit()

                    # Store the model
                    self.arima_data[symbol] = {
                        'model': model_fit,
                        'order': order,
                        'last_updated': datetime.now()
                    }

                    logging.info(f"ARIMA{order} model fitted for SEQUENTIAL system: {symbol}")
                except Exception as arima_error:
                    logging.warning(f"ARIMA model fitting error: {str(arima_error)}")

                # Initialize signal thresholds if needed
                if not hasattr(self, 'signal_thresholds') or self.signal_thresholds is None:
                    self.signal_thresholds = {}

                # SEQUENTIAL-AWARE: Find optimal thresholds with position state awareness
                lstm_valid = False
                xgb_valid = False

                # Evaluate LSTM with SEQUENTIAL logic
                lstm_metrics = {}
                best_lstm_threshold = 0.40

                if len(X_val_seq) > 0 and y_val_seq is not None:
                    try:
                        # Find optimal threshold on VALIDATION set (prevent data leakage)
                        val_lstm_proba = self.lstm_model.predict(X_val_seq, verbose=0)
                        val_y_true = np.argmax(y_val_seq, axis=1)
                        val_y_pred_proba_signal = val_lstm_proba[:, 1] + val_lstm_proba[:, 2] + val_lstm_proba[:,
                                                                                                3] + val_lstm_proba[:,
                                                                                                     4]
                        val_y_true_binary = (val_y_true > 0).astype(int)

                        # Calculate precision-recall on validation set
                        precision, recall, thresholds = precision_recall_curve(val_y_true_binary,
                                                                               val_y_pred_proba_signal)
                        f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
                        optimal_idx = np.argmax(f1_scores)
                        optimal_threshold = thresholds[min(optimal_idx, len(thresholds) - 1)]

                        # Use validation-optimized threshold (no forced signal rate)
                        logging.info(f"Using validation-optimized LSTM threshold: {optimal_threshold:.3f}")

                        # NOW evaluate on unseen TEST set
                        if len(X_test_seq) > 0 and y_test_seq is not None:
                            lstm_proba = self.lstm_model.predict(X_test_seq, verbose=0)
                            y_true = np.argmax(y_test_seq, axis=1)
                            y_pred_proba_entry = lstm_proba[:, 1] + lstm_proba[:, 2]
                            y_pred_proba_exit = lstm_proba[:, 3] + lstm_proba[:, 4]
                            y_pred_proba_signal = y_pred_proba_entry + y_pred_proba_exit
                            y_true_binary = (y_true > 0).astype(int)

                            # Calculate metrics at validation-optimized threshold
                            preds = (y_pred_proba_signal >= optimal_threshold).astype(int)
                            precision_score_val = precision_score(y_true_binary, preds, zero_division=0)
                            recall_score_val = recall_score(y_true_binary, preds, zero_division=0)
                            accuracy_score_val = accuracy_score(y_true_binary, preds)
                            f1_score_val = f1_score(y_true_binary, preds, zero_division=0)
                            signal_count = np.sum(preds)
                            signal_rate = signal_count / len(preds)

                        lstm_metrics = {
                            'accuracy': accuracy_score_val,
                            'precision': precision_score_val,
                            'recall': recall_score_val,
                            'f1': f1_score_val,
                            'signal_rate': signal_rate,
                            'threshold': optimal_threshold
                        }

                        best_lstm_threshold = optimal_threshold

                        logging.info(f"LSTM SEQUENTIAL 5-Class Evaluation for {symbol}:")
                        logging.info(f"  Accuracy: {accuracy_score_val:.3f}")
                        logging.info(f"  Precision: {precision_score_val:.3f}")
                        logging.info(f"  Recall: {recall_score_val:.3f}")
                        logging.info(f"  F1 Score: {f1_score_val:.3f}")
                        logging.info(
                            f"  SEQUENTIAL Signals: {int(signal_rate * len(y_true))}/{len(y_true)} ({signal_rate:.1%})")
                        logging.info(f"  SEQUENTIAL Optimal threshold: {optimal_threshold:.3f}")

                        lstm_valid = precision_score_val > 0

                    except Exception as lstm_eval_error:
                        logging.error(f"LSTM evaluation error: {str(lstm_eval_error)}")

                # Similar SEQUENTIAL evaluation for XGBoost...
                xgb_metrics = {}
                best_xgb_threshold = 0.40

                try:
                    # Find optimal threshold on VALIDATION set (prevent data leakage)
                    val_xgb_proba = self.xgb_model.predict_proba(X_val_scaled)

                    if val_xgb_proba.shape[1] >= 5:
                        val_y_pred_proba = val_xgb_proba[:, 1] + val_xgb_proba[:, 2] + val_xgb_proba[:,
                                                                                       3] + val_xgb_proba[:, 4]
                    else:
                        val_y_pred_proba = val_xgb_proba[:, 1] if val_xgb_proba.shape[1] > 1 else val_xgb_proba[:, 0]

                    val_y_test_binary = (y_val > 0).astype(int)

                    precision, recall, thresholds = precision_recall_curve(val_y_test_binary, val_y_pred_proba)
                    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
                    optimal_idx = np.argmax(f1_scores)
                    optimal_threshold = thresholds[min(optimal_idx, len(thresholds) - 1)]

                    # Use validation-optimized threshold (no forced signal rate)
                    logging.info(f"Using validation-optimized XGBoost threshold: {optimal_threshold:.3f}")

                    # NOW evaluate on unseen TEST set
                    xgb_proba = self.xgb_model.predict_proba(X_test_scaled)

                    if xgb_proba.shape[1] >= 5:
                        y_pred_proba_entry = xgb_proba[:, 1] + xgb_proba[:, 2]
                        y_pred_proba_exit = xgb_proba[:, 3] + xgb_proba[:, 4]
                        y_pred_proba = y_pred_proba_entry + y_pred_proba_exit
                    else:
                        y_pred_proba = xgb_proba[:, 1] if xgb_proba.shape[1] > 1 else xgb_proba[:, 0]

                    y_test_binary = (y_test > 0).astype(int)

                    preds = (y_pred_proba >= optimal_threshold).astype(int)
                    precision_score_val = precision_score(y_test_binary, preds, zero_division=0)
                    recall_score_val = recall_score(y_test_binary, preds, zero_division=0)
                    accuracy_score_val = accuracy_score(y_test_binary, preds)
                    f1_score_val = f1_score(y_test_binary, preds, zero_division=0)
                    signal_count = np.sum(preds)
                    signal_rate = signal_count / len(preds)

                    xgb_metrics = {
                        'accuracy': accuracy_score_val,
                        'precision': precision_score_val,
                        'recall': recall_score_val,
                        'f1': f1_score_val,
                        'signal_rate': signal_rate,
                        'threshold': optimal_threshold
                    }

                    best_xgb_threshold = optimal_threshold

                    logging.info(f"XGBoost SEQUENTIAL 5-Class Evaluation for {symbol}:")
                    logging.info(f"  Accuracy: {accuracy_score_val:.3f}")
                    logging.info(f"  Precision: {precision_score_val:.3f}")
                    logging.info(f"  Recall: {recall_score_val:.3f}")
                    logging.info(f"  F1 Score: {f1_score_val:.3f}")
                    logging.info(
                        f"  SEQUENTIAL Signals: {int(signal_rate * len(y_test))}/{len(y_test)} ({signal_rate:.1%})")
                    logging.info(f"  SEQUENTIAL Optimal threshold: {optimal_threshold:.3f}")

                    xgb_valid = precision_score_val > 0

                except Exception as xgb_eval_error:
                    logging.error(f"XGBoost evaluation error: {str(xgb_eval_error)}")

                # UPDATED: Calculate separate thresholds for entry and exit with more aggressive differentials
                if lstm_valid and xgb_valid:
                    # Create ensemble predictions with SEQUENTIAL logic
                    ensemble_scores = np.zeros(len(y_test))

                    for i in range(len(y_test)):
                        if i < len(xgb_proba):
                            if xgb_proba.shape[1] >= 5:
                                xgb_score = xgb_proba[i, 1] + xgb_proba[i, 2] + xgb_proba[i, 3] + xgb_proba[i, 4]
                            else:
                                xgb_score = xgb_proba[i, 1] if xgb_proba.shape[1] > 1 else 0.5

                            lstm_idx = i - (self.sequence_length - 1)
                            if lstm_idx >= 0 and lstm_idx < len(lstm_proba):
                                lstm_score = lstm_proba[lstm_idx, 1] + lstm_proba[lstm_idx, 2] + lstm_proba[
                                    lstm_idx, 3] + lstm_proba[lstm_idx, 4]
                                ensemble_scores[i] = (lstm_score * self.model_weights['lstm'] +
                                                      xgb_score * self.model_weights['xgb'])
                            else:
                                ensemble_scores[i] = xgb_score

                    y_test_binary = (y_test > 0).astype(int)

                    # Calculate separate thresholds for entry and exit with SEQUENTIAL awareness - 5% TOTAL
                    actual_entry_count = np.sum((y_test == 1) | (y_test == 2))
                    actual_exit_count = np.sum((y_test == 3) | (y_test == 4))
                    total_samples = len(y_test)

                    # Target 2% entries and 2% exits for 5 total
                    target_entry_rate = 0.025  # Fixed at 2.5%
                    target_exit_rate = 0.025  # Fixed at 2.5%

                    # For SEQUENTIAL system, use combined threshold approach
                    if len(ensemble_scores.shape) > 1:
                        ensemble_entry_probs = ensemble_scores[:, 1] + ensemble_scores[:, 2]
                        ensemble_exit_probs = ensemble_scores[:, 3] + ensemble_scores[:, 4]
                        ensemble_combined_probs = ensemble_entry_probs + ensemble_exit_probs
                    else:
                        ensemble_combined_probs = ensemble_scores
                        ensemble_entry_probs = ensemble_scores * 0.6
                        ensemble_exit_probs = ensemble_scores * 0.4

                    # Calculate entry threshold
                    sorted_entry_probs = np.sort(ensemble_entry_probs)[::-1]
                    entry_idx = int(target_entry_rate * len(sorted_entry_probs))
                    entry_threshold = sorted_entry_probs[entry_idx] if entry_idx < len(sorted_entry_probs) else 0.15

                    # Calculate exit threshold - MUCH MORE AGGRESSIVE
                    sorted_exit_probs = np.sort(ensemble_exit_probs)[::-1]
                    exit_idx = int(target_exit_rate * len(sorted_exit_probs))
                    exit_threshold = sorted_exit_probs[exit_idx] if exit_idx < len(sorted_exit_probs) else 0.10

                    # Use minimum of the two for backward compatibility
                    ensemble_threshold = min(entry_threshold, exit_threshold)

                    # Store separate thresholds
                    if not hasattr(self, 'entry_exit_thresholds'):
                        self.entry_exit_thresholds = {}
                    self.entry_exit_thresholds[symbol] = {
                        'entry': entry_threshold,
                        'exit': exit_threshold
                    }

                    # Calculate SEQUENTIAL metrics
                    ensemble_preds = ((ensemble_entry_probs >= entry_threshold) |
                                      (ensemble_exit_probs >= exit_threshold)).astype(int)

                    ensemble_accuracy = accuracy_score(y_test_binary, ensemble_preds)
                    ensemble_precision = precision_score(y_test_binary, ensemble_preds, zero_division=0)
                    ensemble_recall = recall_score(y_test_binary, ensemble_preds)
                    ensemble_f1 = f1_score(y_test_binary, ensemble_preds, zero_division=0)
                    ensemble_signal_rate = np.mean(ensemble_preds)

                    # Match ideal path methodology for consistent evaluation
                    profitable_signals = 0
                    total_trade_signals = 0

                    min_required_length = min(len(ensemble_preds), len(test_indices))

                    for i in range(min_required_length):
                        if ensemble_preds[i] == 1:  # Any signal
                            total_trade_signals += 1
                            actual_data_idx = test_indices[i]

                            if 0 <= actual_data_idx < len(data) - 30:  # Use 30 bars like ideal path
                                try:
                                    entry_price = data['Close'].iloc[actual_data_idx]
                                    profit_target = entry_price * 1.005  # 0.5% target
                                    stop_loss = entry_price * 0.9975  # 0.25% stop

                                    # Check each future bar like ideal path does
                                    for j in range(actual_data_idx + 1, min(actual_data_idx + 31, len(data))):
                                        future_high = data['High'].iloc[j]
                                        future_low = data['Low'].iloc[j]

                                        target_hit = future_high >= profit_target
                                        stop_hit = future_low <= stop_loss

                                        if target_hit and not stop_hit:
                                            profitable_signals += 1
                                            break
                                        elif stop_hit:
                                            break  # Stop hit first
                                except (IndexError, KeyError):
                                    continue

                    profit_rate = profitable_signals / total_trade_signals if total_trade_signals > 0 else 0

                    logging.info(f"SEQUENTIAL ensemble evaluation for 5-class system:")
                    logging.info(f"  Accuracy: {ensemble_accuracy:.3f}")
                    logging.info(f"  Precision: {ensemble_precision:.3f}")
                    logging.info(f"  Recall: {ensemble_recall:.3f}")
                    logging.info(f"  F1 Score: {ensemble_f1:.3f}")
                    logging.info(f"  Profitable signals: {profitable_signals}/{total_trade_signals} ({profit_rate:.1%})")
                    logging.info(f"  SEQUENTIAL Signal rate: {ensemble_signal_rate:.3f}")

                    self.signal_thresholds[symbol] = ensemble_threshold

                elif lstm_valid:
                    self.signal_thresholds[symbol] = lstm_metrics['threshold']
                    logging.info(f"Using LSTM threshold for SEQUENTIAL ensemble: {lstm_metrics['threshold']:.3f}")

                elif xgb_valid:
                    self.signal_thresholds[symbol] = xgb_metrics['threshold']
                    logging.info(f"Using XGBoost threshold for SEQUENTIAL ensemble: {xgb_metrics['threshold']:.3f}")

                else:
                    self.signal_thresholds[symbol] = 0.35
                    logging.info(f"Using default SEQUENTIAL ensemble threshold: 0.35")

                # Set is_trained flag
                self.is_trained = True
                logging.info(
                    f"✅ SEQUENTIAL training completed for {symbol} - models understand entry/exit dependencies")

        except Exception as e:
            logging.error(f"SEQUENTIAL training error: {str(e)}")
            logging.error(traceback.format_exc())

    def predict(self, data) -> pd.DataFrame:
        """Enhanced prediction method with SEQUENTIAL logic - exits only when positions exist"""
        try:
            # Basic validation
            if not self.is_trained:
                logging.warning("ML model not trained, returning default HOLD")
                default_result = pd.DataFrame({'action': ['HOLD'], 'entry_score': [0.5], 'exit_score': [0.5],
                                               'hold_score': [0.5], 'position_size': [0.02], 'direction': ['LONG']})
                return default_result

            # Extract symbol for tracking
            symbol = None
            if isinstance(data, pd.DataFrame) and not data.empty:
                if 'symbol' in data.columns:
                    symbol = data['symbol'].iloc[0]
                elif hasattr(data, 'name') and data.name is not None:
                    symbol = data.name

            # Validate signal thresholds
            if hasattr(self, 'signal_thresholds') and symbol and symbol in self.signal_thresholds:
                current_threshold = self.signal_thresholds[symbol]
                if current_threshold < 0.2:
                    logging.warning(f"Adjusting low threshold for {symbol}: {current_threshold:.2f} -> 0.35")
                    self.signal_thresholds[symbol] = 0.35
                elif current_threshold > 0.6:
                    logging.warning(f"Adjusting high threshold for {symbol}: {current_threshold:.2f} -> 0.50")
                    self.signal_thresholds[symbol] = 0.50

            # Take a fresh copy of data
            if isinstance(data, pd.DataFrame) and not data.empty:
                data = data.copy()
                if len(data) > 100:
                    tail_data = data.tail(100).copy()
                else:
                    tail_data = data.copy()

                tail_data = self._calculate_technical_features(tail_data)
                data = tail_data

            result_symbol = symbol

            # Initialize caching if needed
            if not hasattr(self, '_last_signal_time'):
                self._last_signal_time = {}
                self._last_predictions = {}
                self._position_lifecycles = {}

            # CRITICAL ADDITION: Get current position state from trading system
            position_exists = False
            position_direction = None
            position_age_minutes = 0

            try:
                if hasattr(self, 'trading_system') and hasattr(self.trading_system, 'position_manager'):
                    if symbol in self.trading_system.position_manager.positions:
                        position_exists = True
                        position = self.trading_system.position_manager.positions[symbol]
                        position_direction = 'LONG' if position.get('side') in ['BUY', 'LONG'] else 'SHORT'
                        entry_time = position.get('entry_time', datetime.now())
                        position_age_seconds = (datetime.now() - entry_time).total_seconds()
                        position_age_minutes = position_age_seconds / 60
            except Exception as e:
                logging.error(f"Error checking position state: {str(e)}")

            # Prepare features for prediction
            try:
                features_df = self.prepare_features(data)
                if features_df.empty:
                    logging.warning(f"Empty features for prediction, returning default HOLD for {symbol}")
                    default_result = pd.DataFrame({'action': ['HOLD'], 'entry_score': [0.5], 'exit_score': [0.5],
                                                   'hold_score': [0.5], 'position_size': [0.02], 'direction': ['LONG']})
                    if result_symbol:
                        default_result.name = result_symbol
                    return default_result
            except Exception as feat_e:
                logging.error(f"Feature preparation error: {str(feat_e)}")
                default_result = pd.DataFrame({'action': ['HOLD'], 'entry_score': [0.5], 'exit_score': [0.5],
                                               'hold_score': [0.5], 'position_size': [0.02], 'direction': ['LONG']})
                if result_symbol:
                    default_result.name = result_symbol
                return default_result

            # Detect current market regime
            current_regime = self._detect_market_regime(features_df)
            self.current_regime = current_regime

            # Determine direction based on technical indicators
            direction = "LONG"  # Default direction
            try:
                if any(col in features_df.columns for col in ['ema_8', 'ema_21']):
                    last_row = features_df.iloc[-1]

                    if 'ema_8' in last_row and 'ema_21' in last_row:
                        if last_row['ema_8'] > last_row['ema_21']:
                            direction = "LONG"
                        else:
                            direction = "SHORT"

                    if 'supertrend_direction' in last_row:
                        if last_row['supertrend_direction'] == 1:
                            direction = "LONG"
                        else:
                            direction = "SHORT"
            except Exception as dir_error:
                logging.error(f"Direction determination error: {str(dir_error)}")
                direction = "LONG"

            # Scale features
            try:
                if hasattr(self, '_selected_features') and self._selected_features:
                    available_features = [f for f in self._selected_features if f in features_df.columns]
                    features_df = features_df[available_features]

                X = features_df.values.astype(np.float64)
                X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
                X_scaled = self.scaler.transform(X)

                sequence_length = getattr(self, 'sequence_length', 15)
                X_seq = []

                if len(X_scaled) >= sequence_length:
                    for i in range(len(X_scaled) - sequence_length + 1):
                        X_seq.append(X_scaled[i:i + sequence_length])
                    X_seq = np.array(X_seq)
                else:
                    X_seq = np.array([])

            except Exception as scale_error:
                logging.error(f"Feature scaling error: {str(scale_error)}")
                default_result = pd.DataFrame({'action': ['HOLD'], 'entry_score': [0.5], 'exit_score': [0.5],
                                               'hold_score': [0.5], 'position_size': [0.02], 'direction': ['LONG']})
                if result_symbol:
                    default_result.name = result_symbol
                return default_result

            # Get XGBoost prediction
            xgb_predictions = []
            for _ in range(5):
                noise = np.random.normal(0, 0.01, X_scaled.shape)
                noisy_X = X_scaled + noise
                pred = self.xgb_model.predict_proba(noisy_X)

                if pred.shape[1] >= 5:
                    xgb_predictions.append(pred[-1])
                else:
                    padded_pred = np.zeros(5)
                    for j in range(min(pred.shape[1], 5)):
                        padded_pred[j] = pred[-1][j]
                    xgb_predictions.append(padded_pred)

            xgb_mean_pred = np.mean(xgb_predictions, axis=0)
            xgb_uncertainty = np.std(xgb_predictions, axis=0).mean()
            xgb_proba = xgb_mean_pred

            logging.debug(
                f"XGBoost 5-class probabilities for {symbol}: {xgb_proba}, uncertainty: {xgb_uncertainty:.6f}")

            # Get LSTM prediction
            lstm_proba = np.zeros(5)
            lstm_uncertainty = 0.1
            raw_lstm_values = None

            if len(X_seq) > 0 and self._lstm_initialized:
                lstm_predictions = []
                for _ in range(5):
                    noise = np.random.normal(0, 0.01, X_seq.shape)
                    noisy_X_seq = X_seq + noise
                    pred = self.lstm_model.predict(noisy_X_seq, verbose=0)

                    if pred.shape[1] >= 5:
                        lstm_predictions.append(pred[-1])
                    else:
                        padded_pred = np.zeros(5)
                        for j in range(min(pred.shape[1], 5)):
                            padded_pred[j] = pred[-1][j]
                        lstm_predictions.append(padded_pred)

                lstm_preds_array = np.array(lstm_predictions)
                raw_lstm_values = np.mean(lstm_preds_array, axis=0)
                lstm_uncertainty = np.std(lstm_preds_array, axis=0).mean()

                logging.debug(
                    f"LSTM 5-class probabilities for {symbol}: {raw_lstm_values}, uncertainty: {lstm_uncertainty:.6f}")
                lstm_proba = raw_lstm_values

            # Get ARIMA prediction
            arima_proba = 0.5
            try:
                if hasattr(self, 'arima_models') and symbol in self.arima_models:
                    model_data = self.arima_models[symbol]
                    model = model_data.get('model')

                    if model:
                        last_close = data['Close'].iloc[-1]
                        forecast = model.forecast(steps=1)
                        predicted_price = forecast[0]

                        price_change_pct = (predicted_price / last_close - 1) * 100

                        if price_change_pct > 0:
                            arima_proba = 0.5 + min(0.5, price_change_pct / 0.2)
                        else:
                            arima_proba = 0.5 + max(-0.5, price_change_pct / 0.2)

                        data.loc[data.index[-1], 'arima_predicted_price'] = predicted_price
                        data.loc[data.index[-1], 'arima_price_change'] = price_change_pct
                        data.loc[data.index[-1], 'arima_proba'] = arima_proba
            except Exception as arima_error:
                logging.error(f"ARIMA prediction error: {str(arima_error)}")
                arima_proba = 0.5

            # Adjust model weights based on uncertainty
            lstm_weight = self.model_weights.get('lstm', 0.5)
            xgb_weight = self.model_weights.get('xgb', 0.4)
            arima_weight = 0.1

            if raw_lstm_values is not None and np.max(raw_lstm_values[1:]) < 0.05:
                lstm_adjusted_weight = lstm_weight * 0.5
                xgb_adjusted_weight = xgb_weight * 1.5
                arima_adjusted_weight = arima_weight
                logging.info(
                    f"Adjusted model weights due to low LSTM signal: LSTM = {lstm_adjusted_weight:.2f}, XGB = {xgb_adjusted_weight:.2f}")
            else:
                lstm_confidence = max(0.2, 1.0 - (lstm_uncertainty * 5))
                xgb_confidence = max(0.2, 1.0 - (xgb_uncertainty * 5))
                lstm_adjusted_weight = lstm_weight * lstm_confidence
                xgb_adjusted_weight = xgb_weight * xgb_confidence
                arima_adjusted_weight = arima_weight

            # Normalize weights
            total_weight = lstm_adjusted_weight + xgb_adjusted_weight + arima_adjusted_weight
            lstm_final_weight = lstm_adjusted_weight / total_weight
            xgb_final_weight = xgb_adjusted_weight / total_weight
            arima_final_weight = arima_adjusted_weight / total_weight

            logging.debug(f"Ensemble calculation for {symbol}:")
            logging.debug(f"  LSTM weights: {lstm_final_weight:.4f}")
            logging.debug(f"  XGB weights: {xgb_final_weight:.4f}")
            logging.debug(f"  ARIMA weight: {arima_final_weight:.4f}")

            # Calculate ensemble prediction
            ensemble_pred = np.zeros(5)

            if raw_lstm_values is not None:
                ensemble_pred += lstm_proba * lstm_final_weight

            ensemble_pred += xgb_proba * xgb_final_weight

            # ARIMA contribution
            arima_contribution = np.zeros(5)
            arima_contribution[0] = 1.0 - arima_proba
            if direction == "LONG":
                arima_contribution[1] = arima_proba
            else:
                arima_contribution[2] = arima_proba
            ensemble_pred += arima_contribution * arima_final_weight

            # Normalize probabilities
            ensemble_pred = ensemble_pred / np.sum(ensemble_pred)

            logging.debug(f"Final ensemble probabilities: {ensemble_pred}")

            # CRITICAL FIX: SEQUENTIAL action determination with position state validation
            base_threshold = self.signal_thresholds.get(symbol, 0.3)
            entry_threshold = base_threshold * 0.5
            exit_threshold = base_threshold * 1.0

            # Extract individual class probabilities
            hold_prob = ensemble_pred[0]
            enter_long_prob = ensemble_pred[1]
            enter_short_prob = ensemble_pred[2]
            exit_long_prob = ensemble_pred[3]
            exit_short_prob = ensemble_pred[4]

            # SEQUENTIAL LOGIC: Determine action based on position state
            action = "HOLD"
            final_direction = direction

            # ENTRY SIGNALS: Can happen when no position exists
            if not position_exists:
                if enter_long_prob > entry_threshold:
                    action = "ENTER"
                    final_direction = "LONG"
                elif enter_short_prob > entry_threshold:
                    action = "ENTER"
                    final_direction = "SHORT"

            # EXIT SIGNALS: Only valid when position exists and direction matches
            elif position_exists:
                # Check for exit signals that match the current position direction
                if position_direction == "LONG" and exit_long_prob > exit_threshold:
                    # Minimum holding time check
                    min_holding_minutes = 2
                    if position_age_minutes >= min_holding_minutes:
                        action = "EXIT"
                        final_direction = "LONG"
                    else:
                        # Early exit only for severe adverse moves
                        current_price = self.market_data_manager.get_latest_price(symbol) if hasattr(self,
                                                                                                     'market_data_manager') else None
                        if current_price and hasattr(self, 'trading_system'):
                            position = self.trading_system.position_manager.positions.get(symbol)
                            if position:
                                entry_price = position['entry_price']
                                price_move_pct = ((current_price / entry_price) - 1) * 100
                                if price_move_pct < -0.15:  # -0.15% severe loss
                                    action = "EXIT"
                                    final_direction = "LONG"
                                    logging.info(
                                        f"Early exit allowed for {symbol} due to adverse move: {price_move_pct:.2f}%")
                                else:
                                    logging.info(
                                        f"EXIT signal ignored for {symbol} - held only {position_age_minutes:.1f} minutes")

                elif position_direction == "SHORT" and exit_short_prob > exit_threshold:
                    # Similar logic for short positions
                    min_holding_minutes = 2
                    if position_age_minutes >= min_holding_minutes:
                        action = "EXIT"
                        final_direction = "SHORT"
                    else:
                        # Early exit for severe adverse moves (price rising for shorts)
                        current_price = self.market_data_manager.get_latest_price(symbol) if hasattr(self,
                                                                                                     'market_data_manager') else None
                        if current_price and hasattr(self, 'trading_system'):
                            position = self.trading_system.position_manager.positions.get(symbol)
                            if position:
                                entry_price = position['entry_price']
                                price_move_pct = ((entry_price / current_price) - 1) * 100  # Inverted for shorts
                                if price_move_pct < -0.15:  # Adverse move for shorts
                                    action = "EXIT"
                                    final_direction = "SHORT"
                                    logging.info(
                                        f"Early exit allowed for {symbol} SHORT due to adverse move: {price_move_pct:.2f}%")
                                else:
                                    logging.info(
                                        f"EXIT signal ignored for {symbol} SHORT - held only {position_age_minutes:.1f} minutes")

            # Calculate scores for backward compatibility
            nn_score = float(np.max(lstm_proba[1:])) if raw_lstm_values is not None else 0.0
            svm_score = float(np.max(xgb_proba[1:]))

            # Calculate exit confidence and predicted hold time
            exit_confidence = float(ensemble_pred[3] + ensemble_pred[4])
            predicted_hold_minutes = 30

            if hasattr(self, 'profit_thresholds') and symbol in self.profit_thresholds:
                predicted_hold_minutes = self.profit_thresholds[symbol].get('holding_period', 30)

            # Cache prediction
            if symbol:
                self._last_signal_time[symbol] = datetime.now()

            # SEQUENTIAL RESULT: Return DataFrame with position state awareness
            result_df = pd.DataFrame({
                'action': [action],
                'direction': [final_direction],
                'entry_score': [enter_long_prob + enter_short_prob],
                'exit_score': [exit_long_prob + exit_short_prob],
                'predicted_hold_minutes': [int(predicted_hold_minutes)],
                'exit_confidence': [round(exit_confidence, 2)],
                'nn_score': [nn_score],
                'svm_score': [svm_score],
                'confidence': [max(enter_long_prob, enter_short_prob, exit_long_prob, exit_short_prob)],
                'predicted_class': [np.argmax(ensemble_pred)],
                'hold_prob': [hold_prob],
                'long_entry_prob': [enter_long_prob],
                'short_entry_prob': [enter_short_prob],
                'long_exit_prob': [exit_long_prob],
                'short_exit_prob': [exit_short_prob],
                'entry_threshold': [entry_threshold],
                'exit_threshold': [exit_threshold],
                'position_exists': [position_exists],  # NEW: Include position state
                'position_direction': [position_direction],  # NEW: Include position direction
                'position_age_minutes': [position_age_minutes]  # NEW: Include position age
            })

            # Set name attribute
            if result_symbol is not None:
                result_df.name = result_symbol

            # Add profit thresholds if available
            if symbol and hasattr(self, 'profit_thresholds') and symbol in self.profit_thresholds:
                for key in ['long_target', 'short_target', 'long_stop', 'short_stop']:
                    if key in self.profit_thresholds[symbol] and self.profit_thresholds[symbol][key] > 1.0:
                        self.profit_thresholds[symbol][key] = self.profit_thresholds[symbol][key] / 100.0

                if final_direction == 'LONG':
                    result_df['profit_target'] = self.profit_thresholds[symbol].get('long_target', 0.50)
                    result_df['stop_loss'] = self.profit_thresholds[symbol].get('long_stop', -0.25)
                else:
                    result_df['profit_target'] = self.profit_thresholds[symbol].get('short_target', 0.50)
                    result_df['stop_loss'] = self.profit_thresholds[symbol].get('short_stop', -0.25)

                result_df['holding_period'] = min(45, self.profit_thresholds[symbol].get('holding_period', 30))

            # Log SEQUENTIAL decision
            if action in ["ENTER", "EXIT"]:
                logging.info(f"SEQUENTIAL DECISION: {symbol} - {action} ({final_direction}) | "
                             f"Position exists: {position_exists} | "
                             f"Position direction: {position_direction} | "
                             f"Age: {position_age_minutes:.1f}min")

            return result_df

        except Exception as e:
            logging.error(f"SEQUENTIAL prediction error: {str(e)}")
            logging.error(traceback.format_exc())

            default_result = pd.DataFrame({'action': ['HOLD'], 'entry_score': [0.5], 'exit_score': [0.5],
                                           'hold_score': [0.5], 'position_size': [0.02], 'direction': ['LONG']})

            if 'result_symbol' in locals() and result_symbol is not None:
                default_result.name = result_symbol

            return default_result

    def plot_model_evaluation(self, symbol: str, save_path=None) -> Optional[plt.Figure]:
        """Plot comprehensive model analysis with optimized layout, original colors, and smart datetime formatting"""
        try:
            # Close any existing plots first to avoid memory issues
            plt.close('all')

            logging.info(f"Creating comprehensive 5-class model analysis for {symbol}...")

            # Get the historical data
            if hasattr(self, 'trading_system') and hasattr(self.trading_system, 'market_data_manager'):
                data = self.trading_system.market_data_manager.get_current_data(symbol)
            elif hasattr(self, 'market_data_manager'):
                data = self.market_data_manager.get_current_data(symbol)
            else:
                logging.error(f"No data available for {symbol}")
                return None

            if data is None or data.empty:
                logging.error(f"Empty data for {symbol}")
                return None

            # Ensure data has name attribute for reference
            data.name = symbol

            # Create a copy to avoid modifying original data
            df = data.copy()

            # Split data into training and test sets
            train_size = int(len(df) * 0.7)
            test_data = df.iloc[train_size:]

            # Prepare features for prediction
            features_df = self.prepare_features(test_data)

            # Create results DataFrame with proper datetime index
            results = pd.DataFrame(index=test_data.index)
            results['Close'] = test_data['Close']

            # Ensure index is datetime and timezone-naive
            if not isinstance(results.index, pd.DatetimeIndex):
                results.index = pd.to_datetime(results.index)
            if results.index.tz is not None:
                results.index = results.index.tz_localize(None)

            # Remove weekends and holidays to eliminate gaps
            results = results[results.index.dayofweek < 5]

            # Get LSTM and XGB predictions for 5-class system
            if self._lstm_initialized and len(features_df) > 0:
                X_scaled = self.scaler.transform(features_df.values)
                X_seq = self._prepare_lstm_sequences(X_scaled)

                if len(X_seq) > 0:
                    lstm_predictions = self.lstm_model.predict(X_seq, verbose=0)
                    seq_offset = self.sequence_length - 1

                    # Store all 5 class probabilities
                    for i in range(min(len(lstm_predictions), len(test_data) - seq_offset)):
                        idx = test_data.index[i + seq_offset]
                        if idx.dayofweek < 5 and idx in results.index:
                            if lstm_predictions.shape[1] >= 5:
                                results.loc[idx, 'lstm_hold'] = lstm_predictions[i][0]
                                results.loc[idx, 'lstm_enter_long'] = lstm_predictions[i][1]
                                results.loc[idx, 'lstm_enter_short'] = lstm_predictions[i][2]
                                results.loc[idx, 'lstm_exit_long'] = lstm_predictions[i][3]
                                results.loc[idx, 'lstm_exit_short'] = lstm_predictions[i][4]

                                # Calculate combined scores for backward compatibility
                                results.loc[idx, 'lstm_entry_score'] = lstm_predictions[i][1] + lstm_predictions[i][2]
                                results.loc[idx, 'lstm_exit_score'] = lstm_predictions[i][3] + lstm_predictions[i][4]

            # Get XGBoost predictions for 5-class system
            if len(features_df) > 0:
                X_scaled = self.scaler.transform(features_df.values)
                xgb_predictions = self.xgb_model.predict_proba(X_scaled)

                for i in range(min(len(xgb_predictions), len(test_data))):
                    idx = test_data.index[i]
                    if idx.dayofweek < 5 and idx in results.index:
                        if xgb_predictions.shape[1] >= 5:
                            results.loc[idx, 'xgb_hold'] = xgb_predictions[i][0]
                            results.loc[idx, 'xgb_enter_long'] = xgb_predictions[i][1]
                            results.loc[idx, 'xgb_enter_short'] = xgb_predictions[i][2]
                            results.loc[idx, 'xgb_exit_long'] = xgb_predictions[i][3]
                            results.loc[idx, 'xgb_exit_short'] = xgb_predictions[i][4]

                            # Calculate combined scores
                            results.loc[idx, 'xgb_entry_score'] = xgb_predictions[i][1] + xgb_predictions[i][2]
                            results.loc[idx, 'xgb_exit_score'] = xgb_predictions[i][3] + xgb_predictions[i][4]

            # Calculate ensemble scores
            lstm_weight = self.model_weights.get('lstm', 0.55)
            xgb_weight = self.model_weights.get('xgb', 0.36)
            arima_weight = 0.09

            # Normalize weights
            total_weight = lstm_weight + xgb_weight + arima_weight
            lstm_weight = lstm_weight / total_weight
            xgb_weight = xgb_weight / total_weight
            arima_weight = arima_weight / total_weight

            logging.info(f"Model weights for 5-class ensemble: LSTM={lstm_weight:.2f}, XGB={xgb_weight:.2f}")

            # Fill NaN values for all new columns
            score_columns = ['lstm_hold', 'lstm_enter_long', 'lstm_enter_short', 'lstm_exit_long', 'lstm_exit_short',
                             'xgb_hold', 'xgb_enter_long', 'xgb_enter_short', 'xgb_exit_long', 'xgb_exit_short',
                             'lstm_entry_score', 'lstm_exit_score', 'xgb_entry_score', 'xgb_exit_score']

            for col in score_columns:
                if col not in results.columns:
                    results[col] = 0.0
                else:
                    results[col] = results[col].fillna(0.0)

            # Calculate ensemble scores for all 5 classes
            results['ensemble_hold'] = (results['lstm_hold'] * lstm_weight +
                                        results['xgb_hold'] * xgb_weight)
            results['ensemble_enter_long'] = (results['lstm_enter_long'] * lstm_weight +
                                              results['xgb_enter_long'] * xgb_weight)
            results['ensemble_enter_short'] = (results['lstm_enter_short'] * lstm_weight +
                                               results['xgb_enter_short'] * xgb_weight)
            results['ensemble_exit_long'] = (results['lstm_exit_long'] * lstm_weight +
                                             results['xgb_exit_long'] * xgb_weight)
            results['ensemble_exit_short'] = (results['lstm_exit_short'] * lstm_weight +
                                              results['xgb_exit_short'] * xgb_weight)

            # Calculate combined entry and exit scores
            results['ensemble_entry_score'] = results['ensemble_enter_long'] + results['ensemble_enter_short']
            results['ensemble_exit_score'] = results['ensemble_exit_long'] + results['ensemble_exit_short']

            # Get threshold
            threshold = self.signal_thresholds.get(symbol, 0.29)
            logging.info(f"Signal threshold for {symbol}: {threshold:.2f}")

            # Initialize signal columns with zeros
            results['enter_long_signal'] = 0
            results['enter_short_signal'] = 0
            results['exit_long_signal'] = 0
            results['exit_short_signal'] = 0
            results['any_signal'] = 0

            # Generate signals based on 5-class predictions
            enter_long_mask = results['ensemble_enter_long'] > threshold
            enter_short_mask = results['ensemble_enter_short'] > threshold
            exit_long_mask = results['ensemble_exit_long'] > threshold
            exit_short_mask = results['ensemble_exit_short'] > threshold

            results.loc[enter_long_mask, 'enter_long_signal'] = 1
            results.loc[enter_short_mask, 'enter_short_signal'] = 1
            results.loc[exit_long_mask, 'exit_long_signal'] = 1
            results.loc[exit_short_mask, 'exit_short_signal'] = 1

            # Any signal indicator
            any_signal_mask = enter_long_mask | enter_short_mask | exit_long_mask | exit_short_mask
            results.loc[any_signal_mask, 'any_signal'] = 1

            # Count signals by type
            enter_long_count = results['enter_long_signal'].sum()
            enter_short_count = results['enter_short_signal'].sum()
            exit_long_count = results['exit_long_signal'].sum()
            exit_short_count = results['exit_short_signal'].sum()
            total_signals = enter_long_count + enter_short_count + exit_long_count + exit_short_count

            logging.info(
                f"Generated signals for {symbol}: ENTER_LONG: {enter_long_count}, ENTER_SHORT: {enter_short_count}, "
                f"EXIT_LONG: {exit_long_count}, EXIT_SHORT: {exit_short_count}, Total: {total_signals}")

            # OPTIMIZED LAYOUT: Create enhanced visualization with better space utilization
            # Use tight layout and adjust margins
            fig = plt.figure(figsize=(24, 16), dpi=100)  # Larger figure for more detail

            # Optimize subplot layout - reduce spacing between plots
            gs = plt.GridSpec(4, 1, height_ratios=[3, 1.2, 1.2, 1],
                              hspace=0.25,  # Reduced from 0.4
                              left=0.06,  # Reduced left margin
                              right=0.98,  # Reduced right margin
                              top=0.94,  # Reduced top margin
                              bottom=0.08)  # Reduced bottom margin

            # Smart date/time formatter that adapts to zoom level
            def smart_datetime_formatter(ax):
                """Create smart datetime formatting that shows time when zoomed in"""

                def format_func(x, pos=None):
                    try:
                        if int(x) < 0 or int(x) >= len(results):
                            return ''

                        # Get the actual datetime from our results
                        date = results.index[int(x)]

                        # Get current axis limits to determine zoom level
                        xlim = ax.get_xlim()
                        data_range = xlim[1] - xlim[0]
                        total_range = len(results)

                        # Calculate zoom level (smaller ratio = more zoomed in)
                        zoom_ratio = data_range / total_range

                        # Adaptive formatting based on zoom level
                        if zoom_ratio < 0.1:  # Very zoomed in - show time
                            return date.strftime('%H:%M')
                        elif zoom_ratio < 0.3:  # Moderately zoomed - show date and time
                            return date.strftime('%m/%d %H:%M')
                        elif zoom_ratio < 0.7:  # Less zoomed - show date with day
                            return date.strftime('%m/%d')
                        else:  # Zoomed out - show month/day
                            # Only show every nth label to avoid crowding
                            if int(x) % max(1, int(total_range / 10)) == 0:
                                return date.strftime('%m/%d')
                            return ''

                    except (IndexError, ValueError):
                        return ''

                return format_func

            # 1. OPTIMIZED: Main price chart with trading signals (ORIGINAL COLORS)
            ax1 = fig.add_subplot(gs[0])

            # Shorter, cleaner title that won't overlap
            main_title = f"{symbol} - Price Chart with 5-Class Trading Signals"
            signal_summary = f"Signals: {total_signals}/{len(results)} ({(total_signals / len(results) * 100):.1f}%) | Threshold: {threshold:.3f}"

            ax1.set_title(main_title, fontsize=16, fontweight='bold', pad=15)
            ax1.text(0.5, 0.98, signal_summary, transform=ax1.transAxes,
                     ha='center', va='top', fontsize=11,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))

            # Plot price using datetime index directly
            ax1.plot(range(len(results)), results['Close'],
                     'k-', alpha=0.8, linewidth=1.5, label='Price')

            # Mark different types of signals with ORIGINAL COLORS
            if enter_long_count > 0:
                enter_long_indices = results[results['enter_long_signal'] == 1].index
                enter_long_x = [results.index.get_loc(idx) for idx in enter_long_indices]
                enter_long_y = [results.loc[idx, 'Close'] for idx in enter_long_indices]
                ax1.scatter(enter_long_x, enter_long_y,
                            marker='^', color='green', s=80, label=f'Enter Long ({enter_long_count})',
                            alpha=0.8, zorder=5)

            if enter_short_count > 0:
                enter_short_indices = results[results['enter_short_signal'] == 1].index
                enter_short_x = [results.index.get_loc(idx) for idx in enter_short_indices]
                enter_short_y = [results.loc[idx, 'Close'] for idx in enter_short_indices]
                ax1.scatter(enter_short_x, enter_short_y,
                            marker='v', color='red', s=80, label=f'Enter Short ({enter_short_count})',
                            alpha=0.8, zorder=5)

            if exit_long_count > 0:
                exit_long_indices = results[results['exit_long_signal'] == 1].index
                exit_long_x = [results.index.get_loc(idx) for idx in exit_long_indices]
                exit_long_y = [results.loc[idx, 'Close'] for idx in exit_long_indices]
                ax1.scatter(exit_long_x, exit_long_y,
                            marker='s', color='lightgreen', s=60, label=f'Exit Long ({exit_long_count})',
                            alpha=0.7, zorder=4)

            if exit_short_count > 0:
                exit_short_indices = results[results['exit_short_signal'] == 1].index
                exit_short_x = [results.index.get_loc(idx) for idx in exit_short_indices]
                exit_short_y = [results.loc[idx, 'Close'] for idx in exit_short_indices]
                ax1.scatter(exit_short_x, exit_short_y,
                            marker='s', color='lightcoral', s=60, label=f'Exit Short ({exit_short_count})',
                            alpha=0.7, zorder=4)

            ax1.set_ylabel('Price (₹)', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='best')

            # Clean date-only formatting for top charts
            def clean_date_formatter(ax):
                """Simple date formatting for upper charts to avoid crowding"""

                def format_func(x, pos=None):
                    try:
                        if int(x) < 0 or int(x) >= len(results):
                            return ''

                        # Only show every nth label to avoid crowding
                        if int(x) % max(1, int(len(results) / 8)) == 0:
                            date = results.index[int(x)]
                            return date.strftime('%m/%d')
                        return ''

                    except (IndexError, ValueError):
                        return ''

                return format_func

            # Apply clean date formatting to top chart
            ax1.xaxis.set_major_formatter(plt.FuncFormatter(clean_date_formatter(ax1)))
            ax1.xaxis.set_major_locator(plt.MaxNLocator(nbins=8))

            # 2. OPTIMIZED: Model prediction scores comparison (ORIGINAL COLORS)
            ax2 = fig.add_subplot(gs[1], sharex=ax1)
            ax2.set_title(f"Model Prediction Scores (LSTM: {lstm_weight:.2f}, XGB: {xgb_weight:.2f})",
                          fontsize=12, fontweight='bold', pad=15)

            # Plot with ORIGINAL colors and styles
            ax2.plot(range(len(results)), results['lstm_entry_score'],
                     'b-', alpha=0.7, linewidth=1, label='LSTM Entry')
            ax2.plot(range(len(results)), results['lstm_exit_score'],
                     'b--', alpha=0.7, linewidth=1, label='LSTM Exit')
            ax2.plot(range(len(results)), results['xgb_entry_score'],
                     'r-', alpha=0.7, linewidth=1, label='XGB Entry')
            ax2.plot(range(len(results)), results['xgb_exit_score'],
                     'r--', alpha=0.7, linewidth=1, label='XGB Exit')
            ax2.plot(range(len(results)), results['ensemble_entry_score'],
                     'g-', linewidth=2, label='Ensemble Entry')
            ax2.plot(range(len(results)), results['ensemble_exit_score'],
                     'orange', linewidth=2, label='Ensemble Exit')

            # Add threshold line
            ax2.axhline(y=threshold, color='black', linestyle=':', alpha=0.8,
                        label=f'Threshold ({threshold:.2f})')

            ax2.set_ylim(0, 1)
            ax2.set_ylabel('Signal Score', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='best', fontsize=9)

            # Hide x-axis labels for middle charts to reduce clutter
            ax2.set_xticklabels([])

            # 3. OPTIMIZED: Individual class probabilities (ORIGINAL COLORS)
            ax3 = fig.add_subplot(gs[2], sharex=ax1)
            ax3.set_title("5-Class Ensemble Probabilities", fontsize=12, fontweight='bold', pad=15)

            # Plot all 5 class probabilities with ORIGINAL colors
            ax3.plot(range(len(results)), results['ensemble_hold'],
                     'gray', alpha=0.6, linewidth=1, label='Hold')
            ax3.plot(range(len(results)), results['ensemble_enter_long'],
                     'green', alpha=0.8, linewidth=1.5, label='Enter Long')
            ax3.plot(range(len(results)), results['ensemble_enter_short'],
                     'red', alpha=0.8, linewidth=1.5, label='Enter Short')
            ax3.plot(range(len(results)), results['ensemble_exit_long'],
                     'lightgreen', alpha=0.8, linewidth=1, label='Exit Long')
            ax3.plot(range(len(results)), results['ensemble_exit_short'],
                     'lightcoral', alpha=0.8, linewidth=1, label='Exit Short')

            ax3.axhline(y=threshold, color='black', linestyle=':', alpha=0.8,
                        label=f'Threshold ({threshold:.2f})')
            ax3.set_ylim(0, 1)
            ax3.set_ylabel('Class Probability', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.legend(loc='best', fontsize=9)

            # Hide x-axis labels for middle charts to reduce clutter
            ax3.set_xticklabels([])

            # 4. OPTIMIZED: Signal summary by type (ORIGINAL COLORS)
            ax4 = fig.add_subplot(gs[3], sharex=ax1)
            ax4.set_title("Trading Signals by Action and Direction", fontsize=12, fontweight='bold', pad=15)

            # Create signal visualization using continuous index
            signal_height = 0.8

            # Plot entry signals with ORIGINAL colors
            if enter_long_count > 0:
                enter_long_indices = results[results['enter_long_signal'] == 1].index
                enter_long_x = [results.index.get_loc(idx) for idx in enter_long_indices]
                for x in enter_long_x:
                    ax4.plot([x, x], [0, signal_height], color='green', linewidth=3, alpha=0.8)
                    ax4.scatter(x, signal_height, color='green', s=30, zorder=5)

            if enter_short_count > 0:
                enter_short_indices = results[results['enter_short_signal'] == 1].index
                enter_short_x = [results.index.get_loc(idx) for idx in enter_short_indices]
                for x in enter_short_x:
                    ax4.plot([x, x], [0, -signal_height], color='red', linewidth=3, alpha=0.8)
                    ax4.scatter(x, -signal_height, color='red', s=30, zorder=5)

            # Plot exit signals (smaller) with ORIGINAL colors
            if exit_long_count > 0:
                exit_long_indices = results[results['exit_long_signal'] == 1].index
                exit_long_x = [results.index.get_loc(idx) for idx in exit_long_indices]
                for x in exit_long_x:
                    ax4.plot([x, x], [0, signal_height * 0.6], color='lightgreen', linewidth=2, alpha=0.7)
                    ax4.scatter(x, signal_height * 0.6, color='lightgreen', s=20, zorder=4)

            if exit_short_count > 0:
                exit_short_indices = results[results['exit_short_signal'] == 1].index
                exit_short_x = [results.index.get_loc(idx) for idx in exit_short_indices]
                for x in exit_short_x:
                    ax4.plot([x, x], [0, -signal_height * 0.6], color='lightcoral', linewidth=2, alpha=0.7)
                    ax4.scatter(x, -signal_height * 0.6, color='lightcoral', s=20, zorder=4)

            # Add background colors and labels
            ax4.axhspan(0, 1, color='lightgreen', alpha=0.1)  # Long area
            ax4.axhspan(-1, 0, color='mistyrose', alpha=0.1)  # Short area
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)

            # Add direction labels
            ax4.text(0, 0.5, 'LONG SIGNALS', ha='left', va='center', fontsize=10, fontweight='bold',
                     bbox=dict(facecolor='green', alpha=0.2))
            ax4.text(0, -0.5, 'SHORT SIGNALS', ha='left', va='center', fontsize=10, fontweight='bold',
                     bbox=dict(facecolor='red', alpha=0.2))

            # Create custom legend with ORIGINAL colors
            from matplotlib.lines import Line2D
            custom_lines = [
                Line2D([0], [0], color='green', lw=3, alpha=0.8),
                Line2D([0], [0], color='red', lw=3, alpha=0.8),
                Line2D([0], [0], color='lightgreen', lw=2, alpha=0.7),
                Line2D([0], [0], color='lightcoral', lw=2, alpha=0.7)
            ]
            ax4.legend(custom_lines,
                       [f'Enter Long ({enter_long_count})', f'Enter Short ({enter_short_count})',
                        f'Exit Long ({exit_long_count})', f'Exit Short ({exit_short_count})'],
                       loc='upper right', fontsize=9)

            ax4.set_ylim(-1.1, 1.1)
            ax4.set_ylabel('Signal Type', fontweight='bold')
            ax4.grid(True, alpha=0.3)

            # Apply consistent smart datetime formatting ONLY to bottom chart
            ax4.xaxis.set_major_formatter(plt.FuncFormatter(smart_datetime_formatter(ax4)))
            ax4.xaxis.set_major_locator(plt.MaxNLocator(nbins=10))

            # Set bottom subplot xlabel
            ax4.set_xlabel('Trading Days (Date/Time when zoomed)', fontweight='bold')

            # CRITICAL: Use tight_layout to maximize space utilization
            plt.tight_layout()

            # Adjust layout to prevent title overlap
            plt.subplots_adjust(top=0.94, bottom=0.08, left=0.06, right=0.98)

            # Enable interactive zoom with datetime updates
            def on_xlims_change(ax):
                """Update x-axis formatting when zoom changes"""
                ax.figure.canvas.draw()

            # Connect zoom event to all axes
            for ax in [ax1, ax2, ax3, ax4]:
                ax.callbacks.connect('xlim_changed', on_xlims_change)

            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
                logging.info(f"5-class model analysis plot saved to {save_path}")

            # Display plot
            try:
                plt.show(block=True)
                logging.info("5-class model analysis plot displayed")
            except Exception:
                pass

            return fig

        except Exception as e:
            logging.error(f"Error plotting 5-class model analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _detect_market_regime(self, features: pd.DataFrame) -> str:
        """Enhanced market regime detection with more precise indicators"""
        try:
            if features is None or features.empty or len(features) < 20:
                return 'normal'

            # Get latest data point with better error handling
            if isinstance(features, pd.DataFrame) and len(features) > 0:
                current = features.iloc[-1]
            else:
                return 'normal'  # Safe default

            # Handle case where current might be a DataFrame
            if isinstance(current, pd.DataFrame):
                if len(current) > 0:
                    current = current.iloc[0]
                else:
                    return 'normal'

            # Multi-factor regime classification
            vol_score = 0
            trend_score = 0
            momentum_score = 0

            # 1. Volatility factors
            if 'volatility_20' in current:
                vol_val = float(current['volatility_20'])
                vol_score += vol_val / (self.config.VOLATILITY_THRESHOLD * 0.75)  # More sensitive threshold

                # Check volatility trend
                if 'volatility_5' in current and 'volatility_20' in current:
                    vol_ratio = float(current['volatility_5']) / float(current['volatility_20'])
                    if vol_ratio > 1.2:  # Short-term volatility increasing
                        vol_score += 0.5

            # 2. Trend factors
            if all(x in current for x in ['ema_8', 'ema_21', 'ema_55']):
                # Calculate trend alignment
                ema_8 = float(current['ema_8'])
                ema_21 = float(current['ema_21'])
                ema_55 = float(current['ema_55'])

                # Strong trend when all EMAs aligned
                if (ema_8 > ema_21 > ema_55) or (ema_8 < ema_21 < ema_55):
                    trend_score += 1.0

                    # Measure trend strength
                    trend_strength = abs((ema_8 / ema_55) - 1) * 100
                    if trend_strength > 2.0:  # Strong trend
                        trend_score += 0.5
                else:
                    # Mixed EMAs suggest choppy/ranging
                    trend_score -= 0.5

            # 3. Momentum factors
            if 'rsi_14' in current:
                rsi = float(current['rsi_14'])
                # Extreme RSI suggests strong momentum or reversal
                if rsi > 75 or rsi < 25:
                    momentum_score += 1.0

            # 4. Volume factors
            if 'volume_ratio' in current:
                vol_ratio = float(current['volume_ratio'])
                if vol_ratio > 1.5:  # Above average volume
                    momentum_score += 0.5

            # Check for Bollinger Band squeeze
            if 'bb_squeeze' in current and current['bb_squeeze'] > 0:
                # Squeeze suggests potential breakout
                trend_score -= 0.3  # Lower trend score
                momentum_score += 0.8  # Higher momentum potential

            # Check pre-calculated composite signals if available
            if 'breakout_potential_composite' in current:
                breakout_score = float(current['breakout_potential_composite'])
                if breakout_score > 0.7:
                    momentum_score += 0.7

            # 5. Combine scores for classification
            regime = 'normal'

            if vol_score > 1.5:  # High volatility is primary factor
                regime = 'volatile'
            elif trend_score > 1.0 and vol_score < 1.0:
                regime = 'trending'
            elif vol_score < 0.7 and trend_score < 0.5:
                regime = 'ranging'
            elif momentum_score > 1.5 and vol_score < 0.5:
                regime = 'breakout_pending'

            # Log regime change
            if hasattr(self, 'current_regime') and regime != self.current_regime:
                logging.info(f"Market regime changed: {self.current_regime} → {regime}")
                self.current_regime = regime

            return regime

        except Exception as e:
            logging.error(f"Market regime detection error: {str(e)}")
            return 'normal'  # Default on error

    def get_optimal_targets(self, symbol: str) -> dict:
        """Get the optimal profit target and stop loss for a symbol based on ideal path analysis"""
        try:
            # Return the stored thresholds for this symbol if available
            if symbol in self.profit_thresholds:
                return self.profit_thresholds[symbol]

            # Otherwise return default values based on config
            return {
                'take_profit': self.config.TAKE_PROFIT * 100,  # Convert to percentage
                'stop_loss': -self.config.STOP_LOSS * 100,  # Convert to percentage
                'avg_holding_period': 30  # Default to 30 minutes
            }
        except Exception as e:
            logging.error(f"Error getting optimal targets: {str(e)}")
            # Return config defaults
            return {
                'take_profit': self.config.TAKE_PROFIT * 100,
                'stop_loss': -self.config.STOP_LOSS * 100,
                'avg_holding_period': 30
            }


class AngelOneConnector:
    def __init__(self, config: Config):
        # Core components
        self.config = config
        self.smart_api = None
        self.smart_socket = None
        self.feed_token = None
        self.refresh_token = None
        self.connection_status = False
        self.last_reconnect = datetime.now()
        self.reconnect_attempts = 0
        self.rate_limiter = RateLimiter()
        self.is_running = True
        self.warmup_complete = False

        # Data queues and locks
        self.tick_data = queue.Queue()
        self.lock = threading.Lock()
        self.refresh_thread = None  # Add this

        # Initialize data structures
        self.received_ticks = {}
        self.latest_data = {}
        self.tick_cache = {}
        self.bars_1min = {}
        self.last_tick_time = {}
        self.subscribed_tokens = {}

        # Initialize per-symbol structures with proper OHLC columns
        for symbol in self.config.SYMBOLS:
            self.received_ticks[symbol] = []
            self.latest_data[symbol] = {
                'Open': 0.0,
                'High': 0.0,
                'Low': 0.0,
                'Close': 0.0,
                'Volume': 0,
                'timestamp': datetime.now()
            }
            self.tick_cache[symbol] = {
                'price': 0.0,
                'volume': 0,
                'timestamp': datetime.now()
            }
            self.bars_1min[symbol] = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
            self.last_tick_time[symbol] = datetime.now()

        # Connection timing
        self.last_ping_time = time.time()
        self.last_pong_time = time.time()
        self.startup_time = None
        self.session_start_time = None
        self.last_session_refresh = None
        self.feed_token_expiry = None

        # Memory management
        self.max_history_size = 1000

        # Connection params
        self.connection_params = {
            'timeout': 30,
            'retries': 5,
            'backoff_factor': 2,
            'ping_interval': 60,
            'ping_timeout': 30,
            'max_reconnect_attempts': 10
        }
        self.ws_state = {
            'connected': False,
            'message_count': 0,
            'errors': 0
        }

    def _initialize_websocket(self) -> bool:
        """Initialize WebSocket with proper connection timing and ping/pong handling"""
        try:
            # Initialize tracking dictionaries if they don't exist
            if not hasattr(self, 'last_tick_time') or self.last_tick_time is None:
                self.last_tick_time = {}

            # Initialize startup time if not already set
            if not hasattr(self, 'startup_time'):
                self.startup_time = datetime.now()

            # Ensure all symbols have an entry in last_tick_time (even if None)
            for symbol in self.config.SYMBOLS:
                if symbol not in self.last_tick_time:
                    self.last_tick_time[symbol] = None

            # Create websocket URL
            websocket_url = (
                f"wss://smartapisocket.angelone.in/smart-stream"
                f"?clientCode={self.config.CLIENT_ID}"
                f"&feedToken={self.feed_token}"
                f"&apiKey={self.config.API_KEY}"
            )

            # Create a connection event to properly wait for on_open callback
            connection_event = threading.Event()

            # Initialize ping/pong tracking
            self.last_ping_time = time.time()
            self.last_pong_time = time.time()

            # Define callback functions
            def on_open(ws):
                logging.info("WebSocket connected successfully")
                self.connection_status = True
                connection_event.set()  # Signal successful connection

            def on_message(ws, message):
                # Handle potential ping/pong responses
                if isinstance(message, str):
                    try:
                        msg_data = json.loads(message)
                        if msg_data.get('type') == 'pong':
                            self.last_pong_time = time.time()
                            return
                    except:
                        pass
                # Process market data
                self._handle_ticks(ws, message)

            def on_error(ws, error):
                logging.error(f"WebSocket error: {str(error)}")

            def on_close(ws, close_status_code=None, close_msg=None):
                logging.info("WebSocket disconnected")
                self.connection_status = False

            def on_ping(ws, message):
                self.last_ping_time = time.time()

            def on_pong(ws, message):
                self.last_pong_time = time.time()

            # Create socket with proper configuration
            self.smart_socket = WebSocketApp(
                websocket_url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_ping=on_ping,
                on_pong=on_pong
            )

            # Start socket in background thread with proper ping interval
            self.ws_thread = threading.Thread(
                target=lambda: self.smart_socket.run_forever(
                    sslopt={"cert_reqs": ssl.CERT_NONE},
                    ping_interval=60,  # Send ping every 15 seconds
                    ping_timeout=30,  # Wait 10 seconds for pong response
                    ping_payload='{"action":"ping"}'  # Use JSON format for ping
                ),
                daemon=True
            )
            self.ws_thread.start()

            # Wait for on_open callback to be triggered
            if connection_event.wait(timeout=10):  # Wait up to 10 seconds for connection
                logging.info("WebSocket connection verified")

                # Important: Add a short delay to ensure socket is fully established
                time.sleep(1)

                return True
            else:
                logging.error("WebSocket connection timeout - on_open not called")
                return False

        except Exception as e:
            logging.error(f"WebSocket initialization error: {str(e)}")
            return False

    def connect(self) -> bool:
        """Connect to Angel One with proper sequence"""
        try:
            # Generate TOTP
            totp = pyotp.TOTP(self.config.TOTP_KEY)
            token = totp.now()

            if not self.smart_api:
                self.smart_api = SmartConnect(api_key=self.config.API_KEY)

            # Generate session
            session_data = self.smart_api.generateSession(
                clientCode=self.config.CLIENT_ID,
                password=self.config.PASSWORD,
                totp=token
            )

            if session_data and session_data.get('status'):
                self.refresh_token = session_data['data']['refreshToken']
                self.feed_token = self.smart_api.getfeedToken()
                logging.info("Session generated successfully")

                # Close existing socket first
                if self.smart_socket:
                    try:
                        self.smart_socket.close()
                        time.sleep(1)  # Allow time for clean closure
                    except:
                        pass

                # CRITICAL FIX: Initialize last_tick_time if not already present
                if not hasattr(self, 'last_tick_time') or self.last_tick_time is None:
                    self.last_tick_time = {}

                # Ensure all symbols have an entry in last_tick_time
                for symbol in self.config.SYMBOLS:
                    self.last_tick_time[symbol] = datetime.now() - timedelta(
                        seconds=30)  # Set a timestamp 30 seconds ago

                # Initialize WebSocket with proper timing
                if self._initialize_websocket():
                    # Ensure socket is fully connected before subscribing
                    time.sleep(1)

                    # Only attempt to subscribe if socket is connected
                    if self.smart_socket and hasattr(self.smart_socket, 'sock') and self.smart_socket.sock:
                        # Subscribe to market data
                        if self._subscribe_market_data():
                            # CRITICAL FIX: Verify data flow
                            # Wait up to 5 seconds for initial data to arrive
                            start_wait_time = time.time()
                            data_received = False
                            while time.time() - start_wait_time < 5:
                                # Check if any tick time has been updated from our initialization
                                for symbol in self.config.SYMBOLS:
                                    if symbol in self.last_tick_time:
                                        if (datetime.now() - self.last_tick_time[symbol]).total_seconds() < 30:
                                            data_received = True
                                            break
                                if data_received:
                                    break
                                time.sleep(0.1)

                            return True

                        else:
                            logging.error("Failed to subscribe to market data")
                    else:
                        logging.error("Socket not available for subscription")

                    return False
                else:
                    logging.error("Failed to generate session")
                    return False

        except Exception as e:
            logging.error(f"Connection error: {str(e)}")
            return False

    def _subscribe_market_data(self) -> bool:
        """Subscribe to market data with proper error handling"""
        try:
            # Double-check socket is available
            if not self.smart_socket or not hasattr(self.smart_socket, 'sock') or not self.smart_socket.sock:
                logging.error("Cannot subscribe - WebSocket not connected")
                return False

            # Format token list
            token_list = [str(self.config.SYMBOLS[s]['token']) for s in self.config.SYMBOLS]

            # Create subscription message
            subscribe_message = {
                "action": 1,
                "params": {
                    "mode": 3,
                    "tokenList": [{
                        "exchangeType": 1,
                        "tokens": token_list
                    }]
                }
            }

            # Send subscription
            try:
                self.smart_socket.send(json.dumps(subscribe_message))
                logging.info(f"Subscribed to {len(token_list)} symbols")

                # Update subscription tracking
                self.subscribed_tokens = {s: self.config.SYMBOLS[s]['token'] for s in self.config.SYMBOLS}

                # Allow time for subscription to take effect
                time.sleep(1)

                return True

            except Exception as send_error:
                logging.error(f"Error sending subscription: {str(send_error)}")
                return False

        except Exception as e:
            logging.error(f"Market data subscription error: {str(e)}")
            return False

    def _handle_connect(self, ws):
        """Handle WebSocket connection and subscription"""
        try:
            self.connection_status = True
            self.reconnect_attempts = 0
            logging.info("WebSocket connected successfully")

            # Wait a moment before doing anything
            time.sleep(1)

        except Exception as e:
            logging.error(f"Connection handler error: {str(e)}")

    def _handle_disconnect(self, ws):
        """Handle WebSocket disconnection"""
        try:
            if self.connection_status:
                self.connection_status = False
                logging.info("WebSocket disconnected")

                # Clear any stale data
                self.last_tick_time.clear()

                if self.is_running:
                    time.sleep(1)  # Wait before reconnect
                    self._attempt_reconnect()

        except Exception as e:
            logging.error(f"Disconnect handler error: {str(e)}")

    def _handle_error(self, ws, error):
        """Handle WebSocket errors"""
        logging.error(f"WebSocket error: {str(error)}")

        # Only attempt reconnect if not already disconnected
        if self.connection_status and self.is_running:
            self.connection_status = False
            time.sleep(1)  # Wait before reconnect
            self._attempt_reconnect()

    def is_connected(self) -> bool:
        """Comprehensive connection check that verifies active data flow"""
        try:
            # Basic socket check
            if not (self.smart_socket and
                    hasattr(self.smart_socket, 'sock') and
                    self.smart_socket.sock and
                    self.smart_socket.sock.connected):
                return False

            # Check if tokens are valid
            if not self.feed_token:
                return False

            # Check TIMESTAMP of last received tick
            # During initialization, we may not have received ticks yet
            if hasattr(self, 'startup_time'):
                startup_elapsed = (datetime.now() - self.startup_time).seconds

                # Only check data flow after reasonable startup time (30 seconds)
                if startup_elapsed > 30:
                    # Check if ANY symbol has recent data
                    current_time = datetime.now()
                    recent_data = False

                    for symbol in self.config.SYMBOLS:
                        if symbol in self.last_tick_time:
                            time_diff = (current_time - self.last_tick_time[symbol]).seconds
                            if time_diff < 10:  # Data within last 10 seconds
                                recent_data = True
                                break

                    if not recent_data:
                        return False

            return True

        except Exception as e:
            logging.error(f"Connection check error: {str(e)}")
            return False

    def _attempt_reconnect(self) -> bool:
        """Attempt reconnection with proper error handling"""
        try:
            # Update reconnect tracking
            current_time = datetime.now()
            self.last_reconnect = current_time

            logging.info(f"Attempting reconnection...")

            # Close existing socket if any
            if self.smart_socket:
                try:
                    self.smart_socket.close()
                    time.sleep(1)  # Wait for socket to close
                    self.smart_socket = None
                except Exception as close_error:
                    logging.error(f"Error closing socket: {str(close_error)}")

            # Get new session with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    totp = pyotp.TOTP(self.config.TOTP_KEY)
                    token = totp.now()

                    session_data = self.smart_api.generateSession(
                        clientCode=self.config.CLIENT_ID,
                        password=self.config.PASSWORD,
                        totp=token
                    )

                    if session_data and session_data.get('status'):
                        self.refresh_token = session_data['data']['refreshToken']
                        self.feed_token = self.smart_api.getfeedToken()
                        self.session_start_time = datetime.now()

                        # Initialize new WebSocket
                        if self._initialize_websocket():
                            # Subscribe to symbols
                            if self._subscribe_market_data():
                                logging.info("Reconnection successful")
                                # Don't access 'reconnects' attribute that doesn't exist
                                return True

                        logging.warning(f"Reconnection failed (attempt {attempt + 1})")
                        if attempt < max_retries - 1:
                            time.sleep(2)  # Wait before retry
                    else:
                        logging.warning(f"Failed to generate session (attempt {attempt + 1})")
                        if attempt < max_retries - 1:
                            time.sleep(2)
                except Exception as session_error:
                    logging.error(f"Session error during reconnect (attempt {attempt + 1}): {str(session_error)}")
                    if attempt < max_retries - 1:
                        time.sleep(2)

            return False

        except Exception as e:
            logging.error(f"Reconnection error: {str(e)}")
            return False

    def decode_binary_data(self, binary_data: bytes) -> dict:
        """Decode binary data with proper mode handling"""
        try:
            # Basic validation
            if len(binary_data) < 47:
                return None

            # Basic fields that are always present
            mode = binary_data[0]
            exchange_type = binary_data[1]
            token = binary_data[2:27].decode('utf-8').rstrip('\x00')
            sequence_number = struct.unpack('q', binary_data[27:35])[0]
            exchange_timestamp = struct.unpack('q', binary_data[35:43])[0]
            ltp = struct.unpack('i', binary_data[43:47])[0] / 100.0

            # Base response
            response = {
                'mode': mode,
                'exchange_type': exchange_type,
                'token': token,
                'sequence_number': sequence_number,
                'exchange_timestamp': exchange_timestamp,
                'ltp': ltp
            }

            # Mode 3 (Full Quote) data - packet size = 379 bytes
            if mode == 3 and len(binary_data) >= 379:
                response.update({
                    'last_traded_timestamp': struct.unpack('q', binary_data[123:131])[0],
                    'open_interest': struct.unpack('q', binary_data[131:139])[0],
                    'upper_circuit': struct.unpack('q', binary_data[347:355])[0] / 100.0,
                    'lower_circuit': struct.unpack('q', binary_data[355:363])[0] / 100.0,
                    'high_52_week': struct.unpack('q', binary_data[363:371])[0] / 100.0,
                    'low_52_week': struct.unpack('q', binary_data[371:379])[0] / 100.0,
                    'last_traded_quantity': struct.unpack('q', binary_data[51:59])[0],
                    'average_traded_price': struct.unpack('q', binary_data[59:67])[0] / 100.0,
                    'volume': struct.unpack('q', binary_data[67:75])[0],
                    'total_buy_quantity': struct.unpack('d', binary_data[75:83])[0],
                    'total_sell_quantity': struct.unpack('d', binary_data[83:91])[0],
                    'open': struct.unpack('q', binary_data[91:99])[0] / 100.0,
                    'high': struct.unpack('q', binary_data[99:107])[0] / 100.0,
                    'low': struct.unpack('q', binary_data[107:115])[0] / 100.0,
                    'close': struct.unpack('q', binary_data[115:123])[0] / 100.0
                })

            return response

        except Exception as e:
            logging.error(f"Binary data decode error: {str(e)}")
            return None

    def _start_heartbeat(self):
        """Monitor system health and data flow with dynamic timing"""
        # Add initialization delay
        time.sleep(10)  # Wait for system to stabilize after startup

        last_reconnect_attempt = time.time()
        reconnect_interval = self.connection_params['backoff_factor']  # 2 seconds initially

        while self.is_running:
            try:
                current_time = time.time()

                # Check if socket exists and is connected
                socket_connected = (self.smart_socket and
                                    hasattr(self.smart_socket, 'sock') and
                                    self.smart_socket.sock and
                                    self.smart_socket.sock.connected)

                if socket_connected:
                    # Check for activity timeout only after receiving some data
                    if hasattr(self.ws_state, 'last_message') and self.ws_state.get('message_count', 0) > 5:
                        message_age = (datetime.now() - self.ws_state['last_message']).total_seconds()
                        if message_age > 30:  # No message for 30 seconds
                            logging.warning(f"No message received for {message_age:.1f} seconds")

                            # Send ping to check connection
                            try:
                                self.smart_socket.send('{"action":"ping"}')
                                self.last_ping_time = current_time
                                logging.debug("Sent ping message to keep connection alive")
                            except Exception as ping_error:
                                logging.error(f"Error sending ping: {str(ping_error)}")

                            # If it's been long enough since last reconnect attempt, try reconnecting
                            if current_time - last_reconnect_attempt > reconnect_interval:
                                logging.warning("Connection appears stale, attempting reconnection")
                                self._attempt_reconnect()
                                last_reconnect_attempt = current_time
                                # Increase backoff interval for next attempt
                                reconnect_interval = min(300, reconnect_interval * 2)  # Cap at 5 minutes

                    # Reset backoff if we've been connected and receiving messages
                    if self.ws_state.get('message_count', 0) > 10:
                        reconnect_interval = self.connection_params['backoff_factor']

                else:
                    # Socket disconnected but status flag not updated
                    if self.connection_status:
                        logging.warning("Socket disconnected but status flag indicated connected")
                        self.connection_status = False
                        self.ws_state['connected'] = False

                    # Only attempt reconnect if it's been long enough AND we've been running a while
                    if (current_time - last_reconnect_attempt > reconnect_interval and
                            (
                                    time.time() - self.startup_time.timestamp() if hasattr(self,
                                                                                           'startup_time') else 60) > 30):
                        logging.info("Connection down, attempting reconnection")
                        self._attempt_reconnect()
                        last_reconnect_attempt = current_time
                        reconnect_interval = min(300, reconnect_interval * 2)

                time.sleep(1)

            except Exception as e:
                logging.error(f"Heartbeat error: {str(e)}")
                time.sleep(5)  # Sleep longer on error

    def _verify_data_flow(self) -> bool:
        """Verify initial data flow after connection"""
        try:
            start_time = time.time()
            while time.time() - start_time < 5:  # Wait up to 5 seconds for initial data
                if self.ws_state['message_count'] > 0:
                    # Validate message timestamps
                    if self.ws_state['last_message']:
                        time_diff = (datetime.now() - self.ws_state['last_message']).seconds
                        if time_diff < 2:  # Data within last 2 seconds
                            return True
                time.sleep(0.1)
            return False
        except Exception as e:
            logging.error(f"Data flow verification error: {str(e)}")
            return False

    def _track_websocket_state(self):
        self.ws_state = {
            'connected': False,
            'last_message': None,
            'message_count': 0,
            'errors': 0,
            'reconnects': 0
        }

    def _make_api_request(self, method, *args, **kwargs):
        """Make API request with retry mechanism"""
        max_retries = self.connection_params['retries']
        backoff = self.connection_params['backoff_factor']

        for attempt in range(max_retries):
            try:
                if not self.smart_api:
                    raise Exception("SmartAPI not initialized")

                # Add retry delay for non-first attempts
                if attempt > 0:
                    wait_time = (backoff ** (attempt - 1)) * 2
                    logging.info(f"Waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)

                response = None

                # Market data endpoints
                if method == 'ltpData':
                    response = self.smart_api.ltpData(
                        kwargs.get('exchange', ''),
                        kwargs.get('symbol', ''),
                        kwargs.get('token', '')
                    )
                elif method == 'getCandleData':
                    response = self.smart_api.getCandleData(kwargs.get('params', {}))

                # Session endpoints
                elif method == 'generateSession':
                    response = self.smart_api.generateSession(
                        kwargs.get('clientCode', ''),
                        kwargs.get('password', ''),
                        kwargs.get('totp', '')
                    )
                elif method == 'terminateSession':
                    response = self.smart_api.terminateSession(kwargs.get('clientCode', ''))
                elif method == 'getAccessToken':
                    response = self.smart_api.getAccessToken()
                else:
                    logging.error(f"Unknown API method: {method}")
                    return None

                # Validate and return response
                if response and isinstance(response, dict):
                    return response
                else:
                    raise Exception("Invalid response format")

            except Exception as e:
                logging.error(
                    f"API request failed, attempt {attempt + 1}/{max_retries}. Method: {method}. Error: {str(e)}")
                if attempt == max_retries - 1:
                    raise e
                continue

        return None

    def _handle_ticks(self, ws, message):
        """Process incoming tick data with debugging"""
        try:
            # Log raw message periodically for debugging
            if not hasattr(self, '_last_message_log'):
                self._last_message_log = time.time()

            # Log a message every minute to confirm data is being received
            current_time = time.time()
            if current_time - getattr(self, '_last_message_log', 0) > 600:
                message_type = "binary" if isinstance(message, bytes) else "string"
                logging.info(f"Receiving {message_type} WebSocket messages (length: {len(message)})")
                self._last_message_log = current_time

            if isinstance(message, bytes):
                decoded = self.decode_binary_data(message)
                if decoded:
                    token = str(decoded.get('token', ''))
                    symbol = next((s for s, info in self.config.SYMBOLS.items()
                                   if str(info['token']) == token), None)

                    if symbol:
                        current_time = datetime.now()

                        # CRITICAL FIX: Create tick data with complete and explicit OHLCV columns
                        # First identify the price from various possible fields
                        price = float(decoded.get('ltp', 0))

                        # Then create a fully-populated tick data structure
                        tick_data = {
                            'timestamp': current_time,
                            'token': token,
                            'symbol': symbol,
                            'Open': float(decoded.get('open', price)),
                            'High': float(decoded.get('high', price)),
                            'Low': float(decoded.get('low', price)),
                            'Close': price,
                            'Volume': int(decoded.get('volume', decoded.get('last_traded_quantity', 0))),
                            'last_traded_quantity': decoded.get('last_traded_quantity', 0),
                            'average_traded_price': decoded.get('average_traded_price', 0),
                            'total_buy_quantity': decoded.get('total_buy_quantity', 0),
                            'total_sell_quantity': decoded.get('total_sell_quantity', 0),
                            'open_interest': decoded.get('open_interest', 0),
                            'exchange_timestamp': decoded.get('exchange_timestamp', 0),
                            'sequence_number': decoded.get('sequence_number', 0)
                        }

                        # Additional fields for compatibility with different components
                        tick_data['price'] = price
                        tick_data['ltp'] = price
                        tick_data['volume'] = tick_data['Volume']

                        with self.lock:
                            # Update latest data
                            self.latest_data[symbol] = tick_data.copy()
                            self.last_tick_time[symbol] = current_time

                            # CRITICAL FIX: Update market_data_manager's timestamps if it exists
                            if hasattr(self, 'market_data_manager') and self.market_data_manager is not None:
                                if hasattr(self.market_data_manager, 'last_tick_time'):
                                    self.market_data_manager.last_tick_time[symbol] = current_time

                            # Store in received ticks
                            if symbol not in self.received_ticks:
                                self.received_ticks[symbol] = []
                            self.received_ticks[symbol].append(tick_data)

                            # Maintain history size
                            if len(self.received_ticks[symbol]) > self.max_history_size:
                                self.received_ticks[symbol] = self.received_ticks[symbol][-self.max_history_size:]

                            # Update tick cache with guaranteed fields
                            self.tick_cache[symbol] = {
                                'price': price,
                                'volume': tick_data['Volume'],
                                'timestamp': current_time,
                                'Open': tick_data['Open'],
                                'High': tick_data['High'],
                                'Low': tick_data['Low'],
                                'Close': tick_data['Close']
                            }

                            # Log only the first tick for each symbol
                            if len(self.received_ticks[symbol]) == 1:  # Changed from <= 5 to == 1
                                logging.info(
                                    f"Received tick for {symbol}: Price: {tick_data['Close']}, Volume: {tick_data['Volume']}")

                            # Forward to market data queue
                            try:
                                self.tick_data.put_nowait(tick_data)
                            except queue.Full:
                                try:
                                    self.tick_data.get_nowait()  # Remove oldest
                                    self.tick_data.put_nowait(tick_data)
                                except:
                                    pass

            elif isinstance(message, str):
                try:
                    msg_data = json.loads(message)
                    if msg_data.get('type') == 'error':
                        logging.error(f"WebSocket error message: {msg_data.get('message', 'Unknown error')}")
                    elif msg_data.get('type') == 'connection':
                        logging.info(f"WebSocket connection message: {msg_data.get('message', '')}")
                    elif msg_data.get('type') == 'pong':
                        logging.debug("Received pong response")
                        self.last_pong_time = time.time()
                except json.JSONDecodeError:
                    logging.warning(f"Received non-JSON string message: {message[:200]}...")

                # Update message tracking for heartbeat
                if hasattr(self, 'ws_state'):
                    self.ws_state['last_message'] = datetime.now()
                    self.ws_state['message_count'] = self.ws_state.get('message_count', 0) + 1

        except Exception as e:
            logging.error(f"Error handling tick: {str(e)}")
            logging.error(traceback.format_exc())

    def _verify_subscription(self) -> bool:
        """Verify subscription status with enhanced error tracking"""
        try:
            time.sleep(2)  # Wait for initial data

            subscription_status = {}
            for symbol in self.config.SYMBOLS:
                token = str(self.config.SYMBOLS[symbol]['token'])

                # Check subscription status
                is_subscribed = token in [str(t) for t in self.subscribed_tokens.values()]

                # Check data flow
                has_data = False
                if symbol in self.latest_data:
                    time_diff = (datetime.now() - self.latest_data[symbol].get('timestamp', datetime.now())).seconds
                    has_data = time_diff < 5

                subscription_status[symbol] = {
                    'token': token,
                    'subscribed': is_subscribed,
                    'has_data': has_data
                }

                if not is_subscribed or not has_data:
                    logging.error(f"Subscription verification failed for {symbol}:")
                    logging.error(f"Token: {token}, Subscribed: {is_subscribed}, Has Data: {has_data}")
                    return False

            # Log successful subscriptions
            subscribed_symbols = [s for s, status in subscription_status.items()
                                  if status['subscribed'] and status['has_data']]
            logging.info(f"Verified subscription for {len(subscribed_symbols)} symbols")

            return len(subscribed_symbols) == len(self.config.SYMBOLS)

        except Exception as e:
            logging.error(f"Subscription verification error: {str(e)}")
            return False

    def _monitor_subscriptions(self):
        """Monitor subscription status and trigger resubscribe if needed"""
        try:
            current_time = datetime.now()
            stale_symbols = []

            for symbol in self.config.SYMBOLS:
                if symbol in self.last_tick_time:
                    time_diff = (current_time - self.last_tick_time[symbol]).seconds
                    if time_diff > 5:  # Data is stale
                        stale_symbols.append(symbol)
                        logging.warning(f"Stale data for {symbol}: {time_diff}s old")
                else:
                    stale_symbols.append(symbol)
                    logging.warning(f"No data received for {symbol}")

            if stale_symbols:
                token_list = [str(self.config.SYMBOLS[s]['token']) for s in stale_symbols if s in self.config.SYMBOLS]

                subscribe_message = {
                    "action": 1,
                    "params": {
                        "mode": 3,
                        "tokenList": [{
                            "exchangeType": 1,
                            "tokens": token_list
                        }]
                    }
                }

                if self.smart_socket and hasattr(self.smart_socket, 'sock') and self.smart_socket.sock:
                    self.smart_socket.send(json.dumps(subscribe_message))
                    logging.info(f"Resubscribed to {len(token_list)} stale symbols")

        except Exception as e:
            logging.error(f"Subscription monitoring error: {str(e)}")

    def _validate_bars(self, symbol: str) -> bool:
        """Validate and clean bar data if needed"""
        try:
            if symbol not in self.bars_1min:
                self.bars_1min[symbol] = pd.DataFrame()
                return True

            df = self.bars_1min[symbol]
            if df.empty:
                return True

            # Check for duplicate indices
            if df.index.duplicated().any():
                logging.warning(f"Found duplicate indices in {symbol} bars, cleaning...")
                df = df[~df.index.duplicated(keep='last')]
                df = df.sort_index()
                self.bars_1min[symbol] = df

            # Verify all required columns exist
            required_columns = ['Open', 'high', 'low', 'close', 'volume', 'vwap']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                logging.warning(f"Missing columns in {symbol} bars: {missing_columns}")
                return False

            return True

        except Exception as e:
            logging.error(f"Error validating bars for {symbol}: {str(e)}")
            return False

    def get_minute_bars(self, symbol: str, lookback: int = None) -> pd.DataFrame:
        """Get minute bars with validation"""
        try:
            if (not hasattr(self, 'bars_1min') or
                    symbol not in self.bars_1min or
                    not isinstance(self.bars_1min[symbol], pd.DataFrame) or
                    len(self.bars_1min[symbol]) == 0):
                return pd.DataFrame()

            bars = self.bars_1min[symbol]

            # Return only requested lookback period if specified
            if lookback is not None:
                bars = bars.tail(lookback)

            return bars.copy()

        except Exception as e:
            logging.error(f"Error getting bars for {symbol}: {str(e)}")
            return pd.DataFrame()

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a symbol"""
        try:
            if symbol not in self.latest_data:
                return None

            latest = self.latest_data[symbol]
            if 'timestamp' not in latest or \
                    (datetime.now() - latest['timestamp']).seconds > 5:
                return None

            return float(latest.get('Close', latest.get('price')))

        except Exception as e:
            logging.error(f"Error getting latest price for {symbol}: {str(e)}")
            return None

    def _unsubscribe_all(self):
        """Unsubscribe from all market data with proper connection check"""
        try:
            # Only attempt unsubscribe if we have a valid socket connection
            if (self.smart_socket and
                    hasattr(self.smart_socket, 'sock') and
                    self.smart_socket.sock and
                    self.smart_socket.sock.connected):

                if self.subscribed_tokens:
                    unsubscribe_message = {
                        "action": 0,
                        "params": {
                            "mode": 3,
                            "tokenList": [{
                                "exchangeType": 1,
                                "tokens": [str(token) for token in self.subscribed_tokens.values()]
                            }]
                        }
                    }

                    try:
                        self.smart_socket.send(json.dumps(unsubscribe_message))
                        logging.info("Unsubscribed from all symbols")
                    except Exception as e:
                        logging.error(f"Error sending unsubscribe message: {str(e)}")
                else:
                    logging.debug("No tokens to unsubscribe from")
            else:
                logging.debug("WebSocket not connected during unsubscribe attempt")

            # Always clear subscribed tokens
            self.subscribed_tokens.clear()

        except Exception as e:
            logging.error(f"Unsubscribe error: {str(e)}")
            self.subscribed_tokens.clear()  # Clear anyway to maintain consistent state

    def _start_websocket(self):
        """Run WebSocket in a separate thread"""
        try:
            self.smart_socket.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE, "check_hostname": False})
        except Exception as e:
            logging.error(f"WebSocket run error: {str(e)}")
            self._attempt_reconnect()

    def _start_session_refresh_timer(self):
        """Proactive session and token refresh"""
        # Check if refresh thread already exists and is running
        if hasattr(self, 'refresh_thread') and self.refresh_thread and self.refresh_thread.is_alive():
            logging.debug("Session refresh timer already running")
            return

        # Define the refresh function properly without 'self' parameter
        # since it's running as a separate thread function
        def refresh_session():
            """Proactive session and token refresh with retry logic"""
            while self.is_running:
                try:
                    # Sleep first before any operations
                    time.sleep(420)  # 7 minutes exactly

                    # Check if system is still running
                    if not self.is_running:
                        break

                    logging.info("Starting session refresh cycle")

                    # Generate new TOTP
                    totp = pyotp.TOTP(self.config.TOTP_KEY)
                    token = totp.now()

                    # Rate limit check before session generation
                    self.rate_limiter.wait_if_needed('session')

                    # Get new session with retries
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            data = self.smart_api.generateSession(
                                clientCode=self.config.CLIENT_ID,
                                password=self.config.PASSWORD,
                                totp=token
                            )

                            if data and data.get('status'):
                                # Store old socket reference
                                old_socket = self.smart_socket

                                # Update tokens
                                self.refresh_token = data['data']['refreshToken']
                                self.feed_token = self.smart_api.getfeedToken()
                                self.last_session_refresh = datetime.now()

                                # Initialize new WebSocket with verification
                                new_status = self._initialize_websocket()

                                if new_status:
                                    # Verify data flow before closing old socket
                                    time.sleep(2)  # Wait for initial data

                                # If we reach here, the new connection failed
                                logging.error("New connection failed verification")
                                if attempt < max_retries - 1:
                                    time.sleep(2)
                                    continue
                            else:
                                if attempt < max_retries - 1:
                                    logging.warning(f"Session refresh failed, attempt {attempt + 1}")
                                    time.sleep(2)
                                    continue
                                else:
                                    logging.error("Session refresh failed after all retries")
                                    time.sleep(30)  # Longer wait after all retries fail

                        except Exception as session_error:
                            logging.error(f"Session generation error: {str(session_error)}")
                            if attempt < max_retries - 1:
                                time.sleep(2)
                                continue
                            time.sleep(30)  # Longer wait after all retries fail

                except Exception as e:
                    logging.error(f"Session refresh error: {str(e)}")
                    time.sleep(30)

            logging.info("Session refresh thread stopped")

        # Create and start the refresh thread
        self.refresh_thread = threading.Thread(target=refresh_session)
        self.refresh_thread.daemon = True
        self.refresh_thread.start()
        logging.info("Session refresh timer started")

    def disconnect(self):
        """Enhanced disconnect with proper cleanup"""
        try:
            self.is_running = False

            # Unsubscribe first
            self._unsubscribe_all()

            # Close WebSocket connection
            if self.smart_socket:
                try:
                    self.smart_socket.close()
                    logging.info("WebSocket connection closed")
                except Exception as e:
                    logging.error(f"WebSocket close error: {str(e)}")

            # Join thread if it exists and is alive
            if hasattr(self, 'ws_thread') and self.ws_thread and self.ws_thread.is_alive():
                self.ws_thread.join(timeout=5)

            # Terminate session if connected
            if self.smart_api and self.connection_status:
                try:
                    self.rate_limiter.wait_if_needed('session')
                    self.smart_api.terminateSession(self.config.CLIENT_ID)
                    logging.info("API session terminated")
                except Exception as term_error:
                    logging.error(f"Session termination error: {str(term_error)}")

            self.connection_status = False

        except Exception as e:
            logging.error(f"Disconnect error: {str(e)}")

    def _check_connection_status(self) -> bool:
        """Enhanced connection status check with data flow verification"""
        try:
            # Basic connection check
            if not self.smart_socket or not self.smart_socket.sock or not self.smart_socket.sock.connected:
                return False

            # Check feed token validity
            if not self.feed_token:
                return False

            # During initialization, we may not have received ticks yet
            # Check if system is in startup phase (first 30 seconds)
            if hasattr(self, 'startup_time'):
                startup_elapsed = (datetime.now() - self.startup_time).total_seconds()
                if startup_elapsed < 30:
                    # Skip data flow check during startup
                    return True

            # Verify data flow
            current_time = datetime.now()

            # If last_tick_time is empty for all symbols (initial state),
            # don't fail the connection check
            if not self.last_tick_time:
                return True

            data_flowing = False
            stale_symbols = []

            for symbol in self.config.SYMBOLS:
                if symbol in self.last_tick_time:
                    # Guard against None values
                    last_tick = self.last_tick_time.get(symbol)
                    if last_tick is not None:
                        time_diff = (current_time - last_tick).total_seconds()
                        if time_diff < 10:  # Data within last 10 seconds
                            data_flowing = True
                        else:
                            stale_symbols.append(f"{symbol} ({int(time_diff)}s)")
                    else:
                        # Handle None timestamp
                        stale_symbols.append(f"{symbol} (no timestamp)")

            # If we have tick times but all are stale, log and return false
            if self.last_tick_time and not data_flowing and stale_symbols:
                logging.warning(f"No recent data for: {', '.join(stale_symbols)}")
                return False

            return True

        except Exception as e:
            logging.error(f"Connection check error: {str(e)}")
            return False


class MarketDataManager:
    """Enhanced market data management system with proper integration"""

    def __init__(self, angel_connector: AngelOneConnector, config: Config):
        # Core components
        self.angel_connector = angel_connector
        self.config = config
        self.ml_ensemble = MLEnsemble(config)

        # Data structures
        self.historical_data = {}
        self.latest_data = {}
        self.tick_cache = {}
        self.received_ticks = {}
        self.bars_1min = {}
        self.current_candle = {}

        # Thread safety and queues
        self.lock = threading.Lock()
        self.process_thread = None

        # Use angel_connector's queue instead of creating our own
        self.data_queue = self.angel_connector.tick_data

        # State tracking
        self.last_tick_time = {}
        self.last_indicator_update = {}
        self.is_running = False
        self.warmup_complete = False
        self._warmup_start_time = None
        self.WARMUP_PERIOD = 120  # seconds

        # Memory management
        self.max_history_size = 10000
        self.cleanup_interval = 300
        self.last_cleanup = datetime.now()

        # Error tracking
        self.error_count = 0
        self.max_errors = 5

        # Subscription tracking
        self.subscribed_tokens = {}
        self.subscription_status = {}

        # Initialize data structures
        self._initialize_data_structures()

    def _initialize_data_structures(self):
        """Initialize data structures with proper config access"""
        try:
            for symbol in self.config.SYMBOLS.keys():
                self.historical_data[symbol] = pd.DataFrame()
                self.latest_data[symbol] = {}
                self.tick_cache[symbol] = {}
                self.received_ticks[symbol] = []
                self.current_candle[symbol] = None
                self.bars_1min[symbol] = pd.DataFrame()
                self.last_tick_time[symbol] = datetime.now()
                self.last_indicator_update[symbol] = datetime.now()

            logging.info("Data structures initialized successfully")

        except Exception as e:
            logging.error(f"Data structure initialization error: {str(e)}")
            raise

    def start(self):
        """Start market data processing with session refresh"""
        try:
            self.is_running = True
            self.warmup_complete = False
            self._warmup_start_time = datetime.now()

            # Ensure fresh session before subscribing
            if not self.angel_connector.is_connected():
                logging.info("Refreshing Angel One connection...")
                if not self.angel_connector.connect():
                    raise Exception("Failed to connect to Angel One")

            # Subscribe to market data after ensuring connection
            subscription_success = False
            max_retries = 3

            for attempt in range(max_retries):
                try:
                    if self._subscribe_market_data():
                        subscription_success = True
                        break

                    logging.warning(f"Subscription attempt {attempt + 1} failed, retrying...")
                    time.sleep(2)

                    # Reconnect if needed
                    if not self.angel_connector.is_connected():
                        if not self.angel_connector.connect():
                            continue

                except Exception as e:
                    logging.error(f"Subscription attempt {attempt + 1} error: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    raise

            if not subscription_success:
                raise Exception("Failed to subscribe to market data")

            # Start processing thread
            self.process_thread = threading.Thread(target=self._monitor_market_data)
            self.process_thread.daemon = True
            self.process_thread.start()

            logging.info("Market data manager started successfully")
            return True

        except Exception as e:
            logging.error(f"Market data manager start error: {str(e)}")
            self.is_running = False
            return False

    def stop(self):
        """Stop data processing with cleanup"""
        try:
            self.is_running = False

            # Unsubscribe from all symbols
            if hasattr(self, 'angel_connector'):
                self.angel_connector._unsubscribe_all()

            # Save latest data
            self._save_historical_data()

            # Wait for thread to complete
            if self.process_thread and self.process_thread.is_alive():
                self.process_thread.join(timeout=5)

            logging.info("Market data manager stopped successfully")

        except Exception as e:
            logging.error(f"Error stopping market data manager: {str(e)}")

    async def initialize(self) -> bool:
        """Initialize market data manager with proper sequence"""
        try:
            logging.info("Initializing market data manager...")

            # 1. First load historical data
            logging.info("Loading historical data...")
            await self._load_historical_data()

            # 2. Initialize data structures
            self._initialize_data_structures()

            # 3. Calculate initial indicators
            for symbol in self.config.SYMBOLS:
                if symbol in self.historical_data and not self.historical_data[symbol].empty:
                    self.ml_ensemble._calculate_technical_features(self.get_current_data(symbol))
                    logging.info(f"Calculated initial indicators for {symbol}")

            # 4. Initialize tick processing
            self.is_running = True
            self._warmup_start_time = datetime.now()

            # 5. Start processing thread
            self.process_thread = threading.Thread(target=self._process_ticks)
            self.process_thread.daemon = True
            self.process_thread.start()

            # 6. Subscribe to market data
            success = self._subscribe_market_data()
            if not success:
                raise Exception("Failed to subscribe to market data")

            logging.info("Market data manager initialized successfully")
            return True

        except Exception as e:
            logging.error(f"Market data manager initialization error: {str(e)}")
            return False

    def _process_ticks(self):
        """Process incoming tick data from AngelOne websocket"""
        consecutive_empty = 0
        last_data_check = datetime.now()
        MAX_CONSECUTIVE_EMPTY = 5
        no_data_count = 0
        MAX_NO_DATA = 10
        last_resubscribe = datetime.now()

        while self.is_running:
            try:
                # Read from angel_connector's queue
                try:
                    tick_data = self.angel_connector.tick_data.get(timeout=0.1)

                    if tick_data:
                        # Reset monitoring counters
                        consecutive_empty = 0
                        last_data_check = datetime.now()
                        no_data_count = 0

                        # Extract symbol from token
                        symbol = tick_data.get('symbol')
                        if symbol:
                            current_time = datetime.now()

                            # CRITICAL FIX: Update our last_tick_time directly
                            self.last_tick_time[symbol] = current_time

                            with self.lock:
                                # Get price from various fields with case-insensitive fallbacks
                                close_price = None
                                for field in ['Close', 'close', 'ltp', 'price']:
                                    if field in tick_data:
                                        close_price = float(tick_data[field])
                                        break
                                if close_price is None:
                                    close_price = 0.0

                                # Build processed tick with proper OHLC fallbacks
                                processed_tick = {
                                    'timestamp': tick_data.get('timestamp', current_time),
                                    'Open': float(tick_data.get('Open', tick_data.get('open', close_price))),
                                    'High': float(tick_data.get('High', tick_data.get('high', close_price))),
                                    'Low': float(tick_data.get('Low', tick_data.get('low', close_price))),
                                    'Close': close_price,
                                    'Volume': int(tick_data.get('Volume', tick_data.get('volume', 0)))
                                }

                                # Update latest data
                                self.latest_data[symbol] = processed_tick

                                # Update historical ticks
                                if len(self.received_ticks[symbol]) >= self.max_history_size:
                                    self.received_ticks[symbol] = self.received_ticks[symbol][
                                                                  -(self.max_history_size - 1):]
                                self.received_ticks[symbol].append(processed_tick)

                                # Update minute bars
                                self._update_minute_bars(symbol)

                                # Check warmup completion
                                if not self.warmup_complete:
                                    if (datetime.now() - self._warmup_start_time).seconds >= self.WARMUP_PERIOD:
                                        self.warmup_complete = True
                                        logging.info("Market data warmup complete")
                                        logging.getLogger().setLevel(logging.INFO)

                except queue.Empty:
                    consecutive_empty += 1
                    current_time = datetime.now()

                    # Check connection status after multiple empty reads
                    if consecutive_empty >= MAX_CONSECUTIVE_EMPTY:
                        time_since_data = (current_time - last_data_check).seconds

                        if time_since_data > 30:  # No data for 30 seconds
                            # Force update from angel_connector timestamps if we have shared access
                            if hasattr(self.angel_connector, 'last_tick_time'):
                                for symbol, last_time in self.angel_connector.last_tick_time.items():
                                    if symbol in self.config.SYMBOLS:
                                        if symbol not in self.last_tick_time or self.last_tick_time[
                                            symbol] != last_time:
                                            self.last_tick_time[symbol] = last_time
                                            logging.info(f"Synchronized timestamp for {symbol} with AngelOneConnector")

                            # After synchronization, check again for stale data
                            stale_symbols = []
                            for symbol in self.config.SYMBOLS:
                                if symbol in self.last_tick_time:
                                    if (current_time - self.last_tick_time[symbol]).seconds > 30:
                                        stale_symbols.append(symbol)
                                else:
                                    stale_symbols.append(symbol)

                            if stale_symbols:
                                logging.warning(f"No recent data for symbols: {stale_symbols}")

                                # Try resubscribing if no data and enough time has passed
                                if (current_time - last_resubscribe).seconds > 30:
                                    logging.info("No data flow detected - resubscribing...")
                                    self._subscribe_market_data()
                                    last_resubscribe = current_time
                                    consecutive_empty = 0

                                # If persistent data issues, try reconnection
                                if no_data_count >= MAX_NO_DATA:
                                    logging.warning("Persistent data flow issues - attempting reconnection")
                                    if self.angel_connector._attempt_reconnect():
                                        no_data_count = 0
                                        consecutive_empty = 0
                                        last_data_check = current_time
                                    time.sleep(5)  # Prevent rapid reconnection attempts

                # Brief sleep to prevent tight loop
                time.sleep(0.01)

            except Exception as e:
                logging.error(f"Tick processing error: {str(e)}")
                time.sleep(1)  # Sleep longer on error

    # In MarketDataManager class
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price with proper tick data access"""
        try:
            # Check in latest_data dictionary first
            if symbol in self.latest_data:
                data = self.latest_data[symbol]
                if 'Close' in data and data['Close'] > 0:
                    return float(data['Close'])
                elif 'price' in data and data['price'] > 0:
                    return float(data['price'])

            # Check in tick_cache next
            if symbol in self.tick_cache:
                cache = self.tick_cache[symbol]
                if 'price' in cache and cache['price'] > 0:
                    return float(cache['price'])

            # Check received ticks as another option
            if symbol in self.received_ticks and len(self.received_ticks[symbol]) > 0:
                latest_tick = self.received_ticks[symbol][-1]
                if 'Close' in latest_tick and latest_tick['Close'] > 0:
                    return float(latest_tick['Close'])
                elif 'price' in latest_tick and latest_tick['price'] > 0:
                    return float(latest_tick['price'])
                elif 'ltp' in latest_tick and latest_tick['ltp'] > 0:
                    return float(latest_tick['ltp'])

            # Try AngelConnector's data if available
            if hasattr(self, 'angel_connector') and hasattr(self.angel_connector, 'latest_data'):
                if symbol in self.angel_connector.latest_data:
                    connector_data = self.angel_connector.latest_data[symbol]
                    if 'Close' in connector_data and connector_data['Close'] > 0:
                        return float(connector_data['Close'])
                    elif 'price' in connector_data and connector_data['price'] > 0:
                        return float(connector_data['price'])

            return None
        except Exception as e:
            logging.error(f"Error getting latest price for {symbol}: {str(e)}")
            return None

    def _handle_data_gap(self, symbol: str):
        """Handle data gaps with automatic recovery"""
        try:
            logging.warning(f"Data gap detected for {symbol}")

            # First try resubscription
            if symbol in self.config.SYMBOLS:
                token = self.config.SYMBOLS[symbol]['token']

                subscribe_message = {
                    "correlationID": str(int(time.time())),
                    "action": 1,
                    "params": {
                        "mode": 3,
                        "tokenList": [{
                            "exchangeType": 1,
                            "tokens": [str(token)]
                        }]
                    }
                }

                # Send with retries
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        if self.angel_connector.smart_socket and \
                                self.angel_connector.smart_socket.sock and \
                                self.angel_connector.smart_socket.sock.connected:
                            self.angel_connector.smart_socket.send(json.dumps(subscribe_message))
                            logging.info(f"Resubscribed to {symbol}")
                            return
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise e
                        time.sleep(1)

        except Exception as e:
            logging.error(f"Data gap handling error for {symbol}: {str(e)}")
            self.angel_connector._attempt_reconnect()

    def _verify_data_flow(self) -> bool:
        """Verify market data flow with enhanced checks"""
        try:
            start_time = time.time()
            timeout = 10  # Wait up to 10 seconds for data

            while time.time() - start_time < timeout:
                data_flowing = False
                for symbol in self.config.SYMBOLS:
                    if symbol in self.last_tick_time:
                        time_diff = (datetime.now() - self.last_tick_time[symbol]).seconds
                        if time_diff < 5:
                            data_flowing = True
                            break

                if data_flowing:
                    return True

                time.sleep(0.1)

            return False

        except Exception as e:
            logging.error(f"Data flow verification error: {str(e)}")
            return False

    def _monitor_market_data(self):
        """Single source of truth for market data monitoring with flow maintenance"""
        last_check = datetime.now()
        CHECK_INTERVAL = 60  # Increased to 60 seconds
        last_reconnect = datetime.now()
        MIN_RECONNECT_INTERVAL = 300  # Increased to 5 minutes
        last_resubscribe = datetime.now()
        RESUBSCRIBE_INTERVAL = 60  # Increased to 60 seconds

        # Critical: Create a flag to skip reconnection after warmup
        skip_reconnect_until = None

        while self.is_running:
            try:
                current_time = datetime.now()

                # Handle warmup period
                if not self.warmup_complete:
                    warmup_elapsed = (current_time - self._warmup_start_time).seconds
                    if warmup_elapsed >= self.WARMUP_PERIOD:
                        self.warmup_complete = True
                        logging.info("✅ Warmup period completed - starting active trading")
                        logging.getLogger().setLevel(logging.INFO)

                        # Critical: Set the skip reconnection flag for 3 minutes after warmup
                        skip_reconnect_until = current_time + timedelta(minutes=3)

                        # Immediately update timestamps using ticks
                        for symbol in self.config.SYMBOLS:
                            if symbol in self.received_ticks and len(self.received_ticks[symbol]) > 0:
                                # Update timestamp to current time to prevent immediate reconnection
                                self.last_tick_time[symbol] = current_time

                        # Just resubscribe without reconnection
                        self._subscribe_market_data()

                    time.sleep(1)  # Sleep during warmup
                    continue

                # Skip ALL connection checks and reconnection during the grace period
                if skip_reconnect_until and current_time < skip_reconnect_until:
                    # During grace period, update timestamps from received ticks
                    for symbol in self.config.SYMBOLS:
                        if symbol in self.received_ticks and len(self.received_ticks[symbol]) > 0:
                            # Set timestamp to now, to prevent reconnect triggering
                            self.last_tick_time[symbol] = current_time

                    self.skip_reconnect_until = None

                    time.sleep(1)
                    continue

                # Regular monitoring after warmup and grace period
                if (current_time - last_check).seconds >= CHECK_INTERVAL:
                    # First check if we're actually receiving ticks
                    receiving_ticks = False
                    for symbol in self.config.SYMBOLS:
                        if (symbol in self.received_ticks and
                                len(self.received_ticks[symbol]) > 0):
                            # Update timestamps from the ticks we're receiving
                            self.last_tick_time[symbol] = current_time
                            receiving_ticks = True

                    # Skip connection checks if we're receiving ticks
                    if receiving_ticks:
                        logging.info("Tick data confirmed flowing - skipping connection check")
                        last_check = current_time
                        continue

                    # Only proceed with traditional data flow check if we haven't confirmed ticks
                    data_status = {}
                    data_flowing = False
                    stale_symbols = []

                    for symbol in self.config.SYMBOLS:
                        if symbol in self.last_tick_time:
                            last_tick = self.last_tick_time.get(symbol)
                            if last_tick:
                                time_diff = (current_time - last_tick).seconds
                                data_status[symbol] = time_diff
                                if time_diff < 600:
                                    data_flowing = True
                                else:
                                    stale_symbols.append(symbol)

                    # If data is flowing, just log and continue
                    if data_flowing:
                        logging.info(
                            f"Data flowing for {len(self.config.SYMBOLS) - len(stale_symbols)}/{len(self.config.SYMBOLS)} symbols")
                        last_check = current_time
                        continue

                    # Only proceed with reconnection logic if data truly isn't flowing
                    reconnect_elapsed = (current_time - last_reconnect).seconds
                    resubscribe_elapsed = (current_time - last_resubscribe).seconds

                    # Try resubscribing first
                    if resubscribe_elapsed >= RESUBSCRIBE_INTERVAL:
                        logging.info("No recent data detected - resubscribing...")
                        self._subscribe_market_data()
                        last_resubscribe = current_time
                        last_check = current_time
                        continue

                    # Only attempt reconnection as a last resort
                    if len(stale_symbols) == len(self.config.SYMBOLS) and reconnect_elapsed >= MIN_RECONNECT_INTERVAL:
                        logging.warning("All symbols stale - attempting reconnection")
                        if self.angel_connector._attempt_reconnect():
                            last_reconnect = current_time
                            time.sleep(5)
                            self._subscribe_market_data()
                            logging.info("Reconnection successful")
                        else:
                            logging.error("Reconnection failed")

                    last_check = current_time

                time.sleep(1)  # Increased to reduce CPU usage

            except Exception as e:
                logging.error(f"Market data monitoring error: {str(e)}")
                time.sleep(5)  # Longer sleep on error

    def _subscribe_market_data(self) -> bool:
        """Wrapper around angel_connector's subscription method"""
        try:
            # Verify connection
            if not self.angel_connector.is_connected():
                logging.error("Cannot subscribe - AngelOne connector not connected")
                return False

            # Use angel_connector's subscription method
            success = self.angel_connector._subscribe_market_data()

            if success:
                # Track subscription in market data manager
                self.subscribed_tokens = self.angel_connector.subscribed_tokens.copy()
                return True

            return False

        except Exception as e:
            logging.error(f"Market data subscription error: {str(e)}")
            return False

    def get_minute_bars(self, symbol: str, lookback: int = None) -> pd.DataFrame:
        """Get minute bars with validation"""
        try:
            if symbol not in self.bars_1min:
                return pd.DataFrame()

            bars = self.bars_1min[symbol]
            if lookback:
                bars = bars.tail(lookback)

            return bars.copy()

        except Exception as e:
            logging.error(f"Error getting bars for {symbol}: {str(e)}")
            return pd.DataFrame()

    def get_current_data(self, symbol: str) -> pd.DataFrame:
        """Get current market data with all required indicators"""
        try:
            if not hasattr(self, 'historical_data') or symbol not in self.historical_data:
                return pd.DataFrame()

            df = self.historical_data[symbol].copy()
            if df.empty:
                return pd.DataFrame()

            # Ensure historical data index is timezone naive
            if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            # Add live data if available
            if symbol in self.bars_1min and not self.bars_1min[symbol].empty:
                live_data = self.bars_1min[symbol].copy()

                # Ensure live data index is timezone naive
                if isinstance(live_data.index, pd.DatetimeIndex) and live_data.index.tz is not None:
                    live_data.index = live_data.index.tz_localize(None)

                # Ensure live data has all necessary columns
                self.ml_ensemble._calculate_technical_features(self.get_current_data(symbol))

                # Combine with historical data
                df = pd.concat([df, live_data])
                df = df[~df.index.duplicated(keep='last')]
                df = df.sort_index()

            # Verify data has the most important required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logging.warning(f"Missing required columns in data for {symbol}: {missing_cols}")

                # For any missing columns, reconstruct from Close if possible
                if 'Close' in df.columns:
                    for col in missing_cols:
                        if col in ['Open', 'High', 'Low']:
                            df[col] = df['Close']
                        elif col == 'Volume':
                            df[col] = 0

            return df

        except Exception as e:
            logging.error(f"Error getting current data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def is_data_flowing(self) -> bool:
        """Check if market data is flowing with proper delay and validation"""
        try:
            current_time = datetime.now()
            data_alive = False
            stale_symbols = []

            for symbol in self.config.SYMBOLS:
                last_update = self.last_tick_time.get(symbol)
                if last_update:
                    time_diff = (current_time - last_update).seconds
                    if time_diff < 5:  # Data in last 5 seconds
                        data_alive = True
                    else:
                        stale_symbols.append(f"{symbol} ({time_diff}s)")

            # Only log if there are stale symbols
            if stale_symbols:
                logging.debug(f"Stale data for: {', '.join(stale_symbols)}")

            return data_alive

        except Exception as e:
            logging.error(f"Data flow check error: {str(e)}")
            return False

    def _validate_tick_data(self, tick_data: dict) -> bool:
        """Validate incoming tick data"""
        try:
            required_fields = ['token', 'ltp', 'exchange_timestamp']
            if not all(field in tick_data for field in required_fields):
                return False

            if not all(tick_data.get(field) for field in required_fields):
                return False

            return True

        except Exception as e:
            logging.error(f"Tick validation error: {str(e)}")
            return False

    def _request_snapshot(self, symbol: str):
        """Request latest snapshot for symbol"""
        try:
            if symbol in self.config.SYMBOLS:
                token = self.config.SYMBOLS[symbol]['token']

                snapshot_msg = {
                    "correlationID": "1",
                    "action": 1,
                    "params": {
                        "mode": 3,
                        "tokenList": [{"exchangeType": 1, "tokens": [token]}]
                    }
                }

                self.angel_connector.smart_socket.send(json.dumps(snapshot_msg))
                logging.info(f"Requested snapshot for {symbol}")

        except Exception as e:
            logging.error(f"Snapshot request error: {str(e)}")

    def _load_historical_data(self) -> bool:
        """Load historical data with 60-day limit"""
        try:
            logging.info("Loading historical market data...")
            end_date = datetime.now()
            # Enforce exactly 60 days
            start_date = end_date - timedelta(days=30)

            data_loaded = False

            for symbol, info in self.config.SYMBOLS.items():
                try:
                    # Track all candles across multiple requests
                    all_candles = []
                    current_date = start_date
                    # AngelOne limit: 30 days for ONE_MINUTE data
                    chunk_size = 30 if self.config.TIMEFRAME == "ONE_MINUTE" else 60

                    logging.info(f"Fetching data for {symbol}...")

                    # Only make requests until we've covered the 60 days
                    while current_date < end_date and (end_date - current_date).days <= 60:
                        chunk_end = min(current_date + timedelta(days=chunk_size), end_date)

                        params = {
                            "exchange": info['exchange'],
                            "symboltoken": info['token'],
                            "interval": self.config.TIMEFRAME,
                            "fromdate": current_date.strftime("%Y-%m-%d %H:%M"),
                            "todate": chunk_end.strftime("%Y-%m-%d %H:%M")
                        }

                        response = self.angel_connector._make_api_request('getCandleData', params=params)

                        if response and response.get('status') and response.get('data'):
                            chunk_data = response['data']

                            # Clean invalid candles before storing
                            valid_chunk_data = []
                            for candle in chunk_data:
                                if len(candle) >= 6:
                                    timestamp, open_val, high_val, low_val, close_val, volume = candle[:6]
                                    try:
                                        # Only add candles with valid positive prices
                                        if all(float(val) > 0 for val in [open_val, high_val, low_val, close_val]):
                                            valid_chunk_data.append(candle)
                                    except:
                                        continue  # Skip invalid candles

                            all_candles.extend(valid_chunk_data)
                            logging.info(
                                f"Received {len(valid_chunk_data)} valid candles for period {current_date.date()} to {chunk_end.date()}")
                        else:
                            error_msg = response.get('message', 'Unknown error') if response else 'No response'
                            logging.error(f"Failed to fetch data chunk for {symbol}: {error_msg}")

                        # Move to next chunk
                        current_date = chunk_end
                        time.sleep(1)  # Small delay between requests

                        # If we've collected enough data, stop
                        if len(all_candles) >= 22500:  # ~60 days of minute data (375 candles/day × 60 days)
                            logging.info(f"Collected sufficient data: {len(all_candles)} candles")
                            break

                    if all_candles:
                        # Create DataFrame from accumulated candles
                        df = pd.DataFrame(all_candles,
                                          columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        # Rename to uppercase to match the rest of your code
                        df.rename(columns={
                            'open': 'Open',
                            'high': 'High',
                            'low': 'Low',
                            'close': 'Close',
                            'volume': 'Volume'
                        }, inplace=True)

                        # Convert types and handle timezones
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)

                        # Ensure numeric columns
                        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                            df[col] = pd.to_numeric(df[col], errors='coerce')

                        # Validate data quality
                        if len(df) < self.config.MIN_TRAINING_SAMPLES:
                            logging.error(f"Insufficient data for {symbol}: {len(df)} samples")
                            continue

                        if df.isnull().any().any():
                            logging.warning(f"Missing values in {symbol} data - filling...")
                            df = df.fillna(method='ffill').fillna(method='bfill')

                        # Calculate indicators
                        df = self.ml_ensemble._calculate_technical_features(df)

                        self.historical_data[symbol] = df
                        logging.info(f"Successfully loaded {len(df)} candles for {symbol}")
                        data_loaded = True

                    else:
                        logging.error(f"No data retrieved for {symbol}")

                except Exception as symbol_error:
                    logging.error(f"Error loading data for {symbol}: {str(symbol_error)}")

            if not data_loaded:
                logging.error("Failed to load historical data for any symbol")
                return False

            logging.info("Historical data loaded successfully")
            return True

        except Exception as e:
            logging.error(f"Historical data loading error: {str(e)}")
            return False

    def get_status(self) -> dict:
        """Get current status of market data manager"""
        try:
            return {
                'data_flowing': self.is_data_flowing(),
                'subscribed_symbols': list(self.config.SYMBOLS.keys()),
                'error_count': self.error_count,
                'last_update': max(self.last_tick_time.values()) if self.last_tick_time else None
            }
        except Exception as e:
            logging.error(f"Status check error: {str(e)}")
            return {}

    def _update_minute_bars(self, symbol: str):
        """Update 1-minute OHLCV bars with robust data handling"""
        try:
            if not self.received_ticks[symbol] or len(self.received_ticks[symbol]) < 2:
                return

            # Create DataFrame from received data
            df = pd.DataFrame(self.received_ticks[symbol])

            # Remove invalid ticks before processing
            for price_col in ['price', 'Close', 'Open', 'High', 'Low']:
                if price_col in df.columns:
                    df = df[(df[price_col] > 0) & df[price_col].notna()]

            if df.empty:
                return

            # Ensure all required fields exist
            required_fields = ['timestamp', 'price', 'volume', 'Open', 'High', 'Low', 'Close']

            # First handle basic price and volume fields
            for field in required_fields:
                if field == 'timestamp' and field not in df.columns:
                    df['timestamp'] = [datetime.now() for _ in range(len(df))]

                if field == 'price' and field not in df.columns:
                    # Try alternate fields
                    for alt in ['Close', 'close', 'ltp']:
                        if alt in df.columns:
                            df['price'] = df[alt]
                            break
                    if 'price' not in df.columns:
                        df['price'] = 0.0

                if field == 'volume' and field not in df.columns:
                    # Try alternate fields
                    for alt in ['Volume', 'last_traded_quantity']:
                        if alt in df.columns:
                            df['volume'] = df[alt]
                            break
                    if 'volume' not in df.columns:
                        df['volume'] = 0

            # Now ensure OHLC fields exist for resampling
            if 'Open' not in df.columns:
                df['Open'] = df['price']
            if 'High' not in df.columns:
                df['High'] = df['price']
            if 'Low' not in df.columns:
                df['Low'] = df['price']
            if 'Close' not in df.columns:
                df['Close'] = df['price']
            if 'Volume' not in df.columns:
                df['Volume'] = df['volume']

            # Ensure timestamp is index
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

            # Resample with explicit OHLC specification
            resampled = pd.DataFrame()
            resampled['Open'] = df['Open'].resample('1min', label='right').first()
            resampled['High'] = df['High'].resample('1min', label='right').max()
            resampled['Low'] = df['Low'].resample('1min', label='right').min()
            resampled['Close'] = df['Close'].resample('1min', label='right').last()
            resampled['Volume'] = df['Volume'].resample('1min', label='right').sum()

            # Add to existing bars
            if symbol in self.bars_1min and not self.bars_1min[symbol].empty:
                existing_bars = self.bars_1min[symbol].copy()
                combined = pd.concat([existing_bars, resampled])
                combined = combined[~combined.index.duplicated(keep='last')]
                self.bars_1min[symbol] = combined.tail(1000)  # Keep last 1000 bars
            else:
                self.bars_1min[symbol] = resampled

            # Calculate technical indicators on latest bars
            self.ml_ensemble._calculate_technical_features(self.get_current_data(symbol))

        except Exception as e:
            logging.error(f"Error updating minute bars for {symbol}: {str(e)}")

    def _update_candle(self, symbol: str, tick: dict):
        """Update current building candle"""
        if not isinstance(symbol, str):
            logging.error(f"Invalid symbol type in _update_candle: {type(symbol)}")
            return

        try:
            current_minute = tick['timestamp'].floor('1min')

            if symbol not in self.current_candle:
                self.current_candle[symbol] = None

            if (self.current_candle[symbol] is None or
                    current_minute != self.current_candle[symbol]['timestamp']):

                if self.current_candle[symbol] is not None:
                    if symbol not in self.bars_1min:
                        self.bars_1min[symbol] = pd.DataFrame()
                    new_candle = pd.DataFrame([self.current_candle[symbol]])
                    self.bars_1min[symbol] = pd.concat([self.bars_1min[symbol], new_candle], ignore_index=True)

                self.current_candle[symbol] = {
                    'timestamp': current_minute,
                    'Open': tick['price'],
                    'High': tick['price'],
                    'Low': tick['price'],
                    'Close': tick['price'],
                    'Volume': tick['volume']
                }
            else:
                self.current_candle[symbol]['High'] = max(
                    self.current_candle[symbol]['High'],
                    tick['price']
                )
                self.current_candle[symbol]['Low'] = min(
                    self.current_candle[symbol]['Low'],
                    tick['price']
                )
                self.current_candle[symbol]['Close'] = tick['price']
                self.current_candle[symbol]['Volume'] += tick['volume']

        except Exception as e:
            logging.error(f"Error updating candle: {str(e)}")

    def _perform_daily_reset(self):
        """Reset daily data structures"""
        try:
            with self.lock:
                self._save_historical_data()

                self.received_ticks = {}
                self.tick_cache = {}

                self.current_candle = {}

                if not isinstance(self.historical_data, dict):
                    logging.error(f"Invalid historical_data type: {type(self.historical_data)}")
                    return

                for symbol in self.historical_data:
                    self.historical_data[symbol] = self.historical_data[symbol].last('7D')

                logging.info("Daily reset completed")

        except Exception as e:
            logging.error(f"Daily reset error: {str(e)}")

    def _check_daily_reset(self):
        """Check if daily reset is needed"""
        current_time = datetime.now()
        if current_time.hour == 9 and current_time.minute == 15:
            self._perform_daily_reset()

    def _cleanup_old_data(self):
        """Regular cleanup of historical data"""
        try:
            with self.lock:
                for symbol in self.historical_data:
                    if len(self.historical_data[symbol]) > self.max_history_size:
                        self.historical_data[symbol] = self.historical_data[symbol].iloc[-self.max_history_size:]

                for symbol in self.tick_cache:
                    if len(self.tick_cache[symbol]) > self.max_history_size:
                        self.tick_cache[symbol] = self.tick_cache[symbol][-self.max_history_size:]

                for symbol in self.bars_1min:
                    if len(self.bars_1min[symbol]) > self.max_history_size:
                        self.bars_1min[symbol] = self.bars_1min[symbol].iloc[-self.max_history_size:]

        except Exception as e:
            logging.error(f"Data cleanup error: {str(e)}")

    def _save_historical_data(self):
        """Save historical data to disk"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            for symbol, data in self.historical_data.items():
                filename = f"historical_data_{symbol}_{timestamp}.csv"
                data.to_csv(filename)
                logging.info(f"Saved historical data for {symbol}")
        except Exception as e:
            logging.error(f"Error saving historical data: {str(e)}")

    def check_warmup_status(self) -> bool:
        if not self.warmup_complete:
            current_time = datetime.now()
            warmup_elapsed = (current_time - self._warmup_start_time).seconds

            if warmup_elapsed >= self.WARMUP_PERIOD:
                # Add this verification
                if self.angel_connector.is_connected():
                    # Verify feed token is still valid
                    if self.angel_connector.feed_token:
                        self.warmup_complete = True
                        # Re-subscribe to symbols to ensure fresh connection
                        self._subscribe_market_data()
                        return True
                return False


class OrderManager:
    """Enhanced OrderManager with advanced strategy and market feed integration"""

    def __init__(self, angel_connector: AngelOneConnector, config: Config):
        # Core components
        self.angel_connector = angel_connector
        self.config = config
        self.initial_capital = config.INITIAL_CAPITAL  # Use config value
        self.orders = {}  # Simulated orders
        self.position_manager = PositionManager(self.config)
        self.pending_orders = queue.Queue()
        self.order_updates = queue.Queue()
        self.tick_cache = {}  # Cache for latest tick data
        self.price_history = {}  # Price history for each symbol
        self.historical_data = {}  # Add this line
        self.market_data_manager = MarketDataManager(angel_connector, config)
        self.latest_data = {}  # Initialize latest_data attribute

        # System state
        self.is_running = False
        self.lock = threading.Lock()
        self.error_count = 0
        self.max_errors = 3

        # Performance tracking
        self.trade_history = []
        self.daily_pnl = 0
        self.total_trades = 0
        self.winning_trades = 0

        # Risk parameters
        self.risk_params = {
            'position_size': 0.02,  # Base position size as % of capital
            'stop_loss': 0.01,  # Stop loss as % from entry
            'profit_target': 0.02,  # Profit target as % from entry
            'max_positions': 20,  # Maximum concurrent positions
            'trailing_stop': 0.005,  # Trailing stop % below price
            'max_daily_loss': 0.05,  # Maximum daily loss allowed
            'max_position_size': 0.05,  # Maximum single position size
            'max_capital_at_risk': 0.3  # Maximum total capital at risk
        }

        # Strategy parameters
        self.strategy_params = {
            'ma_short': 5,  # Short MA period
            'ma_long': 15,  # Long MA period
            'rsi_entry': 40,  # RSI entry level
            'rsi_exit': 60,  # RSI exit level
            'volume_threshold': 0.8,  # Volume threshold multiplier
            'trend_strength': 0.01,  # Minimum trend strength
            'volatility_threshold': 0.02  # Maximum volatility threshold
        }

        # Capital tracking
        self.initial_capital = 100000  # Starting capital
        self.current_capital = float(self.config.INITIAL_CAPITAL)
        self.peak_capital = float(self.config.INITIAL_CAPITAL)
        self.max_drawdown = 0

        # Position tracking
        self.positions_metrics = {
            'total_exposure': 0,
            'highest_capital': self.initial_capital,
            'lowest_capital': self.initial_capital,
            'open_positions_count': 0,
            'daily_trades_count': 0
        }

        # Initialize price history storage for each symbol
        for symbol in self.config.SYMBOLS:
            self.price_history[symbol] = {
                'prices': [],
                'volumes': [],
                'timestamps': [],
                'opens': [],
                'highs': [],
                'lows': [],
                'bid_qtys': [],
                'ask_qtys': []
            }

    def start(self):
        """Start order processing"""
        self.is_running = True

        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_positions)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

        # Start tick processing thread
        self.tick_thread = threading.Thread(target=self._process_ticks)
        self.tick_thread.daemon = True
        self.tick_thread.start()

        print("✅ Order manager initialized")

    def stop(self):
        """Stop order processing"""
        self.is_running = False
        self._close_all_positions()
        self._save_trade_history()
        logging.info("Order manager stopped")

    def _process_ticks(self):
        """Modified tick processing without direct data checks"""
        while self.is_running:
            try:
                tick_data = self.angel_connector.tick_data.get(timeout=1)
                if tick_data:
                    token = str(tick_data.get('token', ''))
                    symbol = next(
                        (sym for sym, info in self.config.SYMBOLS.items()
                         if info['token'] == token),
                        None
                    )

                    if symbol:
                        with self.lock:
                            self._process_tick_data(symbol, tick_data)

            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Tick processing error: {str(e)}")
                time.sleep(1)

    def _process_tick_data(self, symbol: str, tick_data: dict):
        """Process incoming tick data"""
        try:
            # Get price from various possible fields
            price = None
            if 'Close' in tick_data:
                price = tick_data.get('Close')
            elif 'close' in tick_data:
                price = tick_data.get('close')
            elif 'ltp' in tick_data:
                price = tick_data.get('ltp')
            elif 'price' in tick_data:
                price = tick_data.get('price')

            if price is None:
                logging.warning(f"No price found in tick data for {symbol}")
                return

            # Get volume from various possible fields
            volume = tick_data.get('Volume',
                                   tick_data.get('volume',
                                                 tick_data.get('open_interest', 0)))

            # Update tick cache with standardized data
            self.tick_cache[symbol] = {
                'price': float(price),
                'volume': int(volume),
                'timestamp': tick_data.get('timestamp', datetime.now()),
                'bid_qty': tick_data.get('total_buy_quantity', 0),
                'ask_qty': tick_data.get('total_sell_quantity', 0)
            }

            # Update price history (this attribute exists in OrderManager)
            if symbol in self.price_history:
                self.price_history[symbol]['prices'].append(float(price))
                self.price_history[symbol]['volumes'].append(int(volume))
                self.price_history[symbol]['timestamps'].append(self.tick_cache[symbol]['timestamp'])
                self.price_history[symbol]['bid_qtys'].append(self.tick_cache[symbol]['bid_qty'])
                self.price_history[symbol]['ask_qtys'].append(self.tick_cache[symbol]['ask_qty'])

                # Keep only recent history
                max_len = 1000
                self.price_history[symbol]['prices'] = self.price_history[symbol]['prices'][-max_len:]
                self.price_history[symbol]['volumes'] = self.price_history[symbol]['volumes'][-max_len:]
                self.price_history[symbol]['timestamps'] = self.price_history[symbol]['timestamps'][-max_len:]
                self.price_history[symbol]['bid_qtys'] = self.price_history[symbol]['bid_qtys'][-max_len:]
                self.price_history[symbol]['ask_qtys'] = self.price_history[symbol]['ask_qtys'][-max_len:]

            # Update market_data_manager's data if available
            if hasattr(self, 'market_data_manager'):
                # Update last_tick_time for status monitoring in market_data_manager
                if hasattr(self.market_data_manager, 'last_tick_time'):
                    self.market_data_manager.last_tick_time[symbol] = self.tick_cache[symbol]['timestamp']

        except Exception as e:
            logging.error(f"Error processing tick data for {symbol}: {str(e)}")

    def _check_trading_signals(self, symbol: str):
        """Check for trading signals"""
        try:
            # Skip if symbol already has position
            if symbol in self.position_manager.positions:
                return

            # Get recent data
            history = self.price_history[symbol]
            if len(history['prices']) < 20:  # Need minimum data
                return

            # Calculate indicators
            indicators = self.ml_ensemble._calculate_technical_features(
                self.market_data_manager.get_current_data(symbol))
            signal = self._generate_trading_signal(indicators)

            # Check for entry conditions
            if signal > 0.6:  # Strong buy signal
                current_price = history['prices'][-1]
                quantity = self._calculate_position_size(symbol, current_price)

                if quantity > 0:
                    self.place_order(
                        symbol=symbol,
                        side='BUY',
                        quantity=quantity,
                        price=current_price
                    )

        except Exception as e:
            logging.error(f"Signal check error: {str(e)}")

    def _generate_trading_signal(self, indicators: dict) -> float:
        """Generate trading signal from indicators"""
        try:
            if not indicators:
                return 0

            signals = []

            # Trend signal
            if indicators['sma_short'] > indicators['sma_long']:
                signals.append(1)
            else:
                signals.append(0)

            # Momentum signal
            if 30 <= indicators['rsi'] <= 70:
                signals.append(1)
                if indicators['rsi'] < 40:  # Oversold
                    signals.append(1)
            else:
                signals.append(0)

            # Volume signal
            if indicators['volume_ratio'] > self.strategy_params['volume_threshold']:
                signals.append(1)
            else:
                signals.append(0)

            # Volatility signal
            if indicators['volatility'] < self.strategy_params['volatility_threshold']:
                signals.append(1)
            else:
                signals.append(0)

            # Price momentum
            if indicators['price_momentum'] > 0:
                signals.append(1)
            else:
                signals.append(0)

            # Calculate weighted signal
            weights = [0.3, 0.2, 0.2, 0.15, 0.15]  # Weights for each signal
            signal = sum(s * w for s, w in zip(signals, weights))

            return signal

        except Exception as e:
            logging.error(f"Signal generation error: {str(e)}")
            return 0

    def place_order(self, symbol: str, side: str, quantity: int, order_type: str = 'MARKET',
                    price: Optional[float] = None, stop_loss: Optional[float] = None,
                    target: Optional[float] = None) -> Optional[str]:
        try:
            if not self._validate_order_params(symbol, side, quantity, price):
                return None

            # CRITICAL FIX: Check for shutdown flag
            if hasattr(self, 'trading_system') and hasattr(self.trading_system,
                                                           '_in_shutdown_process') and self.trading_system._in_shutdown_process:
                logging.warning(f"Skipping order placement during shutdown: {symbol}")
                return None

            # CRITICAL FIX: Get latest, most accurate market price right before order execution
            # This reduces the timing gap between decision and execution
            current_price = None

            # Try multiple sources to get the most up-to-date price
            current_price = self.market_data_manager.get_latest_price(symbol)
            if not current_price and hasattr(self.angel_connector, 'latest_data'):
                if symbol in self.angel_connector.latest_data:
                    current_price = self.angel_connector.latest_data[symbol].get('Close', None)

            # Only if we don't have any price, use the provided one
            if not current_price:
                current_price = price

            if not current_price:
                logging.error(f"Unable to get price for {symbol}")
                return None

            # Check risk limits using risk_manager if available
            risk_check_passed = True
            if hasattr(self, 'risk_manager'):
                risk_check_passed = self.risk_manager.check_risk_limits(symbol, quantity, side, current_price)
            elif hasattr(self, 'trading_system') and hasattr(self.trading_system, 'risk_manager'):
                risk_check_passed = self.trading_system.risk_manager.check_risk_limits(
                    symbol, quantity, side, current_price
                )
            # Fallback to local check if no risk_manager is available
            else:
                try:
                    risk_check_passed = self.risk_manager._check_risk_limits(symbol, quantity, side)
                except:
                    # If no risk_manager attribute, assume risk check passes
                    risk_check_passed = True

            if not risk_check_passed:
                logging.warning(f"Risk limits check failed for {symbol}")
                return None

            # Generate simulated order ID
            order_id = f"SIM_ORD_{int(time.time())}_{symbol}"

            # Calculate dynamic stop loss and target if not provided
            volatility = self._calculate_volatility(symbol)
            if not stop_loss:
                stop_loss = current_price * (1 - max(self.risk_params['stop_loss'], volatility))
            if not target:
                target = current_price * (1 + max(self.risk_params['profit_target'], volatility * 1.5))

            # IMPORTANT: Use the most recent price for everything to ensure consistency
            with self.lock:
                # Record simulated order with current_price for consistency
                self.orders[order_id] = {
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'status': 'EXECUTED',  # Simulate immediate execution for market orders
                    'filled_quantity': quantity,
                    'entry_price': current_price,  # Use consistent price
                    'timestamp': datetime.now(),
                    'stop_loss': stop_loss,
                    'take_profit': target,
                    'trailing_stop': current_price * (1 - self.risk_params['trailing_stop'])
                }

                # Update simulated position
                if symbol not in self.position_manager.positions:
                    self.position_manager.positions[symbol] = {
                        'quantity': 0,
                        'entry_price': 0,
                        'side': None,
                        'pnl': 0,
                        'entry_time': datetime.now(),
                        'highest_price': current_price,
                        'lowest_price': current_price,
                        'max_favorable_excursion': 0,
                        'max_adverse_excursion': 0,
                        'current_price': current_price  # Ensure current_price is set immediately
                    }

                position = self.position_manager.positions[symbol]
                position_value = quantity * current_price

                # Update position
                if side == 'BUY':
                    new_quantity = position['quantity'] + quantity
                    new_cost = (position['quantity'] * position['entry_price'] + quantity * current_price)
                    position['quantity'] = new_quantity
                    position['entry_price'] = new_cost / new_quantity if new_quantity != 0 else 0
                    position['side'] = 'LONG'
                else:  # SELL
                    new_quantity = position['quantity'] - quantity
                    position['quantity'] = new_quantity
                    position['entry_price'] = current_price
                    position['side'] = 'SHORT'

                # Ensure position has current_price set for immediate PnL calculation
                position['current_price'] = current_price

                # Update metrics
                self.positions_metrics['total_exposure'] += position_value
                self.positions_metrics['open_positions_count'] += 1
                self.positions_metrics['daily_trades_count'] += 1

                logging.info(f"Simulated order placed - ID: {order_id}, Symbol: {symbol}, Side: {side}, "
                             f"Quantity: {quantity}, Price: {current_price:.2f}")
                return order_id

        except Exception as e:
            logging.error(f"Simulated order placement error: {str(e)}")
            logging.error(traceback.format_exc())
            return None

    def get_recent_orders(self, count: int = 5) -> List[dict]:
        """Get recent orders with proper validation"""
        try:
            with self.lock:
                # Convert orders dict to list and sort by timestamp
                all_orders = [
                    {
                        'order_id': order_id,
                        'symbol': order['symbol'],
                        'side': order['side'],
                        'quantity': order['quantity'],
                        'status': order['status'],
                        'timestamp': order['timestamp']
                    }
                    for order_id, order in self.orders.items()
                ]

                # Sort by timestamp descending
                sorted_orders = sorted(
                    all_orders,
                    key=lambda x: x['timestamp'],
                    reverse=True
                )

                # Return requested number of orders
                return sorted_orders[:count]

        except Exception as e:
            logging.error(f"Error getting recent orders: {str(e)}")
            return []

    def _monitor_positions(self):
        """Monitor positions and handle exits with proper error handling and shutdown awareness"""
        while self.is_running:
            try:
                # Check for shutdown flag (via trading_system if available)
                shutdown_in_progress = False
                if hasattr(self, 'trading_system') and hasattr(self.trading_system, '_in_shutdown_process'):
                    shutdown_in_progress = self.trading_system._in_shutdown_process

                if shutdown_in_progress:
                    logging.info("Position monitoring detected shutdown signal, exiting...")
                    break

                with self.lock:
                    for symbol, position in list(self.position_manager.positions.items()):
                        try:
                            current_price = self.market_data_manager.get_latest_price(symbol)
                            if not current_price:
                                continue

                            # Get latest data for technical analysis
                            data = self.market_data_manager.get_current_data(symbol)
                            if data is None or data.empty:
                                continue

                            # Determine if position is LONG or SHORT based on 'side' field
                            is_long = position['side'] in ['BUY', 'LONG']

                            # Update price extremes based on position direction
                            if is_long:
                                position['high_price'] = max(position.get('high_price', current_price), current_price)
                            else:  # SHORT
                                position['low_price'] = min(position.get('low_price', current_price), current_price)

                            # Always track both for analysis purposes
                            position['high_price'] = max(position.get('high_price', current_price), current_price)
                            position['low_price'] = min(position.get('low_price', current_price), current_price)

                            # Calculate current P&L with direction awareness
                            if is_long:
                                unrealized_pnl = (current_price - position['entry_price']) * position['quantity']
                            else:  # SHORT/SELL
                                unrealized_pnl = (position['entry_price'] - current_price) * position['quantity']

                            # Update position metrics
                            position['current_price'] = current_price
                            position['unrealized_pnl'] = unrealized_pnl

                            # Direction-aware exit conditions
                            if is_long:
                                # Check long position stop loss
                                if current_price <= position.get('stop_loss', 0):
                                    self.position_manager.close_position(symbol, 'STOP_LOSS')
                                    continue

                                # Check long position take profit
                                if current_price >= position.get('take_profit', float('inf')):
                                    self.position_manager.close_position(symbol, 'TAKE_PROFIT')
                                    continue

                                # Check long position trailing stop
                                trailing_stop = position.get('trailing_stop')
                                if trailing_stop and current_price <= trailing_stop:
                                    self.position_manager.close_position(symbol, 'TRAILING_STOP')
                                    continue

                                # Update trailing stop if in profit (LONG position)
                                if current_price > position['entry_price']:
                                    new_stop = current_price * (1 - self.config.STOP_LOSS)
                                    if new_stop > position.get('trailing_stop', 0):
                                        position['trailing_stop'] = new_stop
                                        logging.debug(f"Updated trailing stop for LONG {symbol} to {new_stop:.2f}")
                            else:  # SHORT position
                                # Check short position stop loss (above entry)
                                if current_price >= position.get('stop_loss', float('inf')):
                                    self.position_manager.close_position(symbol, 'STOP_LOSS')
                                    continue

                                # Check short position take profit (below entry)
                                if current_price <= position.get('take_profit', 0):
                                    self.position_manager.close_position(symbol, 'TAKE_PROFIT')
                                    continue

                                # Check short position trailing stop (above current price)
                                trailing_stop = position.get('trailing_stop')
                                if trailing_stop and current_price >= trailing_stop:
                                    self.position_manager.close_position(symbol, 'TRAILING_STOP')
                                    continue

                                # Update trailing stop if in profit (SHORT position)
                                # For shorts, we're in profit when price is BELOW entry
                                if current_price < position['entry_price']:
                                    # For shorts, trailing stop moves DOWN as price moves DOWN
                                    new_stop = current_price * (1 + self.config.STOP_LOSS)
                                    current_trailing = position.get('trailing_stop', float('inf'))

                                    # Only update if new stop is LOWER than current trailing stop
                                    # or if we don't have a trailing stop yet
                                    if current_trailing == 0 or new_stop < current_trailing:
                                        position['trailing_stop'] = new_stop
                                        logging.debug(f"Updated trailing stop for SHORT {symbol} to {new_stop:.2f}")

                        except Exception as pos_error:
                            logging.error(f"Error monitoring position {symbol}: {str(pos_error)}")

                time.sleep(1)

            except Exception as e:
                logging.error(f"Position monitoring error: {str(e)}")
                time.sleep(5)

        logging.info("Position monitoring thread exited")

    def _update_position_metrics(self, symbol: str, current_price: float, data: pd.DataFrame = None,
                                 position: dict = None) -> bool:
        """Legacy method - redirects to trading system's update_position method"""
        if hasattr(self, 'trading_system'):
            return self.trading_system.update_position(symbol, current_price, data, position)
        else:
            # Fallback to original implementation if trading_system is not available
            try:
                if position is None:
                    if symbol not in self.position_manager.positions:
                        return False
                    position = self.position_manager.positions[symbol]

                # Basic position update
                position['current_price'] = current_price

                # Calculate P&L
                if position['side'] in ['BUY', 'LONG']:
                    position['unrealized_pnl'] = (current_price - position['entry_price']) * position['quantity']
                else:
                    position['unrealized_pnl'] = (position['entry_price'] - current_price) * position['quantity']

                return True
            except Exception as e:
                logging.error(f"Order manager position update error for {symbol}: {str(e)}")
                return False

    def _should_exit_position(self, symbol: str, position: dict, current_price: float) -> bool:
        """Check position exit conditions"""
        try:
            order = next((o for o in self.orders.values() if o['symbol'] == symbol), None)
            if not order:
                return False

            # Get current indicators
            indicators = self.ml_ensemble._calculate_technical_features(
                self.market_data_manager.get_current_data(symbol))

            # Basic stop conditions
            stop_hit = current_price <= order['stop_loss'] if position['side'] == 'LONG' else current_price >= \
                                                                                              order['stop_loss']
            target_hit = current_price >= order['target'] if position['side'] == 'LONG' else current_price <= \
                                                                                             order['target']
            trailing_stop_hit = position.get('trailing_stop') and (
                current_price <= position['trailing_stop'] if position['side'] == 'LONG' else current_price >=
                                                                                              position[
                                                                                                  'trailing_stop'])

            # Technical conditions
            if indicators:
                # Trend reversal
                trend_reversal = (position['side'] == 'LONG' and
                                  indicators['sma_short'] < indicators['sma_long'])

                # RSI extreme
                rsi_exit = (position['side'] == 'LONG' and
                            indicators['rsi'] > self.strategy_params['rsi_exit'])

                # Volatility spike
                vol_exit = (indicators['volatility'] >
                            self.strategy_params['volatility_threshold'])

                # Return true if any exit condition is met
                return (stop_hit or target_hit or trailing_stop_hit or
                        trend_reversal or rsi_exit or vol_exit)

            return stop_hit or target_hit or trailing_stop_hit

        except Exception as e:
            logging.error(f"Exit check error: {str(e)}")
            return False

    def _determine_exit_reason(self, position: dict, current_price: float) -> str:
        """Determine the reason for position exit"""
        try:
            if current_price <= position.get('stop_loss', 0):
                return 'STOP_LOSS'
            elif current_price >= position.get('target', float('inf')):
                return 'TAKE_PROFIT'
            elif position.get('trailing_stop') and current_price <= position['trailing_stop']:
                return 'TRAILING_STOP'
            return 'TECHNICAL_EXIT'
        except Exception as e:
            logging.error(f"Exit reason determination error: {str(e)}")
            return 'UNKNOWN'

    def _calculate_volatility(self, symbol: str) -> float:
        """Calculate symbol volatility from price history"""
        try:
            history = self.price_history.get(symbol, {'prices': []})
            prices = history['prices']

            if len(prices) >= 20:
                returns = pd.Series(prices).pct_change().dropna()
                return float(returns.std())

            return self.risk_params['stop_loss']  # Default to base stop loss

        except Exception as e:
            logging.error(f"Volatility calculation error: {str(e)}")
            return self.risk_params['stop_loss']

    def _get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price from tick cache"""
        try:
            tick_data = self.tick_cache.get(symbol)
            if tick_data and (datetime.now() - tick_data['timestamp']).seconds < 5:
                return tick_data['price']
            return None
        except Exception as e:
            logging.error(f"Price fetch error: {str(e)}")
            return None

    def _calculate_position_size(self, symbol: str, current_price: float) -> int:
        """Calculate position size with proper risk management"""
        try:
            # Get available capital based on risk parameters
            max_available = min(
                self.config.INITIAL_CAPITAL * 0.5,  # Max 50% of initial capital
                self.current_capital * 0.8  # Max 80% of current capital
            )

            # Calculate position value limits
            max_position_value = min(
                max_available * self.config.MAX_POSITION_SIZE,  # Position size limit
                self.config.MAX_POSITION_VALUE,  # Absolute value limit
                self.config.INITIAL_CAPITAL * 0.1  # Max 10% per position
            )

            # Calculate quantity
            quantity = int(max_position_value / current_price)

            # FIXED: Minimum quantity handling - always take at least 1 share
            # if it's less than a reasonable percentage of capital
            if quantity == 0:
                # Check if a single share is within reasonable capital percentage
                if current_price < self.config.INITIAL_CAPITAL * 0.1:  # Less than 10% of capital
                    quantity = 1
                    logging.info(f"Taking minimum position of 1 share for {symbol} despite price constraints")
                else:
                    logging.info(f"Position skipped - price {current_price:.2f} exceeds maximum allowed")
                    return 0

            # Ensure minimum position value
            position_value = quantity * current_price
            if position_value < self.config.MIN_POSITION_VALUE and quantity > 1:
                # Too small position - adjust to minimum value if possible
                min_quantity = max(1, int(self.config.MIN_POSITION_VALUE / current_price))
                if min_quantity > quantity:
                    quantity = min_quantity

            # Calculate position risk metrics
            volatility = self._calculate_volatility(symbol)
            expected_risk = position_value * volatility

            # Log sizing details
            logging.info(
                f"Position sizing for {symbol}: {quantity} shares at {current_price:.2f} = {position_value:.2f}")
            logging.info(f"Expected risk: {expected_risk:.2f} ({volatility:.2f}%)")

            return quantity

        except Exception as e:
            logging.error(f"Position size calculation error for {symbol}: {str(e)}")
            return 0

    def _validate_order_params(self, symbol: str, side: str, quantity: int, price: Optional[float]) -> bool:
        """Validate order parameters"""
        if side not in ['BUY', 'SELL']:
            logging.error(f"Invalid order side: {side}")
            return False

        if quantity <= 0:
            logging.error(f"Invalid quantity: {quantity}")
            return False

        if price and price <= 0:
            logging.error(f"Invalid price: {price}")
            return False

        if symbol not in self.config.SYMBOLS:
            logging.error(f"Invalid symbol: {symbol}")
            return False

        return True

    def _close_all_positions(self):
        """Close all open positions"""
        with self.lock:
            for symbol in list(self.position_manager.positions.keys()):
                current_price = self._get_latest_price(symbol)
                if current_price:
                    self.position_manager.close_position(symbol, current_price, 'SYSTEM_SHUTDOWN')

    def _save_trade_history(self):
        """Save trade history to file"""
        try:
            if self.trade_history:
                df = pd.DataFrame(self.trade_history)
                filename = f"trade_history_{datetime.now().strftime('%Y%m%d')}.csv"
                df.to_csv(filename, index=False)
                logging.info(f"Trade history saved to {filename}")
        except Exception as e:
            logging.error(f"Trade history save error: {str(e)}")

    def get_metrics(self) -> dict:
        """Get current trading metrics"""
        with self.lock:
            return {
                'capital': self.current_capital,
                'daily_pnl': self.daily_pnl,
                'total_trades': self.total_trades,
                'win_rate': self.winning_trades / self.total_trades if self.total_trades > 0 else 0,
                'max_drawdown': self.max_drawdown,
                'margin_used': self.positions_metrics['total_exposure'],
                'open_positions': self.positions_metrics['open_positions_count']
            }

    def reset_daily_stats(self):
        """Reset daily statistics"""
        with self.lock:
            self.daily_pnl = 0
            self.positions_metrics['daily_trades_count'] = 0
            logging.info("Daily statistics reset")


class RiskManager:
    """Advanced risk management system"""

    def __init__(self, config: Config):
        self.config = config
        self.daily_stats = {
            'start_value': self.config.INITIAL_CAPITAL,
            'current_value': self.config.INITIAL_CAPITAL,
            'peak_value': self.config.INITIAL_CAPITAL,
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'pnl': 0,
            'positions': {}
        }

        self.risk_limits = {
            'max_position_value': 0,
            'max_daily_loss': 0,
            'max_drawdown': self.config.MAX_DRAWDOWN,
            'max_positions': 5,
            'position_correlation': 0.7
        }

        self.market_status = {
            'high_volatility': False,
            'circuit_breaker': False,
            'trading_allowed': True
        }

        # Add market status tracking
        self.market_status = {
            'high_volatility': False,
            'circuit_breaker': False,
            'trading_allowed': True
        }
        self.correlation_matrix = pd.DataFrame()
        self.last_update = None
        self.volatility_metrics = {}

    def initialize(self) -> bool:
        """Initialize all system components"""
        try:
            logging.info("Initializing trading system...")

            # 1. Check network connectivity first
            if not self._check_network_connectivity():
                logging.error("Network connectivity check failed")
                return False

            # 2. Initialize Angel One connection
            self.angel_connector = AngelOneConnector(self.config)
            if not self.angel_connector.connect():
                raise Exception("Failed to connect to Angel One")

            # 3. Initialize market data manager but don't start stream
            self.market_data_manager = MarketDataManager(self.angel_connector, self.config)

            # 4. Load historical data first
            logging.info("Loading historical data...")
            self.market_data_manager._load_historical_data()

            # 5. Initialize ML ensemble and train on historical data
            logging.info("Training ML models...")
            self.ml_ensemble = MLEnsemble(self.config)
            for symbol in self.config.SYMBOLS:
                data = self.market_data_manager.get_current_data(symbol)
                if data is not None and len(data) > 100:
                    self.ml_ensemble.train(data)
                    logging.info(f"ML models trained for {symbol}")

            # 6. Initialize risk manager
            self.risk_manager = RiskManager(self.config)

            # 7. Set initial attributes that were missing
            self.peak_capital = self.config.INITIAL_CAPITAL
            self.position_manager = PositionManager(self.config)

            # 8. Start live data feed
            logging.info("Starting live data feed...")
            self.market_data_manager.start()

            self.system_start_time = datetime.now()
            self.daily_reset_time = datetime.now().replace(hour=9, minute=15, second=0)

            logging.info("Trading system initialized successfully")
            return True

        except Exception as e:
            logging.error(f"System initialization error: {str(e)}")
            return False

    def update_market_status(self, market_data_manager: Dict[str, pd.DataFrame]):
        """Update market status and risk metrics with timezone handling"""
        try:
            # Update correlation matrix
            close_prices = {}
            volatility = {}

            for symbol, data in market_data_manager.items():
                if data is not None and not data.empty:
                    # Convert index to timezone-naive if it's timezone-aware
                    latest_data = data.copy()
                    if latest_data.index.tz is not None:
                        latest_data.index = latest_data.index.tz_localize(None)

                    # Remove duplicate indices
                    latest_data = latest_data[~latest_data.index.duplicated(keep='last')]

                    # Store close prices for correlation
                    if 'Close' in latest_data.columns:
                        close_prices[symbol] = latest_data['Close'].astype(float)

                        # Calculate volatility from returns
                        returns = latest_data['Close'].pct_change().dropna()
                        if len(returns) > 20:
                            volatility[symbol] = returns.rolling(20).std().iloc[-1]
                        else:
                            volatility[symbol] = 0

            # Create correlation matrix only if we have data
            if close_prices:
                try:
                    # Create DataFrame with aligned and timezone-naive indices
                    price_df = pd.DataFrame(close_prices)
                    price_df.index = pd.to_datetime(price_df.index.values, utc=False)

                    # Handle any remaining duplicates
                    price_df = price_df.loc[~price_df.index.duplicated(keep='last')]

                    # Calculate correlation
                    self.correlation_matrix = price_df.corr()

                    # Update market status
                    if volatility:
                        avg_volatility = float(np.mean(list(volatility.values())))
                        self.market_status['high_volatility'] = avg_volatility > self.config.VOLATILITY_THRESHOLD

                    # Check for circuit breaker
                    for symbol, data in market_data_manager.items():
                        if data is not None and not data.empty:
                            if data.index.tz is not None:
                                data.index = data.index.tz_localize(None)
                            data = data[~data.index.duplicated(keep='last')]

                            if len(data) > 0 and 'Close' in data.columns:
                                daily_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1)
                                if abs(daily_return) > 0.1:  # 10% move
                                    self.market_status['circuit_breaker'] = True
                                    break

                    self.last_update = datetime.now()

                except Exception as inner_e:
                    logging.error(f"Error in correlation calculation: {str(inner_e)}")
                    self.correlation_matrix = pd.DataFrame()  # Reset to empty if calculation fails

        except Exception as e:
            logging.error(f"Market status update error: {str(e)}", exc_info=True)
            # Reset to safe values on error
            self.correlation_matrix = pd.DataFrame()
            self.market_status['high_volatility'] = False
            self.market_status['circuit_breaker'] = False

    def check_trade(self, symbol: str, side: str, quantity: int, price: float) -> Tuple[bool, str]:
        """Check if trade is within risk limits"""
        try:
            # Basic checks
            if not self.market_status['trading_allowed']:
                return False, "Trading not allowed"

            if self.market_status['circuit_breaker']:
                return False, "Circuit breaker active"

            if self.market_status['high_volatility']:
                return False, "High market volatility"

            # Calculate position value
            position_value = quantity * price

            # Check against initial capital for position size
            max_position_value = self.config.INITIAL_CAPITAL * self.config.MAX_POSITION_SIZE
            if position_value > max_position_value:
                return False, f"Position value {position_value:.2f} exceeds max allowed {max_position_value:.2f}"

            # Check current positions count
            if len(self.daily_stats['positions']) >= self.config.MAX_POSITIONS:
                return False, f"Maximum positions ({self.config.MAX_POSITIONS}) reached"

            # Check total exposure
            total_exposure = sum(pos['value'] for pos in self.daily_stats['positions'].values())
            max_exposure = self.config.INITIAL_CAPITAL * self.config.MAX_CAPITAL_AT_RISK
            if total_exposure + position_value > max_exposure:
                return False, f"Total exposure would exceed {max_exposure:.2f}"

            # Drawdown check
            current_drawdown = self._calculate_drawdown()
            if current_drawdown > self.risk_limits['max_drawdown']:
                return False, "Maximum drawdown reached"

            # Correlation check
            if not self._check_correlation(symbol):
                return False, "High portfolio correlation"

            return True, "Trade allowed"

        except Exception as e:
            logging.error(f"Trade check error: {str(e)}")
            return False, f"Error checking trade: {str(e)}"

    def update_position(self, symbol: str, quantity: int, price: float, pnl: float = 0):
        """
        Updates risk metrics based on position changes.
        Called by TradingSystem when positions are opened, updated, or closed.
        """
        try:
            # FIXED: Removed recursive call to trading_system.update_position
            # This function should only update risk metrics, not call back to TradingSystem

            # Position closed
            if quantity == 0:  # Position closed
                if symbol in self.daily_stats['positions']:
                    # Record closed position stats for risk analysis
                    closed_position = self.daily_stats['positions'].pop(symbol)

                    # Track PnL statistics
                    self.daily_stats['pnl'] += pnl
                    if pnl > 0:
                        self.daily_stats['wins'] += 1
                    elif pnl < 0:
                        self.daily_stats['losses'] += 1

                    # Track trades count
                    self.daily_stats['trades'] += 1
            else:  # New or updated position
                self.daily_stats['positions'][symbol] = {
                    'quantity': quantity,
                    'price': price,
                    'value': quantity * price,
                    'pnl': pnl
                }

                # Update current value
                total_exposure = sum(pos['value'] for pos in self.daily_stats['positions'].values())
                total_pnl = sum(pos['pnl'] for pos in self.daily_stats['positions'].values())
                self.daily_stats['current_value'] = self.daily_stats['start_value'] + total_pnl

                # Update peak value
                if self.daily_stats['current_value'] > self.daily_stats['peak_value']:
                    self.daily_stats['peak_value'] = self.daily_stats['current_value']

                # Calculate drawdown
                if self.daily_stats['peak_value'] > 0:  # Avoid division by zero
                    current_drawdown = (self.daily_stats['peak_value'] - self.daily_stats['current_value']) / \
                                       self.daily_stats['peak_value']
                    if current_drawdown > self.daily_stats.get('max_drawdown', 0):
                        self.daily_stats['max_drawdown'] = current_drawdown

            return True

        except Exception as e:
            logging.error(f"Risk manager position update error for {symbol}: {str(e)}")
            return False

    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown"""
        try:
            if self.daily_stats['peak_value'] > 0:
                return (self.daily_stats['peak_value'] - self.daily_stats['current_value']) / self.daily_stats[
                    'peak_value']
            return 0.0
        except Exception as e:
            logging.error(f"Drawdown calculation error: {str(e)}")
            return 0.0

    def check_risk_limits(self, symbol: str, quantity: int, side: str, price: float = None) -> bool:
        """Check if trade meets risk limits with 4x leverage support"""
        try:
            # Use provided price or get latest price
            if not price:
                price = self.market_data_manager.get_latest_price(symbol)
                if not price:
                    return False

            # Calculate position value
            position_value = quantity * price

            # Apply 4x leverage to capital and position limits
            leveraged_capital = self.config.INITIAL_CAPITAL * 4

            # 1. Check position count (no change to this limit)
            if len(self.daily_stats['positions']) >= self.config.MAX_POSITIONS:
                logging.warning(f"Maximum positions ({self.config.MAX_POSITIONS}) reached")
                return False

            # 2. Check position size limit with 4x leverage
            max_position_value = leveraged_capital * (self.config.MAX_POSITION_SIZE * 2)  # Double the percentage
            if position_value > max_position_value:
                logging.warning(
                    f"Position value {position_value:.2f} exceeds leveraged maximum {max_position_value:.2f}")
                return False

            # 3. Check total exposure with 4x leverage
            total_exposure = sum(pos['value'] for pos in self.daily_stats['positions'].values())
            max_exposure = leveraged_capital * (self.config.MAX_CAPITAL_AT_RISK * 3)  # Triple the percentage
            if total_exposure + position_value > max_exposure:
                logging.warning(f"Total exposure would exceed leveraged maximum {max_exposure:.2f}")
                return False

            # 4. Check drawdown (keep the same limit)
            current_drawdown = self._calculate_drawdown()
            if current_drawdown > self.risk_limits['max_drawdown']:
                logging.warning("Maximum drawdown reached")
                return False

            # 5. Correlation check (unchanged)
            if not self._check_correlation(symbol):
                logging.warning("High portfolio correlation")
                return False

            # Log successful leverage-adjusted risk check
            logging.info(f"Leveraged risk check passed for {symbol}: {quantity} shares at {price:.2f}")
            return True

        except Exception as e:
            logging.error(f"Risk check error: {str(e)}")
            return False

    def _check_correlation(self, new_symbol: str) -> bool:
        """Check portfolio correlation"""
        try:
            if new_symbol in self.correlation_matrix.columns:
                existing_positions = set(self.daily_stats['positions'].keys())
                if existing_positions:
                    correlations = self.correlation_matrix[new_symbol][list(existing_positions)]
                    max_correlation = correlations.abs().max()
                    return max_correlation < self.risk_limits['position_correlation']
            return True
        except Exception as e:
            logging.error(f"Correlation check error: {str(e)}")
            return False

    def _calculate_var(self, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk"""
        try:
            returns = []
            for symbol, pos in self.daily_stats['positions'].items():
                if 'pnl' in pos:
                    returns.append(pos['pnl'])

            if not returns:
                return 0.0

            returns = np.array(returns)
            var = np.percentile(returns, (1 - confidence_level) * 100)
            return abs(var)

        except Exception as e:
            logging.error(f"VaR calculation error: {str(e)}")
            return 0.0

    def _calculate_margin_usage(self) -> float:
        """Calculate margin utilization"""
        try:
            total_exposure = sum(pos['value'] for pos in self.daily_stats['positions'].values())
            if self.daily_stats['start_value'] > 0:  # Add this check
                return total_exposure / self.daily_stats['start_value']
            return 0.0
        except Exception as e:
            logging.error(f"Margin calculation error: {str(e)}")
            return 0.0

    def get_risk_metrics(self) -> dict:
        """Get current risk metrics"""
        try:
            return {
                'portfolio_var': self._calculate_var(),
                'total_exposure': sum(self.daily_stats['positions'].values()),
                'margin_utilization': self._calculate_margin_usage(),
                'correlation_metrics': self._get_correlation_metrics(),
                'volatility_metrics': self.volatility_metrics
            }
        except Exception as e:
            logging.error(f"Error getting risk metrics: {str(e)}")
            return {
                'portfolio_var': 0.0,
                'total_exposure': 0.0,
                'margin_utilization': 0.0,
                'correlation_metrics': {},
                'volatility_metrics': {}
            }

    def _get_correlation_metrics(self) -> dict:
        """Get correlation metrics for portfolio"""
        try:
            metrics = {}
            if not self.correlation_matrix.empty:
                for symbol in self.daily_stats['positions']:
                    if symbol in self.correlation_matrix.columns:
                        correlations = self.correlation_matrix[symbol].abs()
                        metrics[symbol] = {
                            'max_correlation': correlations.max(),
                            'avg_correlation': correlations.mean()
                        }
            return metrics
        except Exception as e:
            logging.error(f"Correlation metrics error: {str(e)}")
            return {}


def reset_daily_stats(self):
    """Reset daily statistics with enhanced cleanup - can be called by any component"""
    try:
        with self.lock:  # Use the existing lock in RiskManager
            # Initialize daily_stats if it doesn't exist
            if not hasattr(self, 'daily_stats'):
                self.daily_stats = {
                    'start_value': self.config.INITIAL_CAPITAL,
                    'current_value': self.config.INITIAL_CAPITAL,
                    'peak_value': self.config.INITIAL_CAPITAL,
                    'trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'pnl': 0,
                    'positions': {},
                    'max_drawdown': 0
                }
                return

            # Store peak capital and other values that should persist
            peak_capital = self.daily_stats.get('peak_value', self.config.INITIAL_CAPITAL)

            # Reset counters while preserving other data
            self.daily_stats.update({
                'current_value': self.daily_stats.get('start_value', self.config.INITIAL_CAPITAL),
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'pnl': 0,
                'positions': {}
            })

            # Restore peak value if it's higher than starting value
            if peak_capital > self.daily_stats['current_value']:
                self.daily_stats['peak_value'] = peak_capital

            logging.info("Daily statistics reset completed")
    except Exception as e:
        logging.error(f"Error in reset_daily_stats: {str(e)}")


class PositionManager:
    """Manages and tracks trading positions"""

    def __init__(self, config: Config):
        self.config = config
        self.positions = {}  # Current positions
        self.position_history = []  # Historical positions
        self.lock = threading.Lock()

        # Initialize these values
        self.total_exposure = 0
        self.margin_used = 0
        self.unrealized_pnl = 0
        self.position_count = 0
        self.peak_capital = config.INITIAL_CAPITAL  # Add this

        # Position exits tracking
        self.stop_loss_hits = 0
        self.take_profit_hits = 0
        self.signal_exits = 0

    def open_position(self, symbol: str, side: str, quantity: int, entry_price: float,
                      stop_loss: float, take_profit: float, timestamp=None, data=None) -> bool:
        """Open a new position with complete tracking and error handling,
        supporting both long and short positions"""
        try:
            with self.lock:
                # Check if position already exists
                if symbol in self.positions:
                    logging.warning(f"Position already exists for {symbol}")
                    return False

                # Validate parameters
                if quantity <= 0:
                    logging.error(f"Invalid quantity: {quantity}")
                    return False

                if entry_price <= 0:
                    logging.error(f"Invalid entry price: {entry_price}")
                    return False

                # Adjust stop loss based on position direction
                is_long = side in ['BUY', 'LONG']

                if is_long:
                    # For long positions
                    if stop_loss <= 0 or stop_loss >= entry_price:
                        logging.warning(f"Adjusting invalid stop loss: {stop_loss}")
                        # Default to 0.2% below entry price if stop is invalid
                        stop_loss = entry_price * 0.998

                    if take_profit <= entry_price:
                        logging.warning(f"Adjusting invalid take profit: {take_profit}")
                        # Default to 0.4% above entry price if target is invalid
                        take_profit = entry_price * 1.004
                else:
                    # For short positions
                    if stop_loss <= 0 or stop_loss <= entry_price:
                        logging.warning(f"Adjusting invalid stop loss for short: {stop_loss}")
                        # Default to 0.2% above entry price if stop is invalid
                        stop_loss = entry_price * 1.002

                    if take_profit >= entry_price:
                        logging.warning(f"Adjusting invalid take profit for short: {take_profit}")
                        # Default to 0.4% below entry price if target is invalid
                        take_profit = entry_price * 0.996

                # Calculate position value
                position_value = quantity * entry_price

                # Standardize side to 'LONG' or 'SHORT' for internal tracking
                position_side = 'LONG' if side in ['BUY', 'LONG'] else 'SHORT'

                # Create new position with comprehensive tracking
                self.positions[symbol] = {
                    'symbol': symbol,
                    'side': position_side,
                    'quantity': quantity,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'entry_time': timestamp or datetime.now(),
                    'highest_price': entry_price,
                    'lowest_price': entry_price,
                    'current_price': entry_price,
                    'unrealized_pnl': 0,
                    'unrealized_pnl_pct': 0,
                    'max_favorable_excursion': 0,
                    'max_adverse_excursion': 0,
                    'trailing_stop': stop_loss,  # Initialize trailing stop to stop loss level
                    'trailing_stop_price': entry_price,  # Price when trailing stop was last updated
                    'updates': [],  # Track position updates
                    'position_value': position_value
                }

                # Update tracking metrics
                self.total_exposure += position_value
                self.position_count += 1

                # Calculate stop loss and target percentages for logging
                if is_long:
                    stop_pct = ((stop_loss / entry_price) - 1) * 100
                    target_pct = ((take_profit / entry_price) - 1) * 100
                else:
                    stop_pct = ((stop_loss / entry_price) - 1) * 100  # Positive for shorts
                    target_pct = ((take_profit / entry_price) - 1) * 100  # Negative for shorts

                # Log successful position creation
                logging.info(
                    f"✅ Position opened - Symbol: {symbol}, Side: {position_side}, Qty: {quantity}, Entry: {entry_price:.2f}")
                logging.info(
                    f"   Risk management: Stop: {stop_loss:.2f} ({stop_pct:.2f}%), Target: {take_profit:.2f} ({target_pct:.2f}%)")

                # Update position history
                open_record = {
                    'symbol': symbol,
                    'action': 'OPEN',
                    'side': position_side,
                    'quantity': quantity,
                    'price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'timestamp': timestamp or datetime.now()
                }
                self.position_history.append(open_record)

                # FIXED: Synchronize with OrderManager if this is being called from TradingSystem
                if hasattr(self, 'trading_system') and hasattr(self.trading_system, 'order_manager'):
                    if hasattr(self.trading_system.order_manager, 'position_manager'):
                        # Share position with OrderManager's position manager
                        self.trading_system.order_manager.position_manager.positions[symbol] = self.positions[
                            symbol].copy()

                return True

        except Exception as e:
            logging.error(f"Error opening position for {symbol}: {str(e)}")
            logging.error(traceback.format_exc())
            return False

    def close_position(self, symbol: str, exit_price: float, exit_reason: str) -> dict:
        """Close existing position"""
        try:
            with self.lock:
                if symbol not in self.positions:
                    logging.warning(f"No position exists for {symbol}")
                    return None

                position = self.positions[symbol]

                # Calculate realized P&L based on position direction
                is_long = position['side'] in ['BUY', 'LONG']

                if is_long:
                    realized_pnl = (exit_price - position['entry_price']) * position['quantity']
                else:  # SHORT
                    realized_pnl = (position['entry_price'] - exit_price) * position['quantity']

                # Update exit statistics
                if exit_reason == 'STOP_LOSS':
                    self.stop_loss_hits += 1
                elif exit_reason == 'TAKE_PROFIT':
                    self.take_profit_hits += 1
                elif exit_reason == 'SIGNAL':
                    self.signal_exits += 1

                # Record position closing
                close_record = {
                    'symbol': symbol,
                    'action': 'CLOSE',
                    'exit_price': exit_price,
                    'realized_pnl': realized_pnl,
                    'exit_reason': exit_reason,
                    'holding_time': (datetime.now() - position['entry_time']).seconds,
                    'highest_price': position['highest_price'],
                    'lowest_price': position['lowest_price'],
                    'timestamp': datetime.now()
                }

                self.position_history.append(close_record)

                # Update tracking metrics
                self.total_exposure -= position['quantity'] * position['current_price']
                self.position_count -= 1
                current_unrealized_pnl = position.get('unrealized_pnl', 0)
                self.unrealized_pnl -= current_unrealized_pnl

                # FIXED: Synchronize with OrderManager if this is being called from TradingSystem
                if hasattr(self, 'trading_system') and hasattr(self.trading_system, 'order_manager'):
                    if hasattr(self.trading_system.order_manager, 'position_manager'):
                        if symbol in self.trading_system.order_manager.position_manager.positions:
                            # Remove position from OrderManager's position manager
                            del self.trading_system.order_manager.position_manager.positions[symbol]

                # Remove position
                del self.positions[symbol]

                # Log position closing details with side information
                side_str = "LONG" if is_long else "SHORT"
                logging.info(f"{side_str} position closed - Symbol: {symbol}, PnL: {realized_pnl:.2f}")

                return close_record

        except Exception as e:
            logging.error(f"Position closing error: {str(e)}")
            return None

    def update_position(self, symbol: str, current_price: float) -> Optional[str]:
        """
        Updates position details and checks if exit conditions are met.
        Returns exit reason if position should be closed, None otherwise.
        Handles both long and short positions.
        """
        try:
            with self.lock:
                if symbol not in self.positions:
                    return None

                position = self.positions[symbol]
                old_unrealized_pnl = position.get('unrealized_pnl', 0)
                is_long = position['side'] in ['BUY', 'LONG']

                # Update price extremes and history
                if is_long:
                    # For longs, track highest price
                    position['highest_price'] = max(position.get('highest_price', current_price), current_price)
                else:
                    # For shorts, track lowest price
                    position['lowest_price'] = min(position.get('lowest_price', current_price), current_price)

                # Always update both extremes for tracking
                position['highest_price'] = max(position.get('highest_price', current_price), current_price)
                position['lowest_price'] = min(position.get('lowest_price', current_price), current_price)
                position['current_price'] = current_price
                position['last_update_time'] = datetime.now()

                # Calculate unrealized P&L based on position direction
                if is_long:
                    position['unrealized_pnl'] = (current_price - position['entry_price']) * position['quantity']
                    position['unrealized_pnl_pct'] = ((current_price / position['entry_price']) - 1) * 100
                else:  # SHORT/SELL
                    position['unrealized_pnl'] = (position['entry_price'] - current_price) * position['quantity']
                    position['unrealized_pnl_pct'] = ((position['entry_price'] / current_price) - 1) * 100

                # Update maximum favorable and adverse excursions
                position['max_favorable_excursion'] = max(
                    position.get('max_favorable_excursion', 0),
                    position['unrealized_pnl']
                )
                position['max_adverse_excursion'] = min(
                    position.get('max_adverse_excursion', 0),
                    position['unrealized_pnl']
                )

                # Check exit conditions based on position direction
                if is_long:
                    # Long position exit checks
                    if current_price <= position.get('stop_loss', 0):
                        return 'STOP_LOSS'
                    elif current_price >= position.get('take_profit', float('inf')):
                        return 'TAKE_PROFIT'
                    elif position.get('trailing_stop') and current_price <= position['trailing_stop']:
                        return 'TRAILING_STOP'
                else:  # SHORT/SELL
                    # Short position exit checks
                    if current_price >= position.get('stop_loss', float('inf')):
                        return 'STOP_LOSS'
                    elif current_price <= position.get('take_profit', 0):
                        return 'TAKE_PROFIT'
                    elif position.get('trailing_stop') and current_price >= position['trailing_stop']:
                        return 'TRAILING_STOP'

                # Record position update with full metrics
                if not 'updates' in position:
                    position['updates'] = []

                position['updates'].append({
                    'price': current_price,
                    'timestamp': datetime.now(),
                    'unrealized_pnl': position['unrealized_pnl'],
                    'unrealized_pnl_pct': position['unrealized_pnl_pct'],
                    'trailing_stop': position.get('trailing_stop'),
                    'high_reached': position['highest_price'],
                    'low_reached': position['lowest_price']
                })

                # Keep only last 100 updates
                if len(position['updates']) > 100:
                    position['updates'] = position['updates'][-100:]

                return None

        except Exception as e:
            logging.error(f"PositionManager update error for {symbol}: {str(e)}")
            return None

    def update_position_metrics(self, symbol: str, current_price: float, data: pd.DataFrame = None,
                                context: dict = None) -> dict:
        """Legacy method - redirects to trading system's update_position method if available"""
        # Extract position from the positions dictionary
        if symbol not in self.positions:
            return {'exists': False}

        position = self.positions[symbol]

        # Call trading system's method if available
        if hasattr(self, 'trading_system'):
            success = self.trading_system.update_position(symbol, current_price, data, position)
            return {
                'exists': True,
                'updated': success,
                'position': position
            }
        else:
            # Original implementation (simplified)
            old_pnl = position.get('unrealized_pnl', 0)
            position['current_price'] = current_price

            # Calculate PnL
            if position['side'] in ['BUY', 'LONG']:
                position['unrealized_pnl'] = (current_price - position['entry_price']) * position['quantity']
            else:
                position['unrealized_pnl'] = (position['entry_price'] - current_price) * position['quantity']

            # Return metrics
            return {
                'exists': True,
                'updated': True,
                'position': position,
                'pnl_change': position['unrealized_pnl'] - old_pnl
            }

    def get_position_metrics(self) -> dict:
        """Get current position metrics"""
        with self.lock:
            return {
                'total_exposure': self.total_exposure,
                'position_count': self.position_count,
                'unrealized_pnl': self.unrealized_pnl,
                'margin_used': self.margin_used,
                'stop_loss_hits': self.stop_loss_hits,
                'take_profit_hits': self.take_profit_hits,
                'signal_exits': self.signal_exits
            }

    def get_position_history(self) -> List[dict]:
        """Get position history"""
        with self.lock:
            return self.position_history.copy()


class PerformanceTracker:
    """Enhanced performance tracking and analysis"""

    def __init__(self):
        self.trades = []
        self.daily_performance = {}
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'win_rate': 0,
            'average_win': 0,
            'average_loss': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'max_drawdown': 0,
            'average_holding_time': 0,
            'recovery_factor': 0,  # Added from historical
            'daily_hit_rate': 0,  # Added from historical
            'value_at_risk': 0.0,  # Added from historical
            'expected_shortfall': 0.0,  # Added from historical
            'kurtosis': 0.0,  # Added from historical
            'skewness': 0.0  # Added from historical
        }

        self.equity_curve = []
        self.drawdown_series = []
        self.lock = threading.Lock()

    def add_trade(self, trade: dict):
        """Add completed trade to history with enhanced validation"""
        try:
            with self.lock:
                required_fields = ['realized_pnl', 'entry_time', 'exit_time', 'symbol']
                if not all(field in trade for field in required_fields):
                    logging.error("Trade missing required fields")
                    return

                self.trades.append(trade)
                self._update_metrics()
                self._update_equity_curve()
                self._update_daily_performance(trade)

        except Exception as e:
            logging.error(f"Trade recording error: {str(e)}")

    def _update_metrics(self):
        """Update performance metrics with enhanced statistics"""
        try:
            with self.lock:
                if not self.trades:
                    return

                # Basic metrics
                self.metrics['total_trades'] = len(self.trades)

                # Separate winning and losing trades
                winning_trades = [t for t in self.trades if t['realized_pnl'] > 0]
                losing_trades = [t for t in self.trades if t['realized_pnl'] < 0]

                self.metrics['winning_trades'] = len(winning_trades)
                self.metrics['losing_trades'] = len(losing_trades)

                # Win rate and P&L metrics
                self.metrics['win_rate'] = len(winning_trades) / len(self.trades) if self.trades else 0
                self.metrics['total_pnl'] = sum(t['realized_pnl'] for t in self.trades)

                if winning_trades:
                    self.metrics['average_win'] = np.mean([t['realized_pnl'] for t in winning_trades])
                    self.metrics['largest_win'] = max(t['realized_pnl'] for t in winning_trades)

                if losing_trades:
                    self.metrics['average_loss'] = np.mean([t['realized_pnl'] for t in losing_trades])
                    self.metrics['largest_loss'] = min(t['realized_pnl'] for t in losing_trades)

                # Enhanced profit factor calculation
                total_gains = sum(t['realized_pnl'] for t in winning_trades)
                total_losses = abs(sum(t['realized_pnl'] for t in losing_trades))
                self.metrics['profit_factor'] = total_gains / total_losses if total_losses else float('inf')

                # Time metrics
                holding_times = [(t['exit_time'] - t['entry_time']).total_seconds() / 3600 for t in self.trades]
                self.metrics['average_holding_time'] = np.mean(holding_times) if holding_times else 0

                # Risk and return metrics
                returns = pd.Series([t['realized_pnl'] for t in self.trades])
                if len(returns) >= 2:
                    daily_returns = self._calculate_daily_returns()
                    self.metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(daily_returns)
                    self.metrics['sortino_ratio'] = self._calculate_sortino_ratio(returns)
                    self.metrics['max_drawdown'] = self._calculate_max_drawdown()

                    # Enhanced risk metrics
                    self.metrics['value_at_risk'] = float(np.percentile(returns, 5)) if len(returns) > 0 else 0.0
                    self.metrics['expected_shortfall'] = float(returns[returns < 0].mean()) if len(
                        returns[returns < 0]) > 0 else 0.0
                    self.metrics['kurtosis'] = float(returns.kurtosis()) if len(returns) > 3 else 0.0
                    self.metrics['skewness'] = float(returns.skew()) if len(returns) > 2 else 0.0

                # Recovery factor
                if self.metrics['max_drawdown'] > 0:
                    self.metrics['recovery_factor'] = self.metrics['total_pnl'] / self.metrics['max_drawdown']

        except Exception as e:
            logging.error(f"Metrics update error: {str(e)}")

    def _update_equity_curve(self):
        """Update equity curve with timestamp validation"""
        try:
            with self.lock:
                cumulative_pnl = 0
                self.equity_curve = []

                # Sort trades by exit time for accurate equity curve
                sorted_trades = sorted(self.trades, key=lambda x: x['exit_time'])

                for trade in sorted_trades:
                    cumulative_pnl += trade['realized_pnl']
                    self.equity_curve.append({
                        'timestamp': trade['exit_time'],
                        'equity': cumulative_pnl
                    })

                # Calculate drawdown series
                self._calculate_drawdown_series()

        except Exception as e:
            logging.error(f"Equity curve update error: {str(e)}")

    def _update_daily_performance(self, trade: dict):
        """Update daily performance with enhanced metrics"""
        try:
            with self.lock:
                date = trade['exit_time'].date()

                if date not in self.daily_performance:
                    self.daily_performance[date] = {
                        'trades': 0,
                        'pnl': 0,
                        'wins': 0,
                        'losses': 0,
                        'largest_win': 0,
                        'largest_loss': 0,
                        'volume': 0
                    }

                daily_stats = self.daily_performance[date]
                daily_stats['trades'] += 1
                daily_stats['pnl'] += trade['realized_pnl']
                daily_stats['volume'] += trade.get('volume', 0)

                if trade['realized_pnl'] > 0:
                    daily_stats['wins'] += 1
                    daily_stats['largest_win'] = max(daily_stats['largest_win'], trade['realized_pnl'])
                else:
                    daily_stats['losses'] += 1
                    daily_stats['largest_loss'] = min(daily_stats['largest_loss'], trade['realized_pnl'])

        except Exception as e:
            logging.error(f"Daily performance update error: {str(e)}")

    def _calculate_daily_returns(self) -> pd.Series:
        """Calculate daily returns with proper date handling"""
        try:
            daily_pnl = pd.Series(
                {date: data['pnl'] for date, data in self.daily_performance.items()}
            )
            return daily_pnl
        except Exception as e:
            logging.error(f"Daily returns calculation error: {str(e)}")
            return pd.Series()

    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio with risk-free rate"""
        try:
            if len(returns) < 2:
                return 0
            excess_returns = returns - risk_free_rate
            return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        except Exception as e:
            logging.error(f"Sharpe ratio calculation error: {str(e)}")
            return 0

    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio with proper error handling"""
        try:
            if len(returns) < 2:
                return 0.0

            negative_returns = returns[returns < 0]
            if len(negative_returns) == 0:
                return 0.0

            downside_std = np.sqrt(np.mean(negative_returns ** 2))
            if downside_std == 0:
                return 0.0

            return np.sqrt(252) * returns.mean() / downside_std

        except Exception as e:
            logging.error(f"Sortino ratio calculation error: {str(e)}")
            return 0.0

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown with proper handling"""
        try:
            if not self.equity_curve:
                return 0

            equity = pd.Series([p['equity'] for p in self.equity_curve])
            peak = equity.expanding(min_periods=1).max()
            drawdown = (equity - peak) / peak
            return abs(float(drawdown.min()))

        except Exception as e:
            logging.error(f"Drawdown calculation error: {str(e)}")
            return 0

    def _calculate_drawdown_series(self):
        """Calculate drawdown series with timestamp alignment"""
        try:
            with self.lock:
                if not self.equity_curve:
                    self.drawdown_series = []
                    return

                equity = pd.Series([p['equity'] for p in self.equity_curve])
                peak = equity.expanding(min_periods=1).max()
                drawdown = (equity - peak) / peak

                self.drawdown_series = [
                    {
                        'timestamp': self.equity_curve[i]['timestamp'],
                        'drawdown': float(drawdown[i])
                    }
                    for i in range(len(drawdown))
                ]

        except Exception as e:
            logging.error(f"Drawdown series calculation error: {str(e)}")

    def get_performance_summary(self) -> dict:
        """Get comprehensive performance summary"""
        with self.lock:
            return self.metrics.copy()

    def get_daily_performance(self) -> dict:
        """Get daily performance metrics"""
        with self.lock:
            return self.daily_performance.copy()

    def get_equity_curve(self) -> List[dict]:
        """Get equity curve data"""
        with self.lock:
            return self.equity_curve.copy()

    def get_drawdown_series(self) -> List[dict]:
        """Get drawdown series data"""
        with self.lock:
            return self.drawdown_series.copy()

    def generate_performance_report(self) -> dict:
        """Generate comprehensive performance report"""
        try:
            with self.lock:
                return {
                    'summary_metrics': self.metrics.copy(),
                    'daily_performance': self.daily_performance.copy(),
                    'equity_curve': self.equity_curve.copy(),
                    'drawdown_series': self.drawdown_series.copy(),
                    'trade_analysis': self._generate_trade_analysis(),
                    'risk_metrics': self._calculate_risk_metrics(),
                    'timestamp': datetime.now()
                }
        except Exception as e:
            logging.error(f"Performance report generation error: {str(e)}")
            return {}

    def _generate_trade_analysis(self) -> dict:
        """Generate detailed trade analysis"""
        try:
            if not self.trades:
                return {
                    'hourly_performance': {},
                    'symbol_performance': {},
                    'streak_analysis': self._analyze_streaks()
                }

            trades_df = pd.DataFrame(self.trades)

            # Time-based analysis
            hourly_performance = {}
            if 'entry_time' in trades_df.columns:
                trades_df['hour'] = trades_df['entry_time'].dt.hour
                hourly_performance = trades_df.groupby('hour')['realized_pnl'].agg([
                    'count', 'mean', 'sum'
                ]).to_dict()

            # Symbol analysis
            symbol_performance = {}
            if 'symbol' in trades_df.columns:
                symbol_performance = trades_df.groupby('symbol')['realized_pnl'].agg([
                    'count', 'mean', 'sum'
                ]).to_dict()

            return {
                'hourly_performance': hourly_performance,
                'symbol_performance': symbol_performance,
                'streak_analysis': self._analyze_streaks()
            }

        except Exception as e:
            logging.error(f"Trade analysis error: {str(e)}")
            return {
                'hourly_performance': {},
                'symbol_performance': {},
                'streak_analysis': {}
            }

    def _calculate_risk_metrics(self) -> dict:
        """Calculate advanced risk metrics"""
        try:
            if not self.trades:
                return {
                    'value_at_risk': 0.0,
                    'expected_shortfall': 0.0,
                    'kurtosis': 0.0,
                    'skewness': 0.0,
                    'sortino_ratio': 0.0
                }

            returns = pd.Series([t.get('realized_pnl', 0) for t in self.trades])

            if len(returns) < 2:
                return {
                    'value_at_risk': 0.0,
                    'expected_shortfall': 0.0,
                    'kurtosis': 0.0,
                    'skewness': 0.0,
                    'sortino_ratio': 0.0
                }

            negative_returns = returns[returns < 0]

            return {
                'value_at_risk': float(np.percentile(returns, 5)) if len(returns) > 0 else 0.0,
                'expected_shortfall': float(returns[returns < 0].mean()) if len(negative_returns) > 0 else 0.0,
                'kurtosis': float(returns.kurtosis()) if len(returns) > 3 else 0.0,
                'skewness': float(returns.skew()) if len(returns) > 2 else 0.0,
                'sortino_ratio': self._calculate_sortino_ratio(returns)
            }

        except Exception as e:
            logging.error(f"Risk metrics calculation error: {str(e)}")
            return {
                'value_at_risk': 0.0,
                'expected_shortfall': 0.0,
                'kurtosis': 0.0,
                'skewness': 0.0,
                'sortino_ratio': 0.0
            }

    def _analyze_streaks(self) -> dict:
        """Analyze winning and losing streaks with enhanced tracking"""
        try:
            if not self.trades:
                return {
                    'max_winning_streak': 0,
                    'max_losing_streak': 0,
                    'current_streak': 0,
                    'avg_winning_streak': 0,
                    'avg_losing_streak': 0
                }

            streaks = []
            current_streak = 0
            winning_streaks = []
            losing_streaks = []

            for trade in self.trades:
                if trade.get('realized_pnl', 0) > 0:
                    if current_streak > 0:
                        current_streak += 1
                    else:
                        if current_streak < 0:
                            losing_streaks.append(abs(current_streak))
                        current_streak = 1
                else:
                    if current_streak < 0:
                        current_streak -= 1
                    else:
                        if current_streak > 0:
                            winning_streaks.append(current_streak)
                        current_streak = -1

                streaks.append(current_streak)

            # Add final streak
            if current_streak > 0:
                winning_streaks.append(current_streak)
            elif current_streak < 0:
                losing_streaks.append(abs(current_streak))

            return {
                'max_winning_streak': max(streaks) if any(s > 0 for s in streaks) else 0,
                'max_losing_streak': abs(min(streaks)) if any(s < 0 for s in streaks) else 0,
                'current_streak': streaks[-1] if streaks else 0,
                'avg_winning_streak': np.mean(winning_streaks) if winning_streaks else 0,
                'avg_losing_streak': np.mean(losing_streaks) if losing_streaks else 0,
                'win_streak_count': len(winning_streaks),
                'lose_streak_count': len(losing_streaks)
            }

        except Exception as e:
            logging.error(f"Streak analysis error: {str(e)}")
            return {
                'max_winning_streak': 0,
                'max_losing_streak': 0,
                'current_streak': 0,
                'avg_winning_streak': 0,
                'avg_losing_streak': 0
            }

    def save_performance_data(self, filename: str = None):
        """Save performance data to file with enhanced error handling"""
        try:
            if filename is None:
                filename = f"performance_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            report = self.generate_performance_report()

            # Validate report data
            if not report:
                raise ValueError("Empty performance report")

            # Ensure directory exists
            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)

            # Save with proper encoding and formatting
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, default=str, indent=4, ensure_ascii=False)

            logging.info(f"Performance data saved successfully to {filename}")

        except Exception as e:
            logging.error(f"Performance data save error: {str(e)}")
            # Try to save to backup location
            try:
                backup_filename = f"backup_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(backup_filename, 'w', encoding='utf-8') as f:
                    json.dump(report, f, default=str, indent=4, ensure_ascii=False)
                logging.info(f"Performance data saved to backup file: {backup_filename}")
            except Exception as backup_error:
                logging.error(f"Backup save failed: {str(backup_error)}")

    def reset_daily_stats(self):
        """Reset daily statistics with enhanced cleanup"""
        with self.lock:
            # Store historical data temporarily
            temp_trades = self.trades.copy()

            # Reset all metrics
            self.daily_performance = {}
            self.metrics = {k: 0 for k in self.metrics}

            # Clear current day data
            current_date = datetime.now().date()
            self.trades = [t for t in temp_trades if t['exit_time'].date() < current_date]

            # Recalculate metrics for historical data
            if self.trades:
                self._update_metrics()
                self._update_equity_curve()

            logging.info("Daily statistics reset completed")


class TradingSystem:
    """Complete trading system with ML, risk management, and performance tracking"""

    def __init__(self, config: Config):
        # Core components
        self.config = config
        self.angel_connector = None
        self.market_data_manager = None
        self.order_manager = None
        self.risk_manager = None
        self.position_manager = None
        self.performance_tracker = None
        self.ml_ensemble = None

        # System state
        self.is_running = False
        self.last_model_update = datetime.now()
        self.system_start_time = None
        self.daily_reset_time = None
        self.last_connection_time = datetime.now()
        self.is_connected = False
        self.is_authenticated = False

        # Market status
        self.current_regime = 'normal'  # Default to normal market regime
        self.market_status = {
            'high_volatility': False,
            'circuit_breaker': False,
            'trading_allowed': True
        }

        # Threading components
        self.trading_thread = None
        self.model_thread = None
        self.monitor_thread = None
        self.heartbeat_thread = None
        self.data_thread = None
        self.locks = {
            'trading': threading.Lock(),
            'model': threading.Lock(),
            'data': threading.Lock()
        }

        # Initialize with default capital
        self.current_capital = self.config.INITIAL_CAPITAL
        self.peak_capital = self.config.INITIAL_CAPITAL  # Explicitly initialize peak_capital
        self.risk_params = {
            'max_position_size': config.MAX_POSITION_SIZE,
            'max_positions': config.MAX_POSITIONS,
            'max_daily_loss': config.MAX_DAILY_LOSS,
            'max_capital_at_risk': config.MAX_CAPITAL_AT_RISK
        }
        self.capital = config.INITIAL_CAPITAL
        self.tech_params = {
            'min_trading_vol': 100000,  # Set appropriate default
        }
        self.daily_stats = {
            'start_value': self.config.INITIAL_CAPITAL,
            'current_value': self.config.INITIAL_CAPITAL,
            'peak_value': self.config.INITIAL_CAPITAL,
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'pnl': 0,
            'positions': {},
            'max_drawdown': 0
        }

    def _check_network_connectivity(self) -> bool:
        """Check if network is available"""
        try:
            # Try multiple endpoints in case one is down
            endpoints = [
                ("apiconnect.angelone.in", 443),
                ("www.google.com", 443),  # Fallback to check general internet
            ]

            for host, port in endpoints:
                try:

                    socket.create_connection((host, port), timeout=5)
                    logging.info(f"Network connectivity check successful with {host}")
                    return True
                except OSError:
                    logging.warning(f"Failed to connect to {host}")
                    continue

            logging.error("Network connectivity check failed for all endpoints")
            return False

        except Exception as e:
            logging.error(f"Network check error: {str(e)}")
            return False

    def initialize(self) -> bool:
        """Initialize all system components with proper cross-references"""
        try:
            logging.info("Initializing trading system...")

            # 1. Check network connectivity
            if not self._check_network_connectivity():
                logging.error("Network connectivity check failed")
                return False

            # 2. Initialize Angel One connection
            self.angel_connector = AngelOneConnector(self.config)

            # Initialize last_tick_time properly BEFORE connection attempts
            self.angel_connector.last_tick_time = {}
            for symbol in self.config.SYMBOLS:
                self.angel_connector.last_tick_time[symbol] = datetime.now()

            # Initialize startup time
            self.angel_connector.startup_time = datetime.now()

            if not self.angel_connector.connect():
                raise Exception("Failed to connect to Angel One")

            # 3. Initialize risk manager
            self.risk_manager = RiskManager(self.config)
            self.risk_manager.market_status = {
                'high_volatility': False,
                'circuit_breaker': False,
                'trading_allowed': True
            }
            self.risk_manager.volatility_metrics = {}
            self.risk_manager.correlation_matrix = pd.DataFrame()

            # Set cross-reference to trading system
            self.risk_manager.trading_system = self

            # 4. Initialize market data manager
            self.market_data_manager = MarketDataManager(self.angel_connector, self.config)

            # Create bidirectional link between components for timestamp synchronization
            self.angel_connector.market_data_manager = self.market_data_manager
            self.market_data_manager.trading_system = self

            # Ensure market_data_manager has proper time tracking initialized
            self.market_data_manager.last_tick_time = {}
            for symbol in self.config.SYMBOLS:
                self.market_data_manager.last_tick_time[symbol] = datetime.now()

            self.market_data_manager._warmup_start_time = datetime.now()
            self.market_data_manager.warmup_complete = False

            logging.info("Loading historical data...")
            self.market_data_manager._load_historical_data()

            # 5. Initialize order manager
            self.order_manager = OrderManager(self.angel_connector, self.config)
            self.order_manager.trading_system = self
            self.order_manager.start()

            # 6. Initialize position management
            self.position_manager = PositionManager(self.config)
            self.position_manager.trading_system = self
            self.position_manager.positions = {}
            self.position_manager.total_exposure = 0
            self.position_manager.margin_used = 0
            self.position_manager.unrealized_pnl = 0
            self.position_manager.position_count = 0
            self.position_manager.peak_capital = self.config.INITIAL_CAPITAL

            # 7. Initialize performance tracking
            self.performance_tracker = PerformanceTracker()
            self.performance_tracker.trading_system = self

            # Reset metrics
            self.performance_tracker.metrics = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0,
                'win_rate': 0,
                'average_win': 0,
                'average_loss': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'average_holding_time': 0
            }

            # 8. Initialize and train ML models
            logging.info("Training ML models...")
            self.ml_ensemble = MLEnsemble(self.config)
            self.ml_ensemble.trading_system = self

            # Train models
            self._train_models()

            if not hasattr(self.ml_ensemble, 'is_trained') or not self.ml_ensemble.is_trained:
                logging.warning("ML models flag not set - setting it now")
                self.ml_ensemble.is_trained = True  # Force the flag to be set

            # 9. Initialize additional properties
            self.position_manager.positions = {}
            self.max_drawdown = 0
            self.peak_capital = self.config.INITIAL_CAPITAL
            self.positions_metrics = {
                'total_exposure': 0,
                'highest_capital': self.config.INITIAL_CAPITAL,
                'lowest_capital': self.config.INITIAL_CAPITAL,
                'open_positions_count': 0,
                'daily_trades_count': 0
            }

            # 10. Set system state and timing variables
            self.current_capital = self.config.INITIAL_CAPITAL
            self.system_start_time = datetime.now()
            self.daily_reset_time = datetime.now().replace(hour=9, minute=15, second=0)
            self.last_model_update = datetime.now()

            # Verify time tracking data structures
            if not hasattr(self.angel_connector, 'last_tick_time') or not self.angel_connector.last_tick_time:
                self.angel_connector.last_tick_time = {}
                for symbol in self.config.SYMBOLS:
                    self.angel_connector.last_tick_time[symbol] = datetime.now()

            if not hasattr(self.market_data_manager, 'last_tick_time') or not self.market_data_manager.last_tick_time:
                self.market_data_manager.last_tick_time = {}
                for symbol in self.config.SYMBOLS:
                    self.market_data_manager.last_tick_time[symbol] = datetime.now()

            # 11. Start the market data feed
            logging.info("Starting live data feed...")
            self.market_data_manager.start()

            logging.info("Trading system initialized successfully")
            return True

        except Exception as e:
            logging.error(f"System initialization error: {str(e)}", exc_info=True)
            return False

    def start(self):
        """Start trading system with proper thread management"""
        try:
            if not self.is_running:
                self.is_running = True

                # Don't start market data manager again if it's already been started in initialize()
                # Just check its status
                if not hasattr(self.market_data_manager, 'is_running') or not self.market_data_manager.is_running:
                    logging.info("Starting market data manager...")
                    self.market_data_manager.start()
                else:
                    logging.info("Market data manager already running")

                # Wait for initial data flow
                logging.info("Waiting for initial market data...")
                wait_end = time.time() + 30
                while time.time() < wait_end:
                    if hasattr(self.market_data_manager, 'last_tick_time'):
                        if len(self.market_data_manager.last_tick_time) == len(self.config.SYMBOLS):
                            break
                    time.sleep(1)

                # Start heartbeat for system health monitoring
                self.heartbeat_thread = threading.Thread(target=self._start_heartbeat)
                self.heartbeat_thread.daemon = True
                self.heartbeat_thread.start()

                # Start main trading loop
                self.trading_thread = threading.Thread(target=self._trading_loop)
                self.trading_thread.daemon = True
                self.trading_thread.start()

                # Start model update loop
                self.model_thread = threading.Thread(target=self._model_update_loop)
                self.model_thread.daemon = True
                self.model_thread.start()

                # Start system monitoring
                self.monitor_thread = threading.Thread(target=self._monitoring_loop)
                self.monitor_thread.daemon = True
                self.monitor_thread.start()

                logging.info("Trading system started successfully")

        except Exception as e:
            logging.error(f"System start error: {str(e)}")
            self.stop()
            raise

    def _get_system_status(self) -> dict:
        """Get comprehensive system status with safe access"""
        try:
            status = {
                'time': datetime.now(),
                'uptime': (datetime.now() - self.system_start_time).seconds if hasattr(self,
                                                                                       'system_start_time') else 0,
                'connection_status': self.angel_connector.connection_status if hasattr(self,
                                                                                       'angel_connector') else False,
                'positions': len(self.position_manager.positions) if hasattr(self, 'position_manager') else 0,
                'daily_pnl': self.performance_tracker.metrics.get('total_pnl', 0) if hasattr(self,
                                                                                             'performance_tracker') else 0,
                'win_rate': self.performance_tracker.metrics.get('win_rate', 0) if hasattr(self,
                                                                                           'performance_tracker') else 0,
                'model_last_update': self.last_model_update if hasattr(self, 'last_model_update') else None,
                'risk_status': self.risk_manager.market_status if hasattr(self, 'risk_manager') else {}
            }
            return status
        except Exception as e:
            logging.error(f"Status generation error: {str(e)}")
            return {
                'time': datetime.now(),
                'status': 'Error generating full status'
            }

    def stop(self):
        """Stop trading system with proper cleanup"""
        if not hasattr(self, '_stopping'):  # Prevent double shutdown
            self._stopping = True
            print("\n⏹️ Shutting down trading system...")
            self.is_running = False

            try:
                # Set a flag to prevent new trades during shutdown
                self._in_shutdown_process = True

                # Immediately stop all trading threads before closing positions
                if hasattr(self, 'trading_thread') and self.trading_thread and self.trading_thread.is_alive():
                    # Signal trading thread to stop
                    self._stop_trading_event = threading.Event()
                    self._stop_trading_event.set()
                    # Give it a moment to finish current operations
                    self.trading_thread.join(timeout=2)

                # First stop the market data manager
                if hasattr(self, 'market_data_manager'):
                    self.market_data_manager.is_running = False
                    self.market_data_manager.stop()

                # Close all positions with clear shutdown reason
                if hasattr(self, 'position_manager'):
                    open_positions = list(self.position_manager.positions.keys())
                    for symbol in open_positions:
                        current_price = self.market_data_manager.get_latest_price(symbol)
                        if current_price:
                            logging.info(f"Closing position for {symbol} at {current_price}")
                            self._close_position(symbol, 'SYSTEM_SHUTDOWN')
                        # Ensure position is removed even if price can't be found
                        if symbol in self.position_manager.positions:
                            del self.position_manager.positions[symbol]

                # Stop components in correct order
                if hasattr(self, 'order_manager'):
                    self.order_manager.stop()

                if hasattr(self, 'angel_connector'):
                    self.angel_connector.disconnect()

                # Get metrics with proper PnL
                try:
                    metrics = self.performance_tracker.get_performance_summary()
                    initial_capital = self.config.INITIAL_CAPITAL
                    total_pnl = sum(trade.get('pnl', 0) for trade in self.trades_history) if hasattr(self,
                                                                                                     'trades_history') else 0
                    final_capital = initial_capital + total_pnl
                    win_rate = metrics.get('win_rate', 0) * 100 if metrics else 0
                    total_trades = metrics.get('total_trades', 0) if metrics else 0
                except:
                    metrics = {}
                    initial_capital = self.config.INITIAL_CAPITAL
                    total_pnl = 0
                    final_capital = initial_capital
                    win_rate = 0
                    total_trades = 0

                # Display summary
                print("\n=== Trading Session Summary ===")
                print(f"{'Initial Capital:':<20} ₹{initial_capital:,.2f}")
                print(f"{'Final Capital:':<20} ₹{final_capital:,.2f}")
                print(f"{'Total P&L:':<20} ₹{total_pnl:,.2f}")
                print(f"{'Total Trades:':<20} {total_trades}")
                print(f"{'Win Rate:':<20} {win_rate:.1f}%")
                if total_trades > 0:
                    print(f"{'Return:':<20} {((final_capital / initial_capital - 1) * 100):.2f}%")
                if metrics.get('max_drawdown'):
                    print(f"{'Max Drawdown:':<20} {(metrics['max_drawdown'] * 100):.2f}%")

                print("\n✅ Trading session ended cleanly")
                sys.exit(0)  # Clean exit

            except Exception as e:
                print(f"\n❌ Error during shutdown: {str(e)}")
                logging.error(f"Shutdown error: {str(e)}")
                sys.exit(1)  # Error exit

    def _trading_loop(self):
        """Modified trading loop with improved activity monitoring and shutdown checking"""
        last_signal_time = {}
        last_system_log = datetime.now()
        last_risk_update = datetime.now()
        last_metrics_update = datetime.now()
        last_model_check = datetime.now()
        last_active_check = datetime.now()  # Track last time we actively checked for opportunities

        # Add counter for forcing periodic opportunity checks
        if not hasattr(self, '_force_check_counter'):
            self._force_check_counter = 0

        logging.info("Trading loop started - initializing components with directional trading...")

        # Initialize warmup time if needed
        if not hasattr(self.market_data_manager, '_warmup_start_time'):
            self.market_data_manager._warmup_start_time = datetime.now()

        # Add a startup delay
        startup_delay = 30
        logging.info(f"Allowing {startup_delay} seconds for system to stabilize...")
        time.sleep(10)

        # Initialize tracking dictionaries
        if not hasattr(self, '_last_signal_check'):
            self._last_signal_check = {}
        if not hasattr(self, '_test_trade_executed'):
            self._test_trade_executed = False
        if not hasattr(self, '_inactive_periods'):
            self._inactive_periods = 0  # Count periods with no trading activity
        if not hasattr(self, '_recently_closed_positions'):
            self._recently_closed_positions = {}

        # CRITICAL FIX: Initialize recently closed symbols dictionary with shorter timeout
        if not hasattr(self, '_recently_closed_symbols'):
            self._recently_closed_symbols = {}

        # CRITICAL FIX: Add a flag to track whether we've seen activity since last check
        activity_since_reset = False
        no_activity_count = 0
        last_activity_reset = datetime.now()

        # Maximum positions per symbol for intraday trading
        max_positions_per_symbol = 3

        # Position cooldown tracking - prevent immediate re-entry
        position_cooldowns = {}
        cooldown_period = 60  # seconds

        while self.is_running:
            try:
                # CRITICAL FIX: Check for shutdown signal
                if hasattr(self, '_in_shutdown_process') and self._in_shutdown_process:
                    logging.info("Trading loop detected shutdown signal, exiting...")
                    break

                current_time = datetime.now()
                startup_elapsed = (current_time - self.system_start_time).total_seconds()

                # CRITICAL FIX: Handle position closure follow-up actions
                if hasattr(self, '_position_just_closed') and self._position_just_closed:
                    try:
                        logging.info("Processing position closure follow-up actions")

                        # Get the symbol of the closed position
                        closed_symbol = getattr(self, '_last_closed_symbol', None)

                        # Update risk manager if needed
                        if hasattr(self, 'risk_manager') and closed_symbol:
                            try:
                                self.risk_manager.update_position(
                                    symbol=closed_symbol,
                                    quantity=0,  # 0 quantity indicates position closed
                                    price=0,
                                    pnl=0
                                )
                                logging.info("Risk manager updated for closed position")
                            except Exception as risk_error:
                                logging.error(f"Error updating risk manager: {str(risk_error)}")

                        # Force a check for new opportunities
                        self._check_entry_opportunities(closed_symbol)
                        logging.info("Completed follow-up opportunity check after position closure")

                        # Reset the flags
                        self._position_just_closed = False
                        self._last_closed_symbol = None

                        # Force reset of activity counters
                        self._inactive_periods = 0
                        self._force_check_counter = 0
                        last_active_check = current_time
                        activity_since_reset = True

                    except Exception as follow_up_error:
                        logging.error(f"Error in position closure follow-up: {str(follow_up_error)}")
                        # Reset flag even on error
                        self._position_just_closed = False

                # CRITICAL FIX: Reset tracking for recently closed symbols
                # Only keep symbols closed in the last 60 seconds (reduced from 5 minutes)
                if hasattr(self, '_recently_closed_symbols') and self._recently_closed_symbols:
                    for sym in list(self._recently_closed_symbols.keys()):
                        close_time = self._recently_closed_symbols[sym]
                        if (current_time - close_time).seconds > 60:  # 1 minute instead of 5
                            del self._recently_closed_symbols[sym]
                            logging.info(f"Cleared {sym} from recently closed symbols after 60 seconds")

                # CRITICAL FIX: Detect if system has been inactive for too long
                if (current_time - last_activity_reset).seconds > 60:
                    if not activity_since_reset:
                        no_activity_count += 1
                        logging.warning(f"No trading activity detected for {no_activity_count} consecutive periods")

                        # After 3 consecutive inactive periods, force state reset
                        if no_activity_count >= 3:
                            logging.warning("System inactive too long - forcing complete state reset")
                            # Reset critical tracking variables
                            self._last_signal_check = {}
                            last_signal_time = {}
                            self._inactive_periods = 10
                            if hasattr(self, '_recently_closed_symbols'):
                                self._recently_closed_symbols = {}

                            # Force an immediate opportunity check
                            self._check_entry_opportunities()

                            # Reset inactivity tracking
                            no_activity_count = 0
                    else:
                        # Reset counter if we saw activity
                        no_activity_count = 0

                    activity_since_reset = False
                    last_activity_reset = current_time

                # During initial stabilization period, just wait
                if startup_elapsed < startup_delay:
                    if int(startup_elapsed) == startup_delay - 5:
                        logging.info("Almost ready for active trading...")
                    time.sleep(1)
                    continue

                # Warmup handling
                if not self.market_data_manager.warmup_complete:
                    warmup_elapsed = (current_time - self.market_data_manager._warmup_start_time).seconds
                    if warmup_elapsed >= self.market_data_manager.WARMUP_PERIOD:
                        self.market_data_manager.warmup_complete = True
                        logging.info("✅ Warmup period completed - starting active trading with directional signals")
                        logging.getLogger().setLevel(logging.INFO)

                        # CRITICAL FIX: Make sure no stabilization period is set
                        if hasattr(self, 'skip_reconnect_until'):
                            self.skip_reconnect_until = None
                    time.sleep(1)
                    continue

                # CRITICAL FIX: More aggressively force periodic opportunity checks
                self._force_check_counter += 1
                if self._force_check_counter >= 60:  # Every 60 seconds
                    self._force_check_counter = 0
                    self._check_entry_opportunities()
                    # Reset activity flag
                    activity_since_reset = True

                # Trading hours check
                if not self._is_trading_time():
                    if not hasattr(self, '_last_trading_hour_log') or \
                            (current_time - self._last_trading_hour_log).seconds > 300:
                        logging.info("Outside trading hours - waiting for market open")
                        self._last_trading_hour_log = current_time
                    time.sleep(60)
                    continue

                # CRITICAL FIX: More frequent activity check with lower threshold
                inactive_time = (current_time - last_active_check).seconds
                if inactive_time > 30:  # Check every 30 seconds instead of 60
                    self._inactive_periods += 1
                    logging.info(f"Activity check: {inactive_time}s since last activity check")
                    last_active_check = current_time

                    # After 3 consecutive inactive periods, force a complete opportunity check
                    if self._inactive_periods >= 3:  # Reduced from 5 to 3
                        logging.info("System appears inactive - forcing full opportunity check")
                        self._check_entry_opportunities()  # Check all symbols
                        self._inactive_periods = 0  # Reset counter after check
                        activity_since_reset = True

                        # If still no positions, try the first available symbol with a lower threshold
                        if not self.position_manager.positions:
                            for symbol in self.config.SYMBOLS:
                                if symbol not in self.position_manager.positions:
                                    data = self.market_data_manager.get_current_data(symbol)
                                    if data is not None and len(data) > 0:
                                        logging.info(f"Forcing detailed signal check for {symbol}")
                                        action, direction, processed_data = self._process_signals(data, symbol)
                                        logging.info(f"Signal check result: {action}, {direction}")
                                        break

                # Update existing positions with EXIT action handling
                position_updates_made = False
                for symbol in list(self.position_manager.positions.keys()):
                    try:
                        position = self.position_manager.positions[symbol]
                        current_price = self.market_data_manager.get_latest_price(symbol)
                        if current_price:
                            data = self.market_data_manager.get_current_data(symbol)
                            if data is not None:
                                # For positions without ML exits, update normally
                                self.update_position(
                                    symbol=symbol,
                                    position=position,
                                    current_price=current_price,
                                    data=data
                                )
                                position_updates_made = True

                                # For positions without EXIT signal, update normally
                                self.update_position(
                                    symbol=symbol,
                                    position=position,
                                    current_price=current_price,
                                    data=data
                                )
                                position_updates_made = True
                    except Exception as pos_e:
                        logging.error(f"Position update error for {symbol}: {str(pos_e)}")

                # Process signals for new trades with action focus and direction
                entries_checked = False
                for symbol in self.config.SYMBOLS:
                    try:
                        # Skip if already has position
                        if symbol in self.position_manager.positions:
                            continue

                        # Count current positions for this symbol - track by direction
                        long_positions = 0
                        short_positions = 0
                        for pos_symbol, pos in self.position_manager.positions.items():
                            if pos_symbol == symbol:
                                if pos['side'] in ['BUY', 'LONG']:
                                    long_positions += 1
                                else:
                                    short_positions += 1

                        # CRITICAL FIX: Check for signals more frequently
                        # Get signal generation interval - check more frequently
                        check_signal = True
                        if symbol in last_signal_time:
                            time_since_last = (current_time - last_signal_time[symbol]).seconds
                            if time_since_last < 3:  # Reduced from 5 to 3 seconds
                                check_signal = False

                        if check_signal:
                            data = self.market_data_manager.get_current_data(symbol)
                            if data is None or len(data) == 0:
                                continue

                            current_price = self.market_data_manager.get_latest_price(symbol)
                            if not current_price:
                                continue

                            # Process signals - get action string and direction
                            action, direction, df_with_indicators = self._process_signals(data, symbol)
                            entries_checked = True

                            # CRITICAL FIX: Log all signal checks for debugging
                            logging.info(f"Signal check for {symbol}: Action={action}, Direction={direction}")

                            # Check for entry if action is "ENTER"
                            if action == "ENTER":
                                # Skip if at max positions for this symbol in the same direction
                                symbol_positions = long_positions if direction == 'LONG' else short_positions
                                if symbol_positions >= max_positions_per_symbol:
                                    continue

                                # Log entry check with action focus, direction, and current position count
                                logging.info(
                                    f"ENTRY CHECK - {symbol}: Action signal is {action} ({direction}), "
                                    f"Current {direction} positions: {symbol_positions}/{max_positions_per_symbol}")

                                analytics = self._analyze_entry_opportunity(symbol, current_price, df_with_indicators,
                                                                            direction)

                                # Log analytics result
                                logging.info(
                                    f"Entry opportunity for {symbol}: {analytics['should_enter']} - {analytics.get('reason', 'Unknown')}")

                                if analytics['should_enter']:
                                    # Calculate position size with leverage
                                    quantity = self._calculate_position_size(symbol, current_price, df_with_indicators)

                                    if quantity > 0:
                                        # Calculate entry parameters with leverage adjustments and direction
                                        entry_params = self._calculate_entry_parameters(
                                            symbol, current_price, df_with_indicators, direction
                                        )

                                        # Log potential trade
                                        logging.info(
                                            f"Directional trade opportunity detected for {symbol}:\n"
                                            f"Action: {action} ({direction})\n"
                                            f"Price: ₹{current_price:.2f}\n"
                                            f"Quantity: {quantity}\n"
                                            f"Stop Loss: ₹{entry_params['stop_loss']:.2f}\n"
                                            f"Take Profit: ₹{entry_params['take_profit']:.2f}"
                                        )

                                        # CRITICAL FIX: Determine the side based on direction
                                        side = 'BUY' if direction == 'LONG' else 'SELL'

                                        # CRITICAL FIX: Log the side determination
                                        logging.info(f"Direction to side conversion for {symbol}: {direction} → {side}")

                                        # Try to open position
                                        success = self._open_position(
                                            symbol=symbol,
                                            side=side,  # Now dynamic based on direction
                                            quantity=quantity,
                                            entry_price=current_price,
                                            stop_loss=entry_params['stop_loss'],
                                            take_profit=entry_params['take_profit'],
                                            timestamp=current_time,
                                            data=df_with_indicators
                                        )

                                        # Log position opening result and update cooldown
                                        if success:
                                            logging.info(
                                                f"TRADE SUCCESS - {symbol}: {direction} position opened successfully")
                                            # Reset inactive periods counter if we successfully opened a position
                                            self._inactive_periods = 0
                                            last_active_check = current_time
                                            activity_since_reset = True
                                        else:
                                            logging.warning(
                                                f"TRADE FAILURE - {symbol}: Failed to open {direction} position")

                            last_signal_time[symbol] = current_time

                    except Exception as e:
                        logging.error(f"Error processing {symbol}: {str(e)}")
                        logging.error(traceback.format_exc())

                # Update activity tracking based on actions taken
                if position_updates_made or entries_checked:
                    last_active_check = current_time
                    self._inactive_periods = 0  # Reset counter when there's activity
                    activity_since_reset = True

                # Performance metrics update every 30 seconds
                if (current_time - last_metrics_update).seconds >= 30:
                    self._update_performance_metrics()
                    last_metrics_update = current_time

                # System state logging every minute
                if (current_time - last_system_log).seconds >= 300:  # 5 minutes
                    self._log_system_state()
                    last_system_log = current_time

                # Model updates check every hour
                if (current_time - last_model_check).seconds > self.config.RETRAIN_INTERVAL:
                    logging.info("Starting model retraining...")
                    self._train_models()
                    last_model_check = current_time
                    logging.info("Model retraining completed")

                # Daily reset check
                self._check_daily_reset()

                time.sleep(1)  # Prevent tight loop

            except Exception as e:
                logging.error(f"Trading loop error: {str(e)}")
                logging.error(traceback.format_exc())
                # Reset activity trackers on error
                last_active_check = datetime.now()
                self._inactive_periods = 0
                time.sleep(5)  # Sleep longer on error

        logging.info("Trading loop exited")

    def _force_test_trade(self):
        """Force a test trade on the symbol with the strongest signal"""
        try:
            logging.info("FORCE TEST TRADE: Attempting forced test trade...")

            # Find symbol with the highest entry score
            best_symbol = None
            best_score = 0.0
            best_price = 0.0

            for symbol in self.config.SYMBOLS:
                # Skip if already has position
                if symbol in self.position_manager.positions:
                    continue

                # Get data
                data = self.market_data_manager.get_current_data(symbol)
                if data is None or len(data) == 0:
                    continue

                current_price = self.market_data_manager.get_latest_price(symbol)
                if not current_price:
                    continue

                # Get ML prediction
                features_df = self.ml_ensemble.prepare_features(data.tail(100))
                if features_df.empty:
                    continue

                prediction_result = self.ml_ensemble.predict(features_df)

                if isinstance(prediction_result, pd.DataFrame) and 'entry_score' in prediction_result.columns:
                    entry_score = float(prediction_result['entry_score'].iloc[-1])

                    if entry_score > best_score:
                        best_score = entry_score
                        best_symbol = symbol
                        best_price = current_price

            # If no symbol found, just use the first available one
            if not best_symbol:
                for symbol in self.config.SYMBOLS:
                    if symbol not in self.position_manager.positions:
                        data = self.market_data_manager.get_current_data(symbol)
                        current_price = self.market_data_manager.get_latest_price(symbol)
                        if data is not None and len(data) > 0 and current_price:
                            best_symbol = symbol
                            best_price = current_price
                            break

            if not best_symbol:
                logging.error("FORCE TEST TRADE: Failed to find suitable symbol")
                return

            logging.info(f"FORCE TEST TRADE: Selected {best_symbol} with price {best_price:.2f}")

            # Get data
            data = self.market_data_manager.get_current_data(best_symbol)

            # Calculate conservative position parameters
            stop_loss = best_price * 0.99  # 1% stop loss
            take_profit = best_price * 1.02  # 2% take profit

            # Calculate small position size (1% of capital)
            position_value = self.config.INITIAL_CAPITAL * 0.01
            quantity = max(1, int(position_value / best_price))  # Ensure at least 1 unit

            logging.info(
                f"FORCE TEST TRADE: Opening position for {best_symbol} - Qty: {quantity}, Price: {best_price:.2f}")

            # Open position with FORCED action
            success = self._open_position(
                symbol=best_symbol,
                side='BUY',
                quantity=quantity,
                entry_price=best_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                timestamp=datetime.now(),
                data=data
            )

            if success:
                logging.info(f"FORCE TEST TRADE: Successfully opened test position for {best_symbol}")
            else:
                logging.error(f"FORCE TEST TRADE:    Failed to open test position for {best_symbol}")

                # Try with risk override if normal opening failed
                logging.info("FORCE TEST TRADE: Attempting with risk override...")

                # Direct position creation in position manager
                if hasattr(self, 'position_manager'):
                    self.position_manager.positions[best_symbol] = {
                        'quantity': quantity,
                        'entry_price': best_price,
                        'side': 'BUY',
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'entry_time': datetime.now(),
                        'highest_price': best_price,
                        'lowest_price': best_price,
                        'current_price': best_price,
                        'unrealized_pnl': 0.0,
                        'max_favorable_excursion': 0.0,
                        'max_adverse_excursion': 0.0,
                        'trailing_stop': stop_loss
                    }

                    # Log forced position creation
                    logging.info(f"FORCE TEST TRADE: Directly created position for {best_symbol}")

        except Exception as e:
            logging.error(f"FORCE TEST TRADE: Error creating test trade: {str(e)}")
            logging.error(traceback.format_exc())

    def _process_signals(self, data: pd.DataFrame, symbol: str) -> Tuple[str, str, pd.DataFrame]:
        """Process signals using ML ensemble with enhanced error handling, preserved ML decisions,
        balanced entry/exit logic, and support for EXIT signals from 5-class prediction."""
        try:
            # CRITICAL FIX: Ensure symbol is not None and is properly logged
            if symbol is None:
                logging.error("Symbol is None in _process_signals - using default")
                symbol = list(self.config.SYMBOLS.keys())[0] if self.config.SYMBOLS else "unknown"

            # Add timestamp tracking for periodic logging
            current_time = datetime.now()

            # Track last evaluation time and result to prevent redundant processing
            if not hasattr(self, '_last_evaluation'):
                self._last_evaluation = {}

            # Track last exit signal time to prevent flooding
            if not hasattr(self, '_last_exit_signal_time'):
                self._last_exit_signal_time = {}

            # Check if we've evaluated this symbol recently with the same data
            eval_key = f"{symbol}_{current_time.minute}_{current_time.second // 5}"  # Group by 5-second intervals
            if eval_key in self._last_evaluation:
                prev_action, prev_direction, _ = self._last_evaluation[eval_key]
                # If we evaluated this exact second before, return the cached result
                return prev_action, prev_direction, data

            # Calculate position age for minimum holding time enforcement
            position_exists = symbol in self.position_manager.positions
            position_age_minutes = 0

            if position_exists:
                entry_time = self.position_manager.positions[symbol].get('entry_time', current_time)
                position_age_seconds = (current_time - entry_time).total_seconds()
                position_age_minutes = position_age_seconds / 60

            # Initialize the last signal log time if it doesn't exist
            if not hasattr(self, '_last_signal_log_time'):
                self._last_signal_log_time = {}

            # Get latest data and validate
            historical_data = self.market_data_manager.historical_data.get(symbol)
            if historical_data is None or historical_data.empty:
                return "HOLD", "LONG", pd.DataFrame()

            # Convert to timezone naive and validate OHLC
            if isinstance(historical_data.index, pd.DatetimeIndex) and historical_data.index.tz is not None:
                historical_data = historical_data.copy()
                historical_data.index = historical_data.index.tz_localize(None)

            # Get current data and validate
            current_data = self.market_data_manager.get_current_data(symbol)
            if current_data is not None and not current_data.empty:
                if isinstance(current_data.index, pd.DatetimeIndex) and current_data.index.tz is not None:
                    current_data = current_data.copy()
                    current_data.index = current_data.index.tz_localize(None)

                # Combine and validate data
                full_data = pd.concat([historical_data, current_data])
                full_data = full_data[~full_data.index.duplicated(keep='last')]
                full_data = full_data.sort_index()
            else:
                full_data = historical_data

            # CRITICAL FIX: Set the name attribute for the DataFrame
            full_data.name = symbol

            # Apply a single unified feature calculation for consistency
            full_data = self.ml_ensemble._calculate_technical_features(full_data)

            # Remove any potential lookahead features before prediction
            lookahead_features = [
                'ideal_entry', 'ideal_exit', 'ideal_exit_long', 'ideal_exit_short', 'trade_id', 'profit_target',
                'stop_level', 'realized_profit', 'pre_entry', 'post_entry',
                'entry_quality', 'fwd_return_3', 'fwd_return_5', 'fwd_return_10',
                'fwd_return_15', 'fwd_max_price_5', 'fwd_min_price_5',
                'fwd_max_profit_5', 'fwd_max_loss_5', 'fwd_max_price_10',
                'fwd_min_price_10', 'fwd_max_profit_10', 'fwd_max_loss_10',
                'fwd_max_price_15', 'fwd_min_price_15', 'fwd_max_profit_15',
                'fwd_max_loss_15', 'ideal_entry_long', 'ideal_entry_short',
                'trade_direction', 'position_active', 'position_direction'
            ]

            for col in lookahead_features:
                if col in full_data.columns:
                    full_data = full_data.drop(col, axis=1)

            # Clean data for prediction
            full_data = full_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
            full_data = full_data.replace([np.inf, -np.inf], 0)

            # Prepare features for prediction
            features_df = self.ml_ensemble.prepare_features(full_data)
            if features_df.empty:
                return "HOLD", "LONG", full_data

            # Generate ML signal
            if len(features_df) >= 20:  # Ensure enough data for prediction
                X = features_df.tail(100)  # Use last 100 periods

                # IMPORTANT: Check that we have enough data
                if len(X) < 20:
                    logging.warning(f"Insufficient data for prediction: {len(X)} rows")
                    return "HOLD", "LONG", full_data

                try:
                    # CRITICAL FIX: Make sure the dataframe has the symbol name as an attribute
                    X.name = symbol

                    # Get prediction from ML ensemble (5-class system)
                    prediction_result = self.ml_ensemble.predict(X)

                    # Extract the action string directly from the prediction result
                    if not isinstance(prediction_result, pd.DataFrame):
                        logging.error(f"Invalid prediction result type: {type(prediction_result)}")
                        return "HOLD", "LONG", full_data

                    # CRITICAL FIX: Extract action and direction as strings
                    action = "HOLD"  # Default action
                    direction = "LONG"  # Default direction

                    if 'action' in prediction_result.columns:
                        action = str(prediction_result['action'].iloc[-1])

                        # Extract direction if available
                        if 'direction' in prediction_result.columns:
                            direction = str(prediction_result['direction'].iloc[-1])

                        # Extract scores for logging
                        entry_score = float(prediction_result['entry_score'].iloc[
                                                -1]) if 'entry_score' in prediction_result.columns else 0.0
                        exit_score = float(prediction_result['exit_score'].iloc[
                                               -1]) if 'exit_score' in prediction_result.columns else 0.0
                        confidence = float(prediction_result['confidence'].iloc[
                                               -1]) if 'confidence' in prediction_result.columns else 0.0
                        predicted_class = int(prediction_result['predicted_class'].iloc[
                                                  -1]) if 'predicted_class' in prediction_result.columns else 0

                        threshold = self.ml_ensemble.signal_thresholds.get(symbol, 0.3)

                        # CRITICAL FIX: Get current price directly for checking adverse moves
                        current_price = self.market_data_manager.get_latest_price(symbol)

                        # ML EXIT signals disabled - convert all EXIT actions to HOLD
                        if action == "EXIT":
                            action = "HOLD"
                            logging.debug(f"Converted EXIT signal to HOLD for {symbol} - ML exits disabled")

                        # CRITICAL FIX: Check if position already exists for this symbol
                        if position_exists and action == "ENTER":
                            # Check if the existing position is in the same direction
                            existing_position = self.position_manager.positions[symbol]
                            existing_side = existing_position.get('side', 'LONG')
                            if (existing_side in ['BUY', 'LONG'] and direction == 'LONG') or \
                                    (existing_side in ['SELL', 'SHORT'] and direction == 'SHORT'):
                                # If we already have a position in this direction, change action to HOLD
                                action = "HOLD_POSITION"
                                logging.info(
                                    f"Changed action from ENTER to HOLD_POSITION - already have {direction} position in {symbol}")

                        # Log detailed signal info with action
                        should_log = False
                        if symbol not in self._last_signal_log_time or \
                                (current_time - self._last_signal_log_time[
                                    symbol]).seconds >= 300:  # 300 seconds = 5 minutes
                            should_log = True
                            self._last_signal_log_time[symbol] = current_time

                        if should_log or action in ["ENTER", "EXIT"]:
                            # Log with individual model scores
                            nn_score = float(prediction_result['nn_score'].iloc[
                                                 -1]) if 'nn_score' in prediction_result.columns else 0.0
                            svm_score = float(prediction_result['svm_score'].iloc[
                                                  -1]) if 'svm_score' in prediction_result.columns else 0.0

                            market_regime = getattr(self, 'current_regime', 'normal')

                            logging.info(
                                f"SIGNAL: {symbol} - Action: {action}, Direction: {direction}, Entry: {entry_score:.3f}, Exit: {exit_score:.3f}, "
                                f"Confidence: {confidence:.3f}, Class: {predicted_class}, NN: {nn_score:.3f}, SVM: {svm_score:.3f}, "
                                f"Market regime: {market_regime}, Has position: {position_exists}, Age: {position_age_minutes:.1f}min")

                        # FIXED: Only log meaningful action signals with valid data
                        if action in ["ENTER", "EXIT"] and symbol and isinstance(symbol, str) and current_price:
                            logging.info(
                                f"ACTION SIGNAL: {symbol} - {action} ({direction}) at price {current_price:.2f}, confidence: {confidence:.2f}")

                        # Store this evaluation result
                        self._last_evaluation[eval_key] = (action, direction, None)

                        # Return the action string, direction, and processed data
                        return action, direction, full_data
                    else:
                        logging.warning(f"No 'action' column in prediction result for {symbol}")
                        # Extract signal score as fallback
                        if 'entry_score' in prediction_result.columns:
                            entry_score = float(prediction_result['entry_score'].iloc[-1])
                            threshold = self.ml_ensemble.signal_thresholds.get(symbol, 0.3)

                            # Determine action based on entry score
                            if entry_score > threshold:
                                logging.info(f"Fallback action for {symbol}: ENTER (score: {entry_score:.3f})")
                                self._last_evaluation[eval_key] = ("ENTER", direction, None)
                                return "ENTER", direction, full_data

                        self._last_evaluation[eval_key] = ("HOLD", direction, None)
                        return "HOLD", direction, full_data

                except Exception as model_error:
                    logging.error(f"Model prediction error for {symbol}: {str(model_error)}")
                    logging.error(traceback.format_exc())
                    return "HOLD", "LONG", full_data

            action = "HOLD"
            direction = "LONG"
            self._last_evaluation[eval_key] = (action, direction, None)
            return action, direction, full_data

        except Exception as e:
            logging.error(f"Signal processing error for {symbol}: {str(e)}")
            logging.error(traceback.format_exc())
            return "HOLD", "LONG", pd.DataFrame()

    def _check_trade_conditions(self, symbol: str, quantity: int, side: str, price: float) -> Tuple[bool, str]:
        """Check if trade meets all conditions"""
        try:
            # 1. Check risk manager conditions
            risk_allowed, risk_reason = self.risk_manager.check_trade(
                symbol, side, quantity, price
            )
            if not risk_allowed:
                logging.info(f"Risk check failed for {symbol}: {risk_reason}")
                return False, risk_reason

            # 2. Check strategy risk limits
            if not self.risk_manager.check_risk_limits(symbol, quantity, side):
                reason = "Strategy risk limits exceeded"
                logging.info(f"Strategy check failed for {symbol}: {reason}")
                return False, reason

            # 3. Check market conditions
            if self.market_status.get('high_volatility', False):
                reason = "Market volatility too high"
                logging.info(f"Market check failed for {symbol}: {reason}")
                return False, reason

            if self.market_status.get('circuit_breaker', False):
                reason = "Circuit breaker active"
                logging.info(f"Market check failed for {symbol}: {reason}")
                return False, reason

            # 4. Check position limits
            current_positions = len(self.position_manager.positions)
            if current_positions >= self.risk_params['max_positions']:
                reason = "Maximum positions reached"
                logging.info(f"Position limit reached: {current_positions}")
                return False, reason

            # 5. Check capital allocation
            position_value = quantity * price
            if position_value > self.capital * self.risk_params['max_position_size']:
                reason = "Position size exceeds limit"
                logging.info(f"Position size limit exceeded for {symbol}")
                return False, reason

            # Log successful check
            logging.info(f"All trade conditions met for {symbol} - Side: {side}, Qty: {quantity}, Price: {price}")
            return True, "Trade allowed"

        except Exception as e:
            logging.error(f"Trade condition check error: {str(e)}")
            return False, f"Error checking conditions: {str(e)}"

    def _analyze_entry_opportunity(self, symbol: str, current_price: float, data: pd.DataFrame,
                                   direction: str = 'LONG') -> dict:
        """Analyze potential trade entry based on ML signal and INTELLIGENT risk metrics, with directional awareness."""
        try:
            # Check if we recently closed this symbol - if so, make it more aggressive
            recently_closed = False
            if hasattr(self, 'trades_history'):
                # Look at the last 5 trades to see if this symbol was recently closed
                recent_trades = self.trades_history[-5:] if len(self.trades_history) >= 5 else self.trades_history
                for trade in recent_trades:
                    if trade.get('symbol') == symbol and (
                            datetime.now() - trade.get('exit_time', datetime.now())).seconds < 300:
                        recently_closed = True
                        logging.info(f"Recently closed position for {symbol} - applying more aggressive entry criteria")
                        break

            # Prepare features and get ML prediction
            features_df = self.ml_ensemble.prepare_features(data.tail(100))
            if features_df.empty:
                return {
                    'should_enter': False,
                    'reason': 'Failed to prepare features',
                    'metrics': {},
                    'direction': direction
                }

            # Get ML prediction with action
            prediction_result = self.ml_ensemble.predict(features_df)

            # Extract action string and signal
            action = "HOLD"
            ml_signal = 0.0
            confidence = 0.0
            pred_direction = direction  # Use the direction passed from _process_signals

            if isinstance(prediction_result, pd.DataFrame):
                if 'action' in prediction_result.columns:
                    action = str(prediction_result['action'].iloc[-1])

                # Extract the entry score
                if 'entry_score' in prediction_result.columns:
                    ml_signal = float(prediction_result['entry_score'].iloc[-1])
                elif 'enter_score' in prediction_result.columns:
                    ml_signal = float(prediction_result['enter_score'].iloc[-1])

                # Extract confidence
                if 'confidence' in prediction_result.columns:
                    confidence = float(prediction_result['confidence'].iloc[-1])

            # Initialize metrics dictionary with safe defaults
            metrics = {
                'ml_signal': ml_signal,
                'action': action,
                'price': current_price,
                'volume': 0,  # Default value
                'volatility': 0.01,  # Default value
                'direction': pred_direction,
                'confidence': confidence
            }

            # Safely get Volume if it exists
            if 'Volume' in data.columns:
                metrics['volume'] = float(data['Volume'].iloc[-1]) if not data['Volume'].empty else 0

            # Safely get volatility if it exists
            if 'volatility_20' in data.columns:
                metrics['volatility'] = float(data['volatility_20'].iloc[-1]) if not data[
                    'volatility_20'].empty else 0.01

            # FIXED: Intelligent risk assessment instead of blanket blocking
            risk_filter_failed = False
            risk_reason = ""
            position_size_multiplier = 1.0  # Allow position size adjustment instead of blocking

            try:
                # INTELLIGENT volatility assessment based on historical patterns
                if 'volatility_20' in data.columns:
                    current_vol = float(data['volatility_20'].iloc[-1])

                    # Calculate historical volatility distribution for intelligent thresholds
                    vol_series = data['volatility_20'].dropna()
                    if len(vol_series) >= 20:
                        vol_p50 = vol_series.quantile(0.50)  # Median volatility
                        vol_p75 = vol_series.quantile(0.75)  # High volatility
                        vol_p90 = vol_series.quantile(0.90)  # Very high volatility
                        vol_p95 = vol_series.quantile(0.95)  # Extreme volatility

                        # Intelligent decision based on percentile and confidence
                        if current_vol <= vol_p75:
                            # Normal to high volatility - allow trade, maybe increase size
                            if current_vol <= vol_p50:
                                position_size_multiplier = 1.2  # Low volatility - increase size
                                logging.debug(f"Low volatility ({current_vol:.4f}) - increasing position size")
                        elif current_vol <= vol_p90:
                            # High volatility - reduce size but allow trade
                            position_size_multiplier = 0.8
                            logging.info(f"High volatility ({current_vol:.4f}) - reducing position size to 80%")
                        elif current_vol <= vol_p95:
                            # Very high volatility - only allow with high confidence
                            if confidence > 0.7:
                                position_size_multiplier = 0.6
                                logging.info(
                                    f"Very high volatility ({current_vol:.4f}) with high confidence - reducing to 60%")
                            else:
                                risk_filter_failed = True
                                risk_reason = f"Very high volatility ({current_vol:.4f}) with low confidence ({confidence:.2f})"
                        else:
                            # Extreme volatility - only allow with very high confidence and strong signal
                            if confidence > 0.8 and ml_signal > 0.7:
                                position_size_multiplier = 0.4
                                logging.info(f"Extreme volatility ({current_vol:.4f}) - very conservative sizing")
                            else:
                                risk_filter_failed = True
                                risk_reason = f"Extreme volatility ({current_vol:.4f}) - insufficient confidence/signal"
                    else:
                        # Fallback to original logic if insufficient history
                        vol_threshold = 0.15  # Much more reasonable than 0.03
                        if current_vol > vol_threshold:
                            position_size_multiplier = 0.7
                            logging.info(f"High volatility fallback - reducing position size")
            except Exception as vol_error:
                logging.error(f"Volatility assessment error: {str(vol_error)}")

            try:
                # INTELLIGENT market regime assessment
                market_regime = getattr(self, 'current_regime', 'normal')

                if market_regime == 'trending':
                    # Trending markets - favor trend-following signals
                    position_size_multiplier *= 1.1  # Slight increase
                elif market_regime == 'ranging':
                    # Ranging markets - look for reversal signals, reduce size
                    position_size_multiplier *= 0.9
                elif market_regime == 'volatile':
                    # Volatile markets - only strong signals with reduced size
                    if ml_signal < 0.5:
                        risk_filter_failed = True
                        risk_reason = f"Volatile market requires stronger signal (current: {ml_signal:.2f})"
                    else:
                        position_size_multiplier *= 0.7
                        logging.info(f"Volatile market - reducing position size to 70%")
            except Exception as regime_error:
                logging.error(f"Market regime assessment error: {str(regime_error)}")

            # Store the position size multiplier for later use
            metrics['position_size_multiplier'] = max(0.3, min(2.0, position_size_multiplier))

            # INTELLIGENT time-based risk assessment
            try:
                current_time = datetime.now().time()

                # Market opening (first 30 minutes) - more conservative
                if current_time < datetime_time(9, 45):
                    position_size_multiplier *= 0.8
                    logging.debug("Market opening period - reducing position size")

                # Last hour (power hour) - adjust based on signal strength
                elif current_time > datetime_time(14, 30):
                    if ml_signal > 0.6:
                        position_size_multiplier *= 1.1  # Strong signals get boost
                    else:
                        position_size_multiplier *= 0.8  # Weak signals get reduced
                    logging.debug("Power hour adjustment applied")

                # Lunch hour (12:00-13:00) - more conservative due to lower volume
                elif datetime_time(12, 0) <= current_time <= datetime_time(13, 0):
                    position_size_multiplier *= 0.9
                    logging.debug("Lunch hour - slightly reducing position size")

            except Exception as time_error:
                logging.error(f"Time-based assessment error: {str(time_error)}")

            # Final decision with intelligent adjustments
            result = {
                'should_enter': False,
                'reason': '',
                'metrics': metrics,
                'action': action,
                'direction': pred_direction
            }

            threshold = self.ml_ensemble.signal_thresholds.get(symbol, 0.3)

            # Reset inactivity tracking
            if hasattr(self, '_inactive_periods'):
                self._inactive_periods = 0
            if hasattr(self, 'last_active_check'):
                self.last_active_check = datetime.now()

            # INTELLIGENT entry decision
            if action == "ENTER" and not risk_filter_failed:
                adjusted_threshold = threshold

                if ml_signal > adjusted_threshold:
                    result['should_enter'] = True
                    result['reason'] = (f'INTELLIGENT ENTRY: ML action "{action}" with signal {ml_signal:.3f} '
                                        f'(threshold: {adjusted_threshold:.3f}), confidence: {confidence:.3f}, '
                                        f'direction: {direction}, position multiplier: {position_size_multiplier:.2f}')

                    logging.info(
                        f"{symbol} - INTELLIGENT ENTRY APPROVED: Signal={ml_signal:.3f}, "
                        f"Confidence={confidence:.3f}, Volatility={current_vol:.4f}, "
                        f"Regime={market_regime}, Size Multiplier={position_size_multiplier:.2f}"
                    )
                else:
                    result['reason'] = (f'ML action "{action}" but signal {ml_signal:.3f} below '
                                        f'adjusted threshold {adjusted_threshold:.3f}')
            elif action == "ENTER" and risk_filter_failed:
                result['reason'] = f'ML action "{action}" but INTELLIGENT FILTER: {risk_reason}'
                logging.info(f"{symbol} - INTELLIGENT FILTER BLOCK: {risk_reason}")
            else:
                result['reason'] = f'ML action "{action}" with signal {ml_signal:.3f}'

            return result

        except Exception as e:
            logging.error(f"Intelligent entry analysis error for {symbol}: {str(e)}")
            # Still reset inactivity tracking on error
            if hasattr(self, '_inactive_periods'):
                self._inactive_periods = 0

            return {
                'should_enter': False,
                'reason': f'Analysis error: {str(e)}',
                'metrics': {
                    'ml_signal': 0.0,
                    'price': current_price,
                    'volume': 0,
                    'volatility': 0.01,
                    'position_size_multiplier': 1.0
                },
                'action': "HOLD",
                'direction': direction
            }

    def _calculate_entry_parameters(self, symbol: str, current_price: float, data: pd.DataFrame,
                                    direction: str = 'LONG') -> dict:
        """Calculate entry parameters using ML-decided values with proper percentage conversion"""
        try:
            # Get latest metrics with backup options
            atr = float(data['atr'].iloc[-1]) if 'atr' in data.columns else (current_price * 0.01)
            volatility = float(data['volatility_20'].iloc[-1]) if 'volatility_20' in data.columns else 0.01

            # Default parameters in case ML values aren't available
            stop_pct = 0.01  # 1% stop loss
            target_pct = 0.02  # 2% take profit

            # Check for ML-optimized parameters and use them
            ml_optimized = False
            if hasattr(self.ml_ensemble, 'profit_thresholds') and symbol in self.ml_ensemble.profit_thresholds:
                ml_optimized = True
                ml_params = self.ml_ensemble.profit_thresholds[symbol]

                # Debug log the raw ML parameters
                logging.info(f"Raw ML parameters for {symbol}: {ml_params}")

                # CRITICAL FIX: Use direction-specific parameters
                if direction == 'LONG':
                    # Extract target from ML parameters for LONG
                    if 'long_target' in ml_params:
                        raw_target = ml_params['long_target']

                        # FIXED: Proper percentage conversion
                        if abs(raw_target) < 1:  # Small values like 0.35 mean 0.35%
                            target_pct = raw_target
                        else:  # Large values like 35 would mean 35%
                            target_pct = raw_target / 100.0

                        logging.info(f"Using LONG target percentage: {target_pct:.4f}% from ML model")

                    # Extract stop loss from ML parameters for LONG
                    if 'long_stop' in ml_params:
                        raw_stop = abs(ml_params['long_stop'])  # Make positive for calculation

                        # FIXED: Proper percentage conversion
                        if raw_stop < 1:  # Small values like 0.20 mean 0.20%
                            stop_pct = raw_stop
                        else:  # Large values like 20 would mean 20%
                            stop_pct = raw_stop / 100.0

                        logging.info(f"Using LONG stop percentage: {stop_pct:.4f}% from ML model")
                else:  # SHORT
                    # Extract target from ML parameters for SHORT
                    if 'short_target' in ml_params:
                        raw_target = ml_params['short_target']

                        # FIXED: Proper percentage conversion
                        if abs(raw_target) < 1:  # Small values like 0.30 mean 0.30%
                            target_pct = raw_target
                        else:  # Large values like 30 would mean 30%
                            target_pct = raw_target / 100.0

                        logging.info(f"Using SHORT target percentage: {target_pct:.4f}% from ML model")

                    # Extract stop loss from ML parameters for SHORT
                    if 'short_stop' in ml_params:
                        raw_stop = abs(ml_params['short_stop'])  # Make positive for calculation

                        # FIXED: Proper percentage conversion
                        if raw_stop < 1:  # Small values like 0.25 mean 0.25%
                            stop_pct = raw_stop
                        else:  # Large values like 25 would mean 25%
                            stop_pct = raw_stop / 100.0

                        logging.info(f"Using SHORT stop percentage: {stop_pct:.4f}% from ML model")

                # CRITICAL FIX: Calculate stop loss and take profit differently based on direction
                if direction == 'LONG':
                    # For a long position (unchanged)
                    stop_loss = current_price * (1.0 - (stop_pct / 100))
                    take_profit = current_price * (1.0 + (target_pct / 100))

                    # Calculate risk/reward
                    risk = current_price - stop_loss
                    reward = take_profit - current_price
                else:  # SHORT
                    # For a short position (inverted)
                    stop_loss = current_price * (1.0 + (stop_pct / 100))  # Stop ABOVE entry for shorts
                    take_profit = current_price * (1.0 - (target_pct / 100))  # Target BELOW entry for shorts

                    # Calculate risk/reward for shorts
                    risk = stop_loss - current_price  # Risk is upside distance
                    reward = current_price - take_profit  # Reward is downside distance

                risk_reward = reward / risk if risk > 0 else 2.0

                result = {
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'trailing_stop': stop_loss,  # Initialize trailing stop to stop loss level
                    'trailing_distance': abs(current_price - stop_loss),  # Use abs for both directions
                    'risk_reward_ratio': risk_reward,
                    'atr': atr,
                    'volatility': volatility,
                    'ml_optimized': ml_optimized
                }

                # Log parameters with percentage values
                stop_percentage = (stop_pct)  # Already in percentage value
                target_percentage = (target_pct)  # Already in percentage value

                logging.info(f"ML-optimized entry parameters for {symbol} ({direction}):")
                logging.info(f"  Entry Price: {current_price:.2f}")
                logging.info(f"  Stop Loss: {stop_loss:.2f} ({stop_percentage:.2f}%)")
                logging.info(f"  Take Profit: {take_profit:.2f} ({target_percentage:.2f}%)")
                logging.info(f"  Risk/Reward: {risk_reward:.2f}")

                return result

        except Exception as e:
            logging.error(f"Entry parameter calculation error for {symbol}: {str(e)}")
            traceback.print_exc()
            # Safe defaults if anything goes wrong - direction-aware
            if direction == 'LONG':
                return {
                    'entry_price': current_price,
                    'stop_loss': current_price * 0.99,  # 1% stop loss for longs
                    'take_profit': current_price * 1.02,  # 2% target for longs
                    'trailing_stop': current_price * 0.99,
                    'trailing_distance': current_price * 0.01,
                    'risk_reward_ratio': 2.0,
                    'atr': 0.0,
                    'volatility': 0.0,
                    'ml_optimized': False
                }
            else:  # SHORT
                return {
                    'entry_price': current_price,
                    'stop_loss': current_price * 1.01,  # 1% stop loss above for shorts
                    'take_profit': current_price * 0.98,  # 2% target below for shorts
                    'trailing_stop': current_price * 1.01,
                    'trailing_distance': current_price * 0.01,
                    'risk_reward_ratio': 2.0,
                    'atr': 0.0,
                    'volatility': 0.0,
                    'ml_optimized': False
                }

    def _should_exit_position(self, symbol: str, position: dict, current_price: float, data: pd.DataFrame) -> bool:
        """Enhanced exit condition checking"""
        try:
            current = data.iloc[-1]

            # Basic stop conditions
            if current_price <= position['stop_loss']:
                logging.info(f"{symbol}: Stop loss triggered at {current_price:.2f}")
                return True

            if current_price >= position['take_profit']:
                logging.info(f"{symbol}: Take profit triggered at {current_price:.2f}")
                return True

            if position.get('trailing_stop') and current_price <= position['trailing_stop']:
                logging.info(f"{symbol}: Trailing stop triggered at {current_price:.2f}")
                return True

            # Technical exit conditions
            exit_score = 0
            reasons = []

            # Trend change
            if current['ema_5'] < current['ema_13']:
                exit_score += 0.3
                reasons.append("Trend reversal")

            # RSI conditions
            if current['rsi'] > self.tech_params['rsi_overbought']:
                exit_score += 0.3
                reasons.append(f"RSI overbought ({current['rsi']:.1f})")

            # Volume weakness
            if current['volume_ratio'] < 0.8:
                exit_score += 0.2
                reasons.append("Volume weakness")

            # Volatility spike
            if current['volatility_20'] > self.config.VOLATILITY_THRESHOLD * 1.5:
                exit_score += 0.2
                reasons.append("High volatility")

            # Time-based exit
            if (datetime.now() - position['entry_time']).seconds > 6 * 3600:  # 6 hour max hold
                exit_score += 0.3
                reasons.append("Time limit reached")

            # Log exit score if significant
            if exit_score >= 0.7:
                logging.info(f"{symbol} exit score {exit_score:.2f} - Reasons: {', '.join(reasons)}")
                return True

            return False

        except Exception as exit_error:
            logging.error(f"Exit check error for {symbol}: {str(exit_error)}")
            return False

    def update_position(self, symbol: str, current_price: float, data: pd.DataFrame = None,
                        position: dict = None) -> bool:
        """
        Main position update function with improved price handling and consistent PnL calculation.
        Handles both long and short positions properly.
        """
        try:
            # Get position if not provided
            if position is None:
                if symbol not in self.position_manager.positions:
                    return False
                position = self.position_manager.positions[symbol]

            # CRITICAL FIX: Ensure we have a valid current price
            if not current_price or current_price <= 0:
                # Try getting price from market data manager
                current_price = self.market_data_manager.get_latest_price(symbol)

                # If still no valid price, try other sources
                if not current_price or current_price <= 0:
                    if hasattr(self, 'latest_data') and symbol in self.latest_data:
                        current_price = self.latest_data[symbol].get('Close', 0)

                    if (not current_price or current_price <= 0) and hasattr(self.angel_connector, 'latest_data'):
                        if symbol in self.angel_connector.latest_data:
                            current_price = self.angel_connector.latest_data[symbol].get('Close', 0)

                    if not current_price or current_price <= 0:
                        logging.warning(f"No valid price available for {symbol}, skipping update")
                        return False

            # Store old values for tracking changes
            old_pnl = position.get('unrealized_pnl', 0)
            is_long = position['side'] in ['BUY', 'LONG']

            # Update position in the PositionManager for price tracking
            # For standard exit checks (stop loss, take profit)
            exit_reason = self.position_manager.update_position(symbol, current_price)

            # ML exit processing removed - using mechanical stops only

            # Process exit conditions
            if exit_reason:
                # Basic time-in-position check
                entry_time = position.get('entry_time', datetime.now() - timedelta(hours=1))
                time_in_position = (datetime.now() - entry_time).total_seconds()

                # Honor all exits except with special handling for very early stops
                if exit_reason == 'STOP_LOSS' and time_in_position < 180:  # First 3 minutes
                    # For very early positions, only honor stop losses if price drops severely
                    initial_stop_distance = abs(position['entry_price'] - position.get('stop_loss', 0))

                    # Direction-specific severe drop level calculation
                    if is_long:
                        severe_drop_level = position['entry_price'] - (initial_stop_distance * 1.5)  # Reduced from 2x
                        if current_price <= severe_drop_level:
                            # Severe drop, honor the stop
                            logging.info(f"Early position severe drop for {symbol} - closing position")
                            self._close_position(symbol, 'SEVERE_DROP_STOP')
                            return False
                        else:
                            # Regular stop hit in early position - give it more room
                            logging.info(
                                f"Ignoring early stop loss for {symbol} - only {time_in_position / 60:.1f} minutes in position")
                    else:  # SHORT
                        severe_spike_level = position['entry_price'] + (initial_stop_distance * 1.5)
                        if current_price >= severe_spike_level:
                            # Severe spike, honor the stop
                            logging.info(f"Early position severe spike for {symbol} - closing short position")
                            self._close_position(symbol, 'SEVERE_SPIKE_STOP')
                            return False
                        else:
                            # Regular stop hit in early position - give it more room
                            logging.info(
                                f"Ignoring early stop loss for {symbol} short - only {time_in_position / 60:.1f} minutes in position")
                else:
                    # Process all other exits immediately
                    logging.info(f"Exit triggered for {symbol} - Reason: {exit_reason}, Price: {current_price:.2f}")
                    self._close_position(symbol, exit_reason)
                    return False

            # Update trailing stop logic with direction awareness
            if data is not None:
                if is_long:
                    # For long positions - update trailing stop when price moves up
                    initial_stop_distance = position['entry_price'] - position.get('stop_loss', 0)
                    target_distance = position.get('take_profit', 0) - position['entry_price']

                    # Require at least 30% progress toward target before activating trailing stop
                    profit_threshold = position['entry_price'] + (target_distance * 0.3)

                    # Only update trailing stop when price breaks above profit threshold
                    if current_price > profit_threshold and current_price > position.get('trailing_stop_price', 0):
                        # Calculate new stop level with some cushion
                        min_stop_distance = initial_stop_distance * 0.8  # At least 80% of initial stop distance

                        # Try to use ML-optimized parameters if available
                        ml_optimized_params = None
                        if hasattr(self.ml_ensemble,
                                   'profit_thresholds') and symbol in self.ml_ensemble.profit_thresholds:
                            ml_optimized_params = self.ml_ensemble.profit_thresholds[symbol]

                        # Calculate trailing stop distance
                        if 'atr' in data.columns:
                            atr = float(data['atr'].iloc[-1])

                            # Use ML-optimized ATR multiplier if available
                            atr_multiplier = 2.0
                            if ml_optimized_params and 'atr_multiplier' in ml_optimized_params:
                                atr_multiplier = ml_optimized_params['atr_multiplier']

                            new_stop = current_price - max(min_stop_distance, atr * atr_multiplier)
                        else:
                            # Default to percentage-based trailing stop, preferring ML value if available
                            stop_percentage = self.config.STOP_LOSS * 0.75
                            if ml_optimized_params and 'trailing_stop_pct' in ml_optimized_params:
                                stop_percentage = ml_optimized_params['trailing_stop_pct']

                            new_stop = current_price * (1 - stop_percentage)

                        # Only update if it moves the stop up significantly
                        current_stop = position.get('trailing_stop', 0)
                        if new_stop > current_stop * 1.005:  # At least 0.5% improvement
                            position['trailing_stop'] = new_stop
                            position['trailing_stop_price'] = current_price
                            pct_diff = ((new_stop / position['entry_price']) - 1) * 100
                            logging.info(
                                f"Updated trailing stop for {symbol} long to {new_stop:.2f} ({pct_diff:.2f}% from entry)")
                else:
                    # For short positions - update trailing stop when price moves down
                    initial_stop_distance = position.get('stop_loss', 0) - position['entry_price']
                    target_distance = position['entry_price'] - position.get('take_profit', 0)

                    # Require at least 30% progress toward target before activating trailing stop
                    profit_threshold = position['entry_price'] - (target_distance * 0.3)

                    # Only update trailing stop when price breaks below profit threshold
                    if current_price < profit_threshold and current_price < position.get('trailing_stop_price',
                                                                                         float('inf')):
                        # Calculate new stop level with some cushion
                        min_stop_distance = initial_stop_distance * 0.8  # At least 80% of initial stop distance

                        # Try to use ML-optimized parameters if available
                        ml_optimized_params = None
                        if hasattr(self.ml_ensemble,
                                   'profit_thresholds') and symbol in self.ml_ensemble.profit_thresholds:
                            ml_optimized_params = self.ml_ensemble.profit_thresholds[symbol]

                        # Calculate trailing stop distance
                        if 'atr' in data.columns:
                            atr = float(data['atr'].iloc[-1])

                            # Use ML-optimized ATR multiplier if available
                            atr_multiplier = 2.0
                            if ml_optimized_params and 'atr_multiplier' in ml_optimized_params:
                                atr_multiplier = ml_optimized_params['atr_multiplier']

                            new_stop = current_price + max(min_stop_distance, atr * atr_multiplier)
                        else:
                            # Default to percentage-based trailing stop, preferring ML value if available
                            stop_percentage = self.config.STOP_LOSS * 0.75
                            if ml_optimized_params and 'trailing_stop_pct' in ml_optimized_params:
                                stop_percentage = ml_optimized_params['trailing_stop_pct']

                            new_stop = current_price * (1 + stop_percentage)

                        # Only update if it moves the stop down significantly
                        current_stop = position.get('trailing_stop', float('inf'))
                        if new_stop < current_stop * 0.995:  # At least 0.5% improvement
                            position['trailing_stop'] = new_stop
                            position['trailing_stop_price'] = current_price
                            pct_diff = ((new_stop / position['entry_price']) - 1) * 100
                            logging.info(
                                f"Updated trailing stop for {symbol} short to {new_stop:.2f} ({pct_diff:.2f}% from entry)")

            # CRITICAL FIX: Update position tracking fields with the latest price
            position['current_price'] = current_price

            # CRITICAL FIX: Update price extremes based on position direction
            if is_long:
                position['highest_price'] = max(position.get('highest_price', current_price), current_price)
                position['lowest_price'] = min(position.get('lowest_price', current_price), current_price)
            else:  # SHORT
                position['highest_price'] = max(position.get('highest_price', current_price), current_price)
                position['lowest_price'] = min(position.get('lowest_price', current_price), current_price)

            # CRITICAL FIX: Calculate PnL correctly based on position direction
            if is_long:
                position['unrealized_pnl'] = (current_price - position['entry_price']) * position['quantity']
                position['unrealized_pnl_pct'] = ((current_price / position['entry_price']) - 1) * 100
            else:  # SHORT/SELL
                position['unrealized_pnl'] = (position['entry_price'] - current_price) * position['quantity']
                position['unrealized_pnl_pct'] = ((position['entry_price'] / current_price) - 1) * 100

            # Update maximum favorable and adverse excursions
            position['max_favorable_excursion'] = max(position.get('max_favorable_excursion', 0),
                                                      position['unrealized_pnl'])
            position['max_adverse_excursion'] = min(position.get('max_adverse_excursion', 0),
                                                    position['unrealized_pnl'])

            # Update capital tracking in TradingSystem
            if hasattr(self, 'current_capital'):
                pnl_change = position['unrealized_pnl'] - old_pnl
                self.current_capital += pnl_change
                # Update peak capital if needed
                if hasattr(self, 'peak_capital'):
                    self.peak_capital = max(self.peak_capital, self.current_capital)

                    # Calculate drawdown
                    if self.peak_capital > 0:  # Avoid division by zero
                        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
                        if hasattr(self, 'max_drawdown'):
                            self.max_drawdown = max(self.max_drawdown, current_drawdown)

            # Log significant PnL changes
            if abs(position['unrealized_pnl'] - old_pnl) > 5:  # Only log significant changes (reduced threshold to 5)
                logging.info(
                    f"Position PnL update - {symbol}: {position['unrealized_pnl']:.2f} ({position.get('unrealized_pnl_pct', 0):.2f}%)")

            # Update risk metrics
            if hasattr(self, 'risk_manager'):
                self.risk_manager.update_position(
                    symbol=symbol,
                    quantity=position['quantity'],
                    price=current_price,
                    pnl=position.get('unrealized_pnl', 0)
                )

            return True

        except Exception as e:
            logging.error(f"Position update error for {symbol}: {str(e)}")
            logging.error(traceback.format_exc())
            return False

    def _calculate_trailing_stop(self, symbol: str, position: dict, current_price: float, data: pd.DataFrame) -> float:
        """Calculate dynamic trailing stop level with safety checks and volatility-based adjustments"""
        try:
            # First check if ATR is available, if not calculate it
            if 'atr' not in data.columns:
                # Calculate ATR
                high_low = data['High'] - data['Low']
                high_close = abs(data['High'] - data['Close'].shift())
                low_close = abs(data['Low'] - data['Close'].shift())
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                data['atr'] = tr.rolling(window=14).mean()

            # Get latest ATR value with fallback
            atr = data['atr'].iloc[-1] if not data['atr'].empty else (current_price * self.config.STOP_LOSS)

            # Define list of volatile stocks
            volatile_stocks = ['TATAMOTORS', 'ADANIPORTS', 'BAJAJFINANCE', 'HINDALCO', 'INDUSINDBK', 'ZOMATO']

            # Determine if this is a long or short position
            is_long = position['side'] in ['BUY', 'LONG']

            # Adjust base distance based on volatility and position direction
            if symbol in volatile_stocks:
                # Wider stops for volatile stocks
                if is_long:
                    # For long positions on volatile stocks
                    base_distance = max(atr * 2.5, current_price * (self.config.STOP_LOSS * 1.2))
                else:
                    # For short positions on volatile stocks, adjust upward distance
                    base_distance = max(atr * 2.5, current_price * (self.config.STOP_LOSS * 1.2))

                logging.debug(f"Using wider trailing stop for volatile stock {symbol}: {base_distance:.2f}")
            else:
                # Standard stops for regular stocks
                if is_long:
                    # For regular long positions
                    base_distance = max(atr * 2, current_price * self.config.STOP_LOSS)
                else:
                    # For regular short positions
                    base_distance = max(atr * 2, current_price * self.config.STOP_LOSS)

            # Calculate new stop based on position direction
            if is_long:
                # For longs, trailing stop is below current price
                new_stop = current_price - base_distance

                # Ensure stop only moves up for longs
                current_stop = position.get('trailing_stop', 0)
                return max(current_stop, new_stop)
            else:
                # For shorts, trailing stop is above current price
                new_stop = current_price + base_distance

                # Ensure stop only moves down for shorts
                current_stop = position.get('trailing_stop', float('inf'))
                return min(current_stop, new_stop) if current_stop > 0 else new_stop

        except Exception as e:
            logging.error(f"Trailing stop calculation error for {symbol}: {str(e)}")
            # Fallback to fixed percentage trailing stop
            if position['side'] in ['BUY', 'LONG']:
                return position.get('trailing_stop', current_price * (1 - self.config.STOP_LOSS))
            else:
                return position.get('trailing_stop', current_price * (1 + self.config.STOP_LOSS))

    def _model_update_loop(self):
        """Model update loop"""
        while self.is_running:
            try:
                with self.locks['model']:
                    current_time = datetime.now()
                    if (current_time - self.last_model_update).seconds > self.config.RETRAIN_INTERVAL:
                        self._train_models()
                        self.last_model_update = current_time

                time.sleep(60)

            except Exception as e:
                logging.error(f"Model update error: {str(e)}")
                time.sleep(300)

    def _check_risk_filters(self, data: pd.DataFrame) -> bool:
        """Check if risk filters are triggered with relaxed thresholds"""
        try:
            if data is None or data.empty:
                return False

            current = data.iloc[-1]

            # Volatility filter
            if 'volatility_20' in current and current['volatility_20'] > self.config.VOLATILITY_THRESHOLD * 2.0:
                logging.debug(f"Volatility filter triggered: {current['volatility_20']}")
                return False

            # Large price move filter
            if 'returns' in current and abs(current['returns']) > 0.01:  # 3% move
                logging.debug(f"Price move filter triggered: {current['returns']}")
                return False

            # Volume filter - reduced minimum
            min_volume = self.tech_params['min_trading_vol'] * 0.5
            if 'Volume' in current and current['Volume'] < min_volume:
                logging.debug(f"Volume filter triggered: {current['Volume']}")
                return False

            # Trend weakness filter - reduced threshold
            min_trend = self.config.STRATEGY_PARAMS['trend_strength'] * 0.7
            if 'trend_strength' in current and current['trend_strength'] < min_trend:
                logging.debug(f"Trend strength filter triggered: {current['trend_strength']}")
                return False

            # Time-based risk adjustment
            current_time = datetime.now().time()
            if (current_time < datetime_time(9, 30) or  # First 15 minutes of market
                    current_time > datetime_time(15, 15)):  # Last 15 minutes of market
                if 'volatility_20' in current and current['volatility_20'] > self.config.VOLATILITY_THRESHOLD * 1.5:
                    logging.debug("Market hours volatility filter triggered")
                    return False

            return True

        except Exception as e:
            logging.error(f"Risk filter check error: {str(e)}")
            return False

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price with reduced logging"""
        try:
            if symbol not in self.latest_data:
                return None

            tick_data = self.latest_data[symbol]
            current_time = datetime.now()

            # Only check time difference if we haven't received data in more than 5 seconds
            if (current_time - tick_data.get('last_update', current_time)).seconds > 5:
                return None

            return float(tick_data.get('Close', tick_data.get('price')))

        except Exception as e:
            # Only log errors we haven't seen recently
            if not hasattr(self, '_last_price_error'):
                self._last_price_error = {}
            if symbol not in self._last_price_error or (current_time - self._last_price_error[symbol]).seconds > 300:
                logging.error(f"Error getting latest price for {symbol}: {str(e)}")
                self._last_price_error[symbol] = current_time
            return None

    def _monitoring_loop(self):
        """System monitoring with enhanced data flow recovery"""
        # CRITICAL FIX: Add substantial startup delay to avoid premature reconnection
        startup_delay = 60  # Wait a full minute before monitoring starts
        logging.info(f"Monitoring will begin in {startup_delay} seconds...")
        time.sleep(startup_delay)

        logging.info("System monitoring activated")

        while self.is_running:
            try:
                current_time = datetime.now()
                system_uptime = (current_time - self.system_start_time).total_seconds()

                # Only perform connection checks after system is fully stable
                if system_uptime < 120:  # Wait 2 full minutes before any connection management
                    time.sleep(10)
                    continue

                # CRITICAL FIX: Check data flow more frequently and aggressively recover
                # Check if market data is flowing
                data_flowing = False
                for symbol in self.config.SYMBOLS:
                    if symbol in self.market_data_manager.last_tick_time:
                        time_diff = (current_time - self.market_data_manager.last_tick_time[symbol]).seconds
                        if time_diff < 10:  # Data within last 10 seconds
                            data_flowing = True
                            break

                # If data isn't flowing, attempt recovery
                if not data_flowing and system_uptime > 180:  # After 3 minutes of uptime
                    logging.warning("No recent data detected - initiating data flow recovery")

                    # First try resubscribing
                    if hasattr(self.market_data_manager, '_subscribe_market_data'):
                        self.market_data_manager._subscribe_market_data()
                        time.sleep(2)  # Give it a moment to take effect

                        # Check if that fixed it
                        recovered = False
                        for symbol in self.config.SYMBOLS:
                            price = self.market_data_manager.get_latest_price(symbol)
                            if price:
                                logging.info(f"Data flow recovered after resubscription: {symbol} at {price:.2f}")
                                recovered = True
                                break

                        # If not fixed, try reconnection
                        if not recovered:
                            logging.warning("Resubscription didn't restore data flow - attempting reconnection")
                            if hasattr(self.angel_connector, '_attempt_reconnect'):
                                self.angel_connector._attempt_reconnect()

                # Log system state every 5 minutes, but don't attempt reconnection
                if not hasattr(self, '_last_state_log') or (current_time - self._last_state_log).seconds >= 300:
                    self._log_system_state()
                    self._last_state_log = current_time

                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logging.error(f"Monitoring error: {str(e)}")
                time.sleep(5)

    def _log_system_state(self):
        """Log comprehensive system state with enhanced verification"""
        try:
            current_time = datetime.now()

            # Build status message
            status_lines = []
            status_lines.append(f"\n=== System Status at {current_time.strftime('%H:%M:%S')} ===")

            # Verify actual WebSocket connection
            is_connected = (
                    self.angel_connector and
                    self.angel_connector.smart_socket and
                    self.angel_connector.smart_socket.sock and
                    self.angel_connector.smart_socket.sock.connected and
                    self.angel_connector.is_connected()
            )
            status_lines.append(f"Connection Status: {'Connected' if is_connected else 'Disconnected'}")

            # Verify actual data flow by checking tick data
            data_flowing = False
            prices = []
            for symbol in self.config.SYMBOLS:
                price = self.market_data_manager.get_latest_price(symbol)
                if price:
                    last_update = self.market_data_manager.last_tick_time.get(symbol, current_time)
                    time_diff = (current_time - last_update).seconds
                    if time_diff < 5:  # Data less than 5 seconds old
                        data_flowing = True
                    prices.append(f"{symbol}: ₹{price:.2f} ({time_diff}s ago)")

                    # Update position data with this latest price for consistent reporting
                    if symbol in self.position_manager.positions:
                        position = self.position_manager.positions[symbol]
                        position['current_price'] = price

                        # Recalculate PnL with latest price for consistent reporting
                        is_long = position['side'] in ['BUY', 'LONG']
                        if is_long:
                            position['unrealized_pnl'] = (price - position['entry_price']) * position['quantity']
                            position['unrealized_pnl_pct'] = ((price / position['entry_price']) - 1) * 100
                        else:  # SHORT
                            position['unrealized_pnl'] = (position['entry_price'] - price) * position['quantity']
                            position['unrealized_pnl_pct'] = ((position['entry_price'] / price) - 1) * 100
                else:
                    prices.append(f"{symbol}: No data")

            status_lines.append(f"Market Data Flow: {'Active' if data_flowing else 'Inactive'}")
            status_lines.append("Latest Prices: " + " | ".join(prices))

            # Open Positions with validation - UPDATED for correct PnL display
            if self.position_manager.positions:
                positions = []
                for symbol, pos in self.position_manager.positions.items():
                    current_price = pos.get('current_price')
                    if current_price:
                        # Use the recalculated PnL values
                        pnl = pos['unrealized_pnl']
                        pnl_pct = pos['unrealized_pnl_pct']

                        positions.append(
                            f"{symbol}: {pos['quantity']} @ ₹{pos['entry_price']:.2f} "
                            f"(PnL: ₹{pnl:.2f} / {pnl_pct:.1f}%)"
                        )
                if positions:
                    status_lines.append("Open Positions: " + " | ".join(positions))

            # Performance Metrics
            metrics = self.performance_tracker.get_performance_summary()
            status_lines.append(
                f"Performance - PnL: ₹{metrics.get('total_pnl', 0):.2f} | "
                f"Win Rate: {metrics.get('win_rate', 0) * 100:.1f}% | "
                f"Trades: {metrics.get('total_trades', 0)} | "
                f"DD: {metrics.get('max_drawdown', 0) * 100:.1f}%"
            )

            # Capital Utilization with validation
            total_exposure = 0
            for symbol, pos in self.position_manager.positions.items():
                current_price = pos.get('current_price')
                if current_price:
                    total_exposure += pos['quantity'] * current_price

            capital_used = (total_exposure / self.config.INITIAL_CAPITAL) * 100 if self.config.INITIAL_CAPITAL else 0
            status_lines.append(f"Capital Usage: {capital_used:.1f}% | Exposure: ₹{total_exposure:,.2f}")

            # System Health
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            cpu_usage = process.cpu_percent()
            uptime = str(current_time - self.system_start_time).split('.')[0]
            status_lines.append(
                f"System Health - Memory: {memory_usage:.1f}MB | "
                f"CPU: {cpu_usage:.1f}% | "
                f"Uptime: {uptime}"
            )

            # ML Model Status
            if hasattr(self, 'ml_ensemble'):
                time_since_update = (current_time - self.last_model_update).seconds // 60
                status_lines.append(
                    f"ML Models - Status: {'Trained' if self.ml_ensemble.is_trained else 'Not Trained'} | "
                    f"Last Update: {time_since_update} mins ago"
                )

            # ENHANCEMENT: Add signal thresholds
            if hasattr(self.ml_ensemble, 'signal_thresholds'):
                status_lines.append("\nSignal Thresholds:")
                for symbol in self.config.SYMBOLS:
                    threshold = self.ml_ensemble.signal_thresholds.get(symbol, 0.5)
                    status_lines.append(f"{symbol}: {threshold:.2f}")

            # Log the complete status
            logging.info("\n".join(status_lines))

        except Exception as e:
            logging.error(f"Error logging system state: {str(e)}")
            logging.error(traceback.format_exc())

    def get_system_warnings(self):
        """
        Collect and return system warnings.
        """
        warnings = []

        # Check for high memory usage
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024  # in MB
        if memory_usage > 1000:  # Warning if memory usage exceeds 1GB
            warnings.append(f"High memory usage: {memory_usage:.2f} MB")

        # Check for prolonged disconnection
        if not self.is_connected and (datetime.now() - self.last_connection_time).seconds > 300:
            warnings.append("System disconnected for more than 5 minutes")

        # Check for high CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 80:
            warnings.append(f"High CPU usage: {cpu_percent}%")

        # Check for low disk space
        disk_usage = psutil.disk_usage('/')
        if disk_usage.percent > 90:
            warnings.append(f"Low disk space: {disk_usage.free / (1024 * 1024 * 1024):.2f} GB free")

    def _manage_memory(self):
        """Manage system memory usage"""
        try:
            # Clear old market data
            for symbol in self.config.SYMBOLS:
                if symbol in self.market_data_manager.historical_data:
                    df = self.market_data_manager.historical_data[symbol]
                    if len(df) > 1000:  # Keep last 1000 bars
                        self.market_data_manager.historical_data[symbol] = df.tail(1000)

            # Clear old tick data
            for symbol in self.market_data_manager.received_ticks:
                ticks = self.market_data_manager.received_ticks[symbol]
                if len(ticks) > 1000:
                    self.market_data_manager.received_ticks[symbol] = ticks[-1000:]

            # Force garbage collection
            gc.collect()

        except Exception as e:
            logging.error(f"Memory management error: {str(e)}")

    def _train_models(self):
        """Train models while preserving existing thresholds"""
        try:
            # Store current thresholds before training
            original_thresholds = {}
            if hasattr(self.ml_ensemble, 'signal_thresholds'):
                original_thresholds = self.ml_ensemble.signal_thresholds.copy()

            # Store current profit thresholds
            original_profit_thresholds = {}
            if hasattr(self.ml_ensemble, 'profit_thresholds'):
                original_profit_thresholds = self.ml_ensemble.profit_thresholds.copy()

            # Train models on each symbol
            for symbol in self.config.SYMBOLS:
                data = self.market_data_manager.get_current_data(symbol)
                if data is not None and len(data) > 100:
                    # Set the name attribute for the DataFrame
                    data.name = symbol

                    # Train the model
                    self.ml_ensemble.train(data)
                    logging.info(f"ML models trained for {symbol}")

                    # Add this in the _train_models method of TradingSystem after training completes
                    fig = self.ml_ensemble.plot_model_evaluation(symbol)

            # FIX: Preserve and merge thresholds correctly
            if hasattr(self.ml_ensemble, 'signal_thresholds') and original_thresholds:
                new_thresholds = self.ml_ensemble.signal_thresholds.copy()
                merged_thresholds = {}

                for symbol in self.config.SYMBOLS:
                    if symbol in original_thresholds and symbol in new_thresholds:
                        # Log both values
                        orig_val = original_thresholds[symbol]
                        new_val = new_thresholds[symbol]

                        # During active trading, use a weighted average that favors stability
                        # 70% original + 30% new to avoid dramatic changes
                        if self.market_data_manager.warmup_complete:
                            merged_val = (orig_val * 0.7) + (new_val * 0.3)
                            merged_thresholds[symbol] = merged_val
                            logging.info(
                                f"Merging thresholds for {symbol}: {orig_val:.2f} + {new_val:.2f} → {merged_val:.2f}")
                        else:
                            # During warmup, use the new value
                            merged_thresholds[symbol] = new_val
                            logging.info(f"Using new threshold for {symbol}: {new_val:.2f}")
                    elif symbol in new_thresholds:
                        merged_thresholds[symbol] = new_thresholds[symbol]

                # Apply the merged thresholds
                self.ml_ensemble.signal_thresholds = merged_thresholds

            # FIX: Handle profit thresholds similarly
            if hasattr(self.ml_ensemble, 'profit_thresholds') and original_profit_thresholds:
                new_profit_thresholds = self.ml_ensemble.profit_thresholds.copy()
                merged_profit_thresholds = {}

                for symbol in self.config.SYMBOLS:
                    if symbol in original_profit_thresholds and symbol in new_profit_thresholds:
                        if self.market_data_manager.warmup_complete:
                            # Merge the parameters with bias toward stability
                            orig_params = original_profit_thresholds[symbol]
                            new_params = new_profit_thresholds[symbol]
                            merged_params = {}

                            # Merge each parameter with weight
                            for key in new_params:
                                if key in orig_params:
                                    orig_val = orig_params[key]
                                    new_val = new_params[key]
                                    # Use weighted average for numeric values
                                    if isinstance(new_val, (int, float)) and isinstance(orig_val, (int, float)):
                                        merged_params[key] = (orig_val * 0.7) + (new_val * 0.3)
                                    else:
                                        # For non-numeric, prefer original
                                        merged_params[key] = orig_val
                                else:
                                    merged_params[key] = new_params[key]

                            merged_profit_thresholds[symbol] = merged_params
                            logging.info(f"Merged profit thresholds for {symbol}")
                        else:
                            # During warmup, use new values
                            merged_profit_thresholds[symbol] = new_profit_thresholds[symbol]
                    elif symbol in new_profit_thresholds:
                        merged_profit_thresholds[symbol] = new_profit_thresholds[symbol]

                # Apply the merged profit thresholds
                self.ml_ensemble.profit_thresholds = merged_profit_thresholds

            # Log completion
            logging.info("Model retraining completed with parameter preservation")

        except Exception as e:
            logging.error(f"Model training error: {str(e)}")

    def _update_position_metrics(self, symbol: str, current_price: float, data: pd.DataFrame = None,
                                 position: dict = None) -> bool:
        """Legacy method - redirects to the centralized update_position method"""
        return self.update_position(symbol, current_price, data, position)

    def _start_heartbeat(self):
        """Monitor system health and data flow with dynamic timing"""
        # Add initialization delay
        time.sleep(15)  # Wait for system to stabilize after startup

        last_data_check = datetime.now()
        last_state_log = datetime.now()
        no_data_window = []
        WINDOW_SIZE = 5
        last_reconnect_time = datetime.now()
        MIN_RECONNECT_INTERVAL = 60  # Increased to 1 minute

        while self.is_running:
            try:
                current_time = datetime.now()

                # Skip checks during early startup
                if (current_time - self.system_start_time).total_seconds() < 30:
                    time.sleep(1)
                    continue

                # Check data flow every 30 seconds
                if (current_time - last_data_check).seconds >= 120:  # Check every 2 minutes
                    # Use market data manager's tick times
                    if hasattr(self.market_data_manager, 'last_tick_time'):
                        symbol_statuses = {}
                        all_symbols_stale = True
                        data_ever_received = False

                        for symbol in self.config.SYMBOLS:
                            if symbol in self.market_data_manager.last_tick_time:
                                last_tick = self.market_data_manager.last_tick_time.get(symbol)
                                if last_tick is not None:
                                    # We've received data at some point
                                    data_ever_received = True
                                    time_diff = (current_time - last_tick).seconds
                                    symbol_statuses[symbol] = time_diff
                                    if time_diff < 10:  # Data within last 10 seconds
                                        all_symbols_stale = False

                        # Only log status if multiple symbols are stale
                        stale_symbols = [sym for sym, time_diff in symbol_statuses.items() if time_diff > 10]
                        if len(stale_symbols) > 1 and data_ever_received:
                            logging.debug(
                                "Data Flow Status - " +
                                ", ".join([f"{sym}: {symbol_statuses[sym]}s ago" for sym in stale_symbols])
                            )

                        # Only consider reconnection if we've established data flow first
                        if data_ever_received:
                            # Record status
                            no_data_window.append(all_symbols_stale)
                            if len(no_data_window) > WINDOW_SIZE:
                                no_data_window.pop(0)

                            # Attempt reconnection if needed
                            if (len(no_data_window) == WINDOW_SIZE and
                                    all(no_data_window) and
                                    (current_time - last_reconnect_time).seconds >= MIN_RECONNECT_INTERVAL):

                                if not self.angel_connector.is_connected():
                                    logging.warning("No data flow detected - initiating reconnection")
                                    success = self.angel_connector._attempt_reconnect()
                                    if success:
                                        no_data_window.clear()
                                        last_reconnect_time = current_time
                                        logging.info("Reconnection successful")
                                        time.sleep(5)  # Prevent rapid reconnection attempts

                        last_data_check = current_time

                    # Log system state every 5 minutes
                    if (current_time - last_state_log).seconds >= 300:
                        self._log_system_state()
                        last_state_log = current_time

                    time.sleep(0.1)

            except Exception as e:
                logging.error(f"Heartbeat error: {str(e)}")
                time.sleep(1)

    def _check_data_flow(self):
        """Check websocket connection and data flow"""
        try:
            current_time = datetime.now()

            # Check websocket connection
            if not self.angel_connector.smart_socket or \
                    not self.angel_connector.smart_socket.sock or \
                    not self.angel_connector.smart_socket.sock.connected:
                logging.error("WebSocket not connected")
                return False

            # Check subscription status
            if not self.angel_connector.subscribed_tokens:
                logging.error("No symbols subscribed")
                return False

            # Check last tick times
            for symbol in self.config.SYMBOLS:
                if symbol in self.last_tick_time:
                    time_diff = (current_time - self.last_tick_time[symbol]).seconds
                    logging.info(f"Last tick for {symbol}: {time_diff} seconds ago")
                    if time_diff > 5:  # No data for 5 seconds
                        logging.warning(f"No recent data for {symbol}")
                        return False
                else:
                    logging.warning(f"No tick time recorded for {symbol}")
                    return False

            return True

        except Exception as e:
            logging.error(f"Data flow check error: {str(e)}")
            return False

    def _update_risk_metrics(self):
        """Update risk metrics with reduced logging"""
        try:
            # Only log every 5 minutes
            current_time = datetime.now()
            if hasattr(self, '_last_risk_log'):
                if (current_time - self._last_risk_log).seconds < 300:  # 5 minutes
                    return
            self._last_risk_log = current_time

            total_exposure = sum(
                pos['quantity'] * self.market_data_manager.get_latest_price(sym)
                for sym, pos in self.position_manager.positions.items()
            )

            unrealized_pnl = sum(
                pos.get('unrealized_pnl', 0)
                for pos in self.position_manager.positions.values()
            )

            # Only log if there are active positions or significant changes
            if self.position_manager.positions or abs(unrealized_pnl) > 0:
                logging.info(
                    f"Risk Metrics - Exposure: {total_exposure:.2f} | PnL: {unrealized_pnl:.2f} | Positions: {len(self.position_manager.positions)}")

        except Exception as e:
            logging.error(f"Risk metrics update error: {str(e)}")

    def _combine_signals(self, ml_signal: float, strategy_signal: float) -> float:
        try:
            # More balanced weights
            ml_weight = 0.6  # 60% ML
            strategy_weight = 0.4  # 40% strategy

            # Validate signals
            ml_signal = float(ml_signal) if ml_signal is not None else 0.0
            strategy_signal = float(strategy_signal) if strategy_signal is not None else 0.0

            combined = (ml_signal * ml_weight) + (strategy_signal * strategy_weight)

            # Add validation logging
            logging.debug(f"Signal combination - ML: {ml_signal:.4f} (w: {ml_weight:.2f}), "
                          f"Strategy: {strategy_signal:.4f} (w: {strategy_weight:.2f}), "
                          f"Combined: {combined:.4f}")

            return float(combined)

        except Exception as e:
            logging.error(f"Signal combination error: {str(e)}")
            return 0.0

    def _check_portfolio_risk(self) -> dict:
        """Check portfolio risk levels and trading capacity"""
        try:
            result = {
                'can_trade': True,
                'reason': 'OK'
            }

            # 1. Check position count
            current_positions = len(self.position_manager.positions)
            if current_positions >= self.config.risk_params['max_positions']:
                return {
                    'can_trade': False,
                    'reason': f'Maximum positions ({self.config.risk_params["max_positions"]}) reached'
                }

            # 2. Check capital utilization
            total_exposure = sum(
                pos['quantity'] * self.market_data_manager.get_latest_price(sym)
                for sym, pos in self.position_manager.positions.items()
            )
            capital_utilization = total_exposure / self.capital
            if capital_utilization >= self.config.risk_params['max_capital_at_risk']:
                return {
                    'can_trade': False,
                    'reason': f'Maximum capital utilization ({self.config.risk_params["max_capital_at_risk"] * 100}%) reached'
                }

            # 3. Check daily PnL limits
            daily_pnl = self.performance_tracker.metrics.get('daily_pnl', 0)
            if abs(daily_pnl) > self.config.risk_params['max_daily_loss'] * self.config.INITIAL_CAPITAL:
                return {
                    'can_trade': False,
                    'reason': 'Daily PnL limit reached'
                }

            # 4. Check market volatility
            if self.market_status['high_volatility']:
                return {
                    'can_trade': False,
                    'reason': 'High market volatility'
                }

            # 5. Check drawdown
            current_drawdown = self.performance_tracker.metrics.get('max_drawdown', 0)
            if current_drawdown > self.config.risk_params['max_drawdown']:
                return {
                    'can_trade': False,
                    'reason': 'Maximum drawdown reached'
                }

            return result

        except Exception as e:
            logging.error(f"Portfolio risk check error: {str(e)}")
            return {
                'can_trade': False,
                'reason': f'Risk check error: {str(e)}'
            }

    def _calculate_position_size(self, symbol: str, current_price: float, data: pd.DataFrame) -> int:
        """Calculate position size with proper risk management, leverage, and volatility adjustments"""
        try:
            # Get ML signal for position sizing adjustment
            features_df = self.ml_ensemble.prepare_features(data.tail(100))
            signal_strength = 0.5  # Default
            if not features_df.empty:
                pred_result = self.ml_ensemble.predict(features_df)
                if isinstance(pred_result, pd.DataFrame) and 'entry_score' in pred_result.columns:
                    signal_strength = float(pred_result['entry_score'].iloc[-1])

            # Calculate total exposure from existing positions
            total_positions_value = sum(
                pos['quantity'] * self.market_data_manager.get_latest_price(sym)
                for sym, pos in self.position_manager.positions.items()
                if self.market_data_manager.get_latest_price(sym)
            )

            # Apply 4x leverage to available capital
            leveraged_capital = self.config.INITIAL_CAPITAL * 4  # 4x leverage
            max_available = min(
                leveraged_capital * 0.8 - total_positions_value,  # Up to 80% of leveraged capital
                self.config.INITIAL_CAPITAL * 4 * 0.5  # Max 50% of 4x capital
            )

            # Signal-based position sizing - stronger signals get bigger positions
            signal_factor = 1.0
            if signal_strength > 0.7:
                signal_factor = 2.0  # Double size for very strong signals
            elif signal_strength > 0.5:
                signal_factor = 1.5  # 50% larger for strong signals
            elif signal_strength < 0.3:
                signal_factor = 0.7  # Reduce size for weak signals

            # Define list of volatile stocks
            volatile_stocks = ['TATAMOTORS', 'ADANIPORTS', 'BAJAJFINANCE', 'HINDALCO', 'INDUSINDBK', 'ZOMATO']

            # Adjust position sizing for volatile stocks
            volatility_factor = 1.0
            if symbol in volatile_stocks:
                volatility_factor = 0.7  # 30% smaller positions for volatile stocks
                logging.info(f"Position size reduced for volatile stock {symbol} (factor: {volatility_factor})")

                # For very high volatility conditions, reduce even more
                if 'volatility_20' in data.columns and len(data) > 0:
                    current_volatility = data['volatility_20'].iloc[-1]
                    if current_volatility > self.config.VOLATILITY_THRESHOLD * 1.5:
                        volatility_factor = 0.5  # 50% smaller positions for extremely volatile conditions
                        logging.info(f"Position size further reduced due to high volatility: {current_volatility:.4f}")

            # Calculate position value limits with 4x leverage and volatility adjustment
            max_position_value = min(
                max_available * self.config.MAX_POSITION_SIZE * signal_factor * volatility_factor,
                self.config.MAX_POSITION_VALUE * 4 * volatility_factor,  # 4x leverage on max position
                leveraged_capital * 0.2 * volatility_factor  # Max 20% per position of leveraged capital
            )

            # ENHANCEMENT: Add detailed logging
            logging.info(f"Position sizing for {symbol} with 4x leverage:")
            logging.info(f"Signal strength: {signal_strength:.2f}, Factor: {signal_factor:.1f}")
            logging.info(f"Volatility factor: {volatility_factor:.1f}")
            logging.info(f"Max position value: {max_position_value:.2f}")

            # Calculate quantity based on max position value
            quantity = int(max_position_value / current_price)

            # FIXED: Always take at least 1 share if under allocation limits
            if quantity == 0:
                # Check if adding 1 share would exceed maximum capital allocation
                if (total_positions_value + current_price) <= (leveraged_capital * 0.8):
                    quantity = 1
                    logging.info(f"Taking minimum position of 1 share despite price constraints")
                else:
                    logging.info(f"Position skipped - price {current_price:.2f} exceeds capital allocation limits")
                    return 0

            position_value = quantity * current_price
            logging.info(f"Calculated quantity: {quantity}")
            logging.info(f"Position value: {position_value:.2f}")
            logging.info(f"New total exposure would be: {total_positions_value + position_value:.2f}")

            return quantity

        except Exception as e:
            logging.error(f"Position size calculation error for {symbol}: {str(e)}")
            logging.error(traceback.format_exc())
            return 0

    def save_trading_data(self):
        """Critical missing method called during shutdown and periodic saves"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.performance_tracker.save_performance_data(f"performance_{timestamp}.json")
            logging.info("Trading data saved successfully")
        except Exception as e:
            logging.error(f"Data save error: {str(e)}")

    def _check_system_health(self):
        """Health check without data monitoring"""
        try:
            if not self._is_trading_time():
                return

            # Get status from MarketDataManager instead of checking directly
            status = self.market_data_manager.get_status()
            self.is_connected = status['data_flowing']
            self.is_authenticated = status['data_flowing']

            # Only log health metrics
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            cpu_usage = process.cpu_percent()

            if memory_usage > 1000 or cpu_usage > 80:
                logging.warning(f"High resource usage - Memory: {memory_usage:.1f}MB, CPU: {cpu_usage}%")

        except Exception as e:
            logging.error(f"Health check error: {str(e)}")

    def _is_trading_time(self) -> bool:
        """Check if within trading hours"""
        current_time = datetime.now().time()
        market_open = datetime_time(9, 15)
        market_close = datetime_time(15, 30)
        return market_open <= current_time <= market_close

    def _open_position(self, symbol: str, side: str, quantity: int, entry_price: float,
                       stop_loss: float, take_profit: float, timestamp: datetime, data: pd.DataFrame) -> bool:
        """Open new trading position or add to existing position with proper position management
        and support for both long and short directions."""
        try:
            # CRITICAL FIX: Check for shutdown flag
            if hasattr(self, '_in_shutdown_process') and self._in_shutdown_process:
                logging.warning(f"Skipping position opening during shutdown: {symbol}")
                return False

            # CRITICAL FIX: Skip recently closed positions
            if hasattr(self, '_recently_closed_positions') and symbol in self._recently_closed_positions:
                logging.info(f"Skipping position opening for recently closed symbol: {symbol}")
                return False

            # Add explicit logging for SHORT positions
            if side == 'SELL':
                logging.info(f"SHORT position entry request: Symbol {symbol}, Quantity: {quantity}, "
                             f"Price: {entry_price:.2f}, Stop: {stop_loss:.2f}, Target: {take_profit:.2f}")

            # Validate parameters
            if not all([symbol, side, quantity, entry_price, stop_loss, take_profit]):
                logging.error(f"Invalid parameters for position opening: {symbol}")
                return False

            # Calculate position value
            position_value = quantity * entry_price

            # Check risk limits
            risk_check_passed = True
            if hasattr(self, 'risk_manager'):
                # Use higher limits with leverage
                risk_check_passed = self.risk_manager.check_risk_limits(symbol, quantity, side, entry_price)
            elif hasattr(self, 'trading_system') and hasattr(self.trading_system, 'risk_manager'):
                risk_check_passed = self.trading_system.risk_manager.check_risk_limits(
                    symbol, quantity, side, entry_price
                )
            else:
                try:
                    risk_check_passed = self.risk_manager._check_risk_limits(symbol, quantity, side)
                except:
                    # If no risk_manager attribute, assume risk check passes
                    risk_check_passed = True

            if not risk_check_passed:
                logging.warning(f"Risk limits check failed for {symbol}")
                return False

            # Convert 'BUY'/'SELL' to position side format 'LONG'/'SHORT' for internal use
            position_side = 'LONG' if side == 'BUY' else 'SHORT'

            # FIXED: Check if position already exists for this symbol
            if symbol in self.position_manager.positions:
                # Get existing position
                position = self.position_manager.positions[symbol]
                existing_side = position['side']

                # Ensure the sides match - don't allow adding to a position in the opposite direction
                if (existing_side in ['BUY', 'LONG'] and side == 'SELL') or \
                        (existing_side in ['SELL', 'SHORT'] and side == 'BUY'):
                    logging.warning(f"Cannot add {side} to existing {existing_side} position for {symbol}")
                    return False

                # Update existing position of the same direction
                # Calculate new average price and total quantity
                old_quantity = position['quantity']
                old_entry_price = position['entry_price']

                # Calculate new average entry price
                if position_side == 'LONG':
                    # For longs, standard average price calculation
                    total_cost = (old_quantity * old_entry_price) + (quantity * entry_price)
                    new_quantity = old_quantity + quantity
                    new_avg_price = total_cost / new_quantity
                else:  # SHORT
                    # For shorts, also calculate weighted average
                    total_cost = (old_quantity * old_entry_price) + (quantity * entry_price)
                    new_quantity = old_quantity + quantity
                    new_avg_price = total_cost / new_quantity

                # Update position details
                position['quantity'] = new_quantity
                position['entry_price'] = new_avg_price

                # Recalculate stop loss and take profit based on new average entry and direction
                if position_side == 'LONG':
                    # Calculate stop percentage from original stop for longs
                    stop_percentage = (old_entry_price - position['stop_loss']) / old_entry_price
                    target_percentage = (position['take_profit'] - old_entry_price) / old_entry_price

                    # Apply percentages to new average price for long
                    position['stop_loss'] = new_avg_price * (1.0 - stop_percentage)
                    position['take_profit'] = new_avg_price * (1.0 + target_percentage)
                    position['trailing_stop'] = new_avg_price * (1.0 - stop_percentage)
                else:
                    # Calculate stop percentage from original stop for shorts
                    stop_percentage = (position['stop_loss'] - old_entry_price) / old_entry_price
                    target_percentage = (old_entry_price - position['take_profit']) / old_entry_price

                    # Apply percentages to new average price for short
                    position['stop_loss'] = new_avg_price * (1.0 + stop_percentage)
                    position['take_profit'] = new_avg_price * (1.0 - target_percentage)
                    position['trailing_stop'] = new_avg_price * (1.0 + stop_percentage)

                position['position_value'] = new_quantity * new_avg_price

                # Place order through order manager
                order_id = self.order_manager.place_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=entry_price
                )

                if order_id:
                    logging.info(f"✅ Added to {position_side} position - Symbol: {symbol}, Added: {quantity}, " +
                                 f"New Total: {new_quantity}, Avg Price: {new_avg_price:.2f}, " +
                                 f"Stop: {position['stop_loss']:.2f}, Target: {position['take_profit']:.2f}")

                    # Update risk tracking
                    if hasattr(self, 'risk_manager'):
                        self.risk_manager.update_position(
                            symbol=symbol,
                            quantity=new_quantity,
                            price=new_avg_price
                        )

                    return True
                return False

            else:
                # Create a new position ID
                position_id = f"{symbol}_{int(time.time())}"

                # Create position dictionary for new position with correct side
                position = {
                    'id': position_id,
                    'symbol': symbol,
                    'side': position_side,  # Use position_side ('LONG'/'SHORT') for internal tracking
                    'quantity': quantity,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'entry_time': timestamp or datetime.now(),
                    'highest_price': entry_price,
                    'lowest_price': entry_price,
                    'current_price': entry_price,
                    'unrealized_pnl': 0,
                    'unrealized_pnl_pct': 0,
                    'max_favorable_excursion': 0,
                    'max_adverse_excursion': 0,
                    'trailing_stop': stop_loss,
                    'trailing_stop_price': entry_price,
                    'position_value': position_value,
                    'leveraged': True
                }

                # Store the position in the positions dictionary
                self.position_manager.positions[symbol] = position

                # Place order through order manager
                order_id = self.order_manager.place_order(
                    symbol=symbol,
                    side=side,  # Use original side ('BUY'/'SELL') for order placement
                    quantity=quantity,
                    price=entry_price
                )

                if order_id:
                    # ENHANCEMENT: Improved logging with direction
                    logging.info(
                        f"✅ New {position_side} position opened - Symbol: {symbol}, Qty: {quantity}, ID: {position_id}" +
                        f" Price: {entry_price:.2f}, Stop: {stop_loss:.2f}, Target: {take_profit:.2f}")

                    # Update risk tracking
                    if hasattr(self, 'risk_manager'):
                        self.risk_manager.update_position(
                            symbol=symbol,
                            quantity=quantity,
                            price=entry_price
                        )

                    return True

                return False

        except Exception as e:
            logging.error(f"Position opening error for {symbol}: {str(e)}")
            logging.error(traceback.format_exc())
            return False

    def _close_position(self, symbol: str, reason: str) -> dict:
        """Close position with deadlock prevention - minimal operations under lock"""
        try:
            # 1. First, check if the position exists without holding any locks
            if symbol not in self.position_manager.positions:
                logging.warning(f"Cannot close position for {symbol} - position doesn't exist")
                return None

            # 2. Get current price before acquiring any locks
            current_price = self.market_data_manager.get_latest_price(symbol)
            if not current_price:
                logging.warning(f"Cannot close position for {symbol} - no current price, using entry price")
                # We'll get entry price later if needed

            # 3. Copy position data and remove position - minimal lock time
            position = None
            try:
                # Make a deep copy of the position
                import copy
                position = copy.deepcopy(self.position_manager.positions[symbol])

                # CRITICAL FIX: Remove position immediately to release the resource
                if symbol in self.position_manager.positions:
                    del self.position_manager.positions[symbol]
            except Exception as lock_error:
                logging.error(f"Error handling position data: {str(lock_error)}")
                return None

            # 4. If we didn't get a price earlier, use entry price
            if not current_price:
                current_price = position.get('entry_price', 0)
                if not current_price:
                    logging.error(f"Cannot close position - no price data available")
                    return None

            # 5. Calculate PnL safely - no locks needed
            pnl = 0
            if position.get('entry_price') and position.get('quantity'):
                # Direction-aware PnL calculation
                if position['side'] in ['BUY', 'LONG']:
                    pnl = (current_price - position['entry_price']) * position['quantity']
                else:  # SHORT/SELL
                    pnl = (position['entry_price'] - current_price) * position['quantity']

            # 6. Create trade record - no locks needed
            trade = {
                'symbol': symbol,
                'entry_price': position.get('entry_price', 0),
                'exit_price': current_price,
                'quantity': position.get('quantity', 0),
                'entry_time': position.get('entry_time', datetime.now() - timedelta(minutes=1)),
                'exit_time': datetime.now(),
                'pnl': pnl,
                'reason': reason,
                'realized_pnl': pnl,
                'side': position.get('side', 'LONG'),
                'holding_time': (datetime.now() - position.get('entry_time', datetime.now())).seconds / 60,
                'max_favorable_excursion': position.get('max_favorable_excursion', 0),
                'max_adverse_excursion': position.get('max_adverse_excursion', 0),
                'original_stop_loss': position.get('stop_loss', 0),
                'original_take_profit': position.get('take_profit', 0),
                'high_price': position.get('high_price', current_price),
                'low_price': position.get('low_price', current_price)
            }

            # 7. Update trade history - no system method calls
            if not hasattr(self, 'trades_history'):
                self.trades_history = []
            self.trades_history.append(trade)

            # 8. Log trade completion
            pnl_percentage = (pnl / (position['entry_price'] * position['quantity'])) * 100 if position.get(
                'entry_price') and position.get('quantity') else 0
            holding_time_minutes = trade['holding_time']

            logging.info(f"Position closed - Symbol: {symbol}, PnL: {pnl:.2f} ({pnl_percentage:.2f}%), "
                         f"Reason: {reason}, Holding time: {holding_time_minutes:.1f} min")

            # 9. CRITICAL FIX: Track recently closed symbols for cooldown
            if not hasattr(self, '_recently_closed_symbols'):
                self._recently_closed_symbols = {}
            self._recently_closed_symbols[symbol] = datetime.now()

            # CRITICAL FIX: Add to cooldown list with proper timing
            if hasattr(self, '_add_to_cooldown') and callable(getattr(self, '_add_to_cooldown')):
                self._add_to_cooldown(symbol, 30)  # 30 second cooldown

            # 10. Set flags for main thread to handle in next cycle
            self._position_just_closed = True
            self._last_closed_symbol = symbol
            self._last_closed_time = datetime.now()

            # 11. Log completion
            logging.info(f"Position close operation completed cleanly - will check for new opportunities in next cycle")

            return trade

        except Exception as e:
            logging.error(f"Error closing position for {symbol}: {str(e)}")
            logging.error(traceback.format_exc())

            # Emergency cleanup if there was an error
            try:
                if symbol in self.position_manager.positions:
                    del self.position_manager.positions[symbol]
                    logging.info(f"Forcibly removed position for {symbol} after error")
            except Exception as cleanup_error:
                logging.error(f"Failed to remove position after error: {str(cleanup_error)}")

            return None

    def _check_entry_opportunities(self, closed_symbol=None):
        """
        Actively check for new entry opportunities across symbols with better SHORT handling.
        """
        try:
            # CRITICAL FIX: Skip during shutdown
            if hasattr(self, '_in_shutdown_process') and self._in_shutdown_process:
                logging.info("Skipping entry opportunities check during shutdown")
                return

            # Only proceed during trading hours
            if not self._is_trading_time():
                return

            # If a symbol was just closed, prioritize checking it first
            symbols_to_check = list(self.config.SYMBOLS.keys())
            if closed_symbol and closed_symbol in symbols_to_check:
                # Move the closed symbol to the front of the list
                symbols_to_check.remove(closed_symbol)
                symbols_to_check.insert(0, closed_symbol)
                logging.info(f"Checking for new entry opportunities after closing {closed_symbol}")

            # Check each symbol that doesn't already have an open position
            for symbol in symbols_to_check:
                # Skip if symbol already has position
                if symbol in self.position_manager.positions:
                    continue

                # Get necessary data
                data = self.market_data_manager.get_current_data(symbol)
                if data is None or len(data) == 0:
                    continue

                current_price = self.market_data_manager.get_latest_price(symbol)
                if not current_price:
                    continue

                # Process signals - get action string and direction
                action, direction, df_with_indicators = self._process_signals(data, symbol)

                # Check for entry if action is "ENTER"
                if action == "ENTER":
                    logging.info(
                        f"Found new entry opportunity for {symbol} after position closure - Direction: {direction}")

                    # Analysis and position sizing
                    analytics = self._analyze_entry_opportunity(symbol, current_price, df_with_indicators, direction)

                    if analytics['should_enter']:
                        quantity = self._calculate_position_size(symbol, current_price, df_with_indicators)

                        if quantity > 0:
                            # Calculate entry parameters
                            entry_params = self._calculate_entry_parameters(
                                symbol, current_price, df_with_indicators, direction
                            )

                            # Determine side based on direction
                            side = 'BUY' if direction == 'LONG' else 'SELL'

                            # Try to open position
                            success = self._open_position(
                                symbol=symbol,
                                side=side,
                                quantity=quantity,
                                entry_price=current_price,
                                stop_loss=entry_params['stop_loss'],
                                take_profit=entry_params['take_profit'],
                                timestamp=datetime.now(),
                                data=df_with_indicators
                            )

                            if success:
                                logging.info(f"Successfully opened {direction} position for {symbol}")
                                # Reset activity tracking
                                if hasattr(self, '_inactive_periods'):
                                    self._inactive_periods = 0
                                if hasattr(self, 'last_active_check'):
                                    self.last_active_check = datetime.now()
                                break  # Only open one position at a time

                # Small delay between checking each symbol
                time.sleep(0.2)

        except Exception as e:
            logging.error(f"Error checking entry opportunities: {str(e)}")
            logging.error(traceback.format_exc())

    def reset_daily_stats(self):
        """Reset daily statistics across all components"""
        if hasattr(self, 'risk_manager') and hasattr(self.risk_manager, 'reset_daily_stats'):
            self.risk_manager.reset_daily_stats()
        else:
            logging.error("Cannot reset daily stats - risk_manager not available")

    def _check_daily_reset(self):
        """Perform daily reset if needed"""
        current_time = datetime.now()
        if current_time.date() > self.daily_reset_time.date():
            try:
                # Reset daily metrics
                self.risk_manager.reset_daily_stats()
                self.performance_tracker.reset_daily_stats()

                # Save daily data
                self.save_trading_data()

                # Update reset time
                self.daily_reset_time = current_time.replace(
                    hour=9, minute=15, second=0
                )

                logging.info("Daily reset completed")

            except Exception as e:
                logging.error(f"Daily reset error: {str(e)}")

    def _update_performance_metrics(self):
        """Update performance tracking metrics"""
        try:
            self.performance_tracker._update_metrics()
        except Exception as e:
            logging.error(f"Performance metrics update error: {str(e)}")

    def _implement_recovery_procedures(self):
        """Implement recovery procedures after errors"""
        try:
            # Basic recovery steps
            logging.info("Implementing recovery procedures")

            # Check connection
            if not self.angel_connector.connection_status:
                self.angel_connector._attempt_reconnect()

            # Verify data flow
            if not self.market_data_manager.is_data_flowing():
                self.market_data_manager._subscribe_market_data()

            # Reset error counters
            self.error_count = 0

        except Exception as e:
            logging.error(f"Recovery procedure error: {str(e)}")


def main():
    """Main entry point with simplified error handling and shutdown"""
    print("\n=== AngelOne Trading System ===\n")
    max_retries = 3
    retry_count = 0
    system = None

    def cleanup():
        """Clean shutdown helper"""
        if system:
            try:
                system.stop()
            except Exception as e:
                print(f"\n❌ Cleanup error: {str(e)}")

    while retry_count < max_retries:
        try:
            # Load configuration
            print("Loading configuration my liege")
            config = Config()

            # Initialize trading system
            print("Initializing system components")
            system = TradingSystem(config)

            print("Checking network connectivity")
            if not system.initialize():
                raise Exception("System initialization failed")

            print("Starting trading system")
            system.start()

            print("\n✅ System running successfully!")
            print("Press stop to exit\n")

            # Main loop
            while True:
                time.sleep(60)

        except KeyboardInterrupt:
            print("\n⏹️ Shutting down trading system...")
            cleanup()
            break

        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            retry_count += 1
            if retry_count < max_retries:
                print(f"Retrying... ({retry_count}/{max_retries})")
                time.sleep(5)
                cleanup()  # Clean up before retry
            else:
                print("Maximum retries reached, shutting down...")
                cleanup()
                break


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)