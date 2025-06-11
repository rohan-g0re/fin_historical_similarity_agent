"""
7-Day Window Creator for Pattern Matching

Creates sliding 7-day windows with comprehensive feature vectors combining:
- Price movement patterns (daily returns, cumulative returns, volatility)
- Technical indicator patterns (all 5 core indicators across 7 days)
- Normalized features optimized for similarity calculations

Each window represents a "market fingerprint" for pattern matching.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

from ..core.config_manager import ConfigManager
from ..indicators.technical_indicators import TechnicalIndicators


class WindowCreator:
    """
    High-performance 7-day window creator for pattern matching.
    
    Creates comprehensive feature vectors from sliding windows that include:
    - Price patterns: returns, volatility, momentum
    - Technical indicators: RSI, MACD, BB Position, Volume ROC, ATR Percentile
    - Normalized features for effective similarity comparison
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the window creator.
        
        Args:
            config_manager (ConfigManager, optional): Configuration manager instance.
                                                     Creates new one if None.
        """
        self.config = config_manager if config_manager else ConfigManager()
        self.window_config = self.config.get_window_config()
        
        # Window parameters
        self.lookback_days = self.window_config.get('lookback_days', 7)
        self.feature_types = self.window_config.get('feature_types', [
            'price_returns', 'intraday_volatility', 'cumulative_returns', 'technical_indicators'
        ])
        self.normalization = self.window_config.get('normalization', 'z_score')
        
        # Initialize technical indicators calculator
        self.indicators = TechnicalIndicators(config_manager)
        
        # Scalers for normalization
        self._scaler = None
        self._setup_scaler()
    
    def _setup_scaler(self) -> None:
        """Setup the appropriate scaler based on configuration."""
        if self.normalization == 'z_score':
            self._scaler = StandardScaler()
        elif self.normalization == 'min_max':
            self._scaler = MinMaxScaler()
        else:
            self._scaler = None  # No normalization
    
    def _calculate_price_features(self, data: pd.DataFrame, window_start: int, window_end: int) -> Dict[str, List[float]]:
        """
        Calculate price-based features for a 7-day window.
        
        Args:
            data (pd.DataFrame): Complete dataset with price data
            window_start (int): Start index of window (inclusive)
            window_end (int): End index of window (exclusive)
            
        Returns:
            Dict[str, List[float]]: Price features for the window
        """
        window_data = data.iloc[window_start:window_end]
        features = {}
        
        # Daily returns (7 values)
        if 'price_returns' in self.feature_types:
            if 'daily_return' in window_data.columns:
                returns = window_data['daily_return'].fillna(0).tolist()
            else:
                # Calculate if not present
                returns = window_data['close'].pct_change().fillna(0).tolist()
            features['daily_returns'] = returns
        
        # Intraday volatility (7 values)
        if 'intraday_volatility' in self.feature_types:
            if 'intraday_volatility' in window_data.columns:
                volatility = window_data['intraday_volatility'].fillna(0).tolist()
            else:
                # Calculate if not present
                volatility = ((window_data['high'] - window_data['low']) / window_data['close']).fillna(0).tolist()
            features['intraday_volatility'] = volatility
        
        # Cumulative returns (1 value - 7-day total return)
        if 'cumulative_returns' in self.feature_types:
            start_price = window_data['close'].iloc[0]
            end_price = window_data['close'].iloc[-1]
            cumulative_return = (end_price - start_price) / start_price if start_price != 0 else 0
            features['cumulative_return'] = [cumulative_return]
        
        # Additional price features
        # Price momentum (3-day vs 7-day)
        if len(window_data) >= 7:
            price_3d = window_data['close'].iloc[3]
            price_7d = window_data['close'].iloc[-1]
            momentum = (price_7d - price_3d) / price_3d if price_3d != 0 else 0
            features['price_momentum'] = [momentum]
        
        # Volatility of returns in window
        if 'daily_returns' in features and len(features['daily_returns']) > 1:
            volatility_window = np.std(features['daily_returns'])
            features['window_volatility'] = [volatility_window]
        
        return features
    
    def _calculate_indicator_features(self, data: pd.DataFrame, window_start: int, window_end: int) -> Dict[str, List[float]]:
        """
        Calculate technical indicator features for a 7-day window.
        
        Args:
            data (pd.DataFrame): Complete dataset with indicator data
            window_start (int): Start index of window (inclusive)
            window_end (int): End index of window (exclusive)
            
        Returns:
            Dict[str, List[float]]: Indicator features for the window
        """
        if 'technical_indicators' not in self.feature_types:
            return {}
        
        window_data = data.iloc[window_start:window_end]
        features = {}
        
        # Required indicator columns
        indicator_columns = ['rsi', 'macd_signal', 'bb_position', 'volume_roc', 'atr_percentile']
        
        for indicator in indicator_columns:
            if indicator in window_data.columns:
                # Get all 7 days of indicator values
                values = window_data[indicator].ffill().fillna(50.0).tolist()
                features[f'{indicator}_values'] = values
                
                # Add derived features for better pattern matching
                if len(values) > 1:
                    # Trend of indicator over window
                    indicator_trend = values[-1] - values[0]
                    features[f'{indicator}_trend'] = [indicator_trend]
                    
                    # Average indicator value
                    indicator_mean = np.mean(values)
                    features[f'{indicator}_mean'] = [indicator_mean]
        
        return features
    
    def create_window_features(self, data: pd.DataFrame, window_end_date: Union[str, pd.Timestamp, int]) -> Dict[str, List[float]]:
        """
        Create comprehensive feature vector for a single 7-day window.
        
        Args:
            data (pd.DataFrame): Complete dataset with price and indicator data
            window_end_date (Union[str, pd.Timestamp, int]): End date/index of window
            
        Returns:
            Dict[str, List[float]]: Complete feature dictionary for the window
            
        Raises:
            ValueError: If insufficient data or invalid window specification
        """
        # Convert window_end_date to index if needed
        if isinstance(window_end_date, (str, pd.Timestamp)):
            if isinstance(window_end_date, str):
                window_end_date = pd.to_datetime(window_end_date)
            
            # Find the index for this date
            try:
                window_end_idx = data.index.get_indexer([window_end_date], method='nearest')[0]
                if window_end_idx == -1:
                    raise ValueError(f"Date {window_end_date} not found in data")
            except Exception:
                raise ValueError(f"Invalid date format or date not found: {window_end_date}")
        else:
            window_end_idx = int(window_end_date)
        
        # Validate window bounds
        window_start_idx = window_end_idx - self.lookback_days + 1
        
        if window_start_idx < 0:
            raise ValueError(f"Insufficient data for window: need {self.lookback_days} days, start index would be {window_start_idx}")
        
        if window_end_idx >= len(data):
            raise ValueError(f"Window end index {window_end_idx} exceeds data length {len(data)}")
        
        # Calculate features
        features = {}
        
        # Price features
        price_features = self._calculate_price_features(data, window_start_idx, window_end_idx + 1)
        features.update(price_features)
        
        # Indicator features
        indicator_features = self._calculate_indicator_features(data, window_start_idx, window_end_idx + 1)
        features.update(indicator_features)
        
        # Add metadata
        features['window_start_date'] = [data.index[window_start_idx].strftime('%Y-%m-%d')]
        features['window_end_date'] = [data.index[window_end_idx].strftime('%Y-%m-%d')]
        features['window_start_idx'] = [window_start_idx]
        features['window_end_idx'] = [window_end_idx]
        
        return features
    
    def create_feature_vector(self, features: Dict[str, List[float]]) -> np.ndarray:
        """
        Convert feature dictionary to normalized feature vector for similarity calculation.
        
        Args:
            features (Dict[str, List[float]]): Feature dictionary from create_window_features()
            
        Returns:
            np.ndarray: Normalized feature vector ready for similarity calculation
        """
        # Exclude metadata from feature vector
        metadata_keys = ['window_start_date', 'window_end_date', 'window_start_idx', 'window_end_idx']
        numeric_features = []
        
        # Collect all numeric features in a consistent order
        for key in sorted(features.keys()):
            if key not in metadata_keys and isinstance(features[key], list):
                # Ensure all values are numeric
                numeric_values = []
                for val in features[key]:
                    if isinstance(val, (int, float, np.number)):
                        numeric_values.append(float(val))
                    else:
                        numeric_values.append(0.0)  # Default for non-numeric
                
                numeric_features.extend(numeric_values)
        
        # Convert to numpy array
        feature_vector = np.array(numeric_features)
        
        # Handle any NaN or infinite values
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return feature_vector
    
    def normalize_feature_vector(self, feature_vector: np.ndarray, fit_data: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Normalize feature vector for similarity calculation.
        
        Args:
            feature_vector (np.ndarray): Raw feature vector
            fit_data (np.ndarray, optional): Data to fit scaler on. If None, uses the feature_vector itself.
            
        Returns:
            np.ndarray: Normalized feature vector
        """
        if self._scaler is None:
            return feature_vector
        
        # Reshape for sklearn (expects 2D)
        vector_2d = feature_vector.reshape(1, -1)
        
        if fit_data is not None:
            # Fit scaler on provided data and transform the vector
            if len(fit_data.shape) == 1:
                fit_data = fit_data.reshape(1, -1)
            self._scaler.fit(fit_data)
            normalized = self._scaler.transform(vector_2d)
        else:
            # Fit and transform on the same data (not ideal but works for single vectors)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                normalized = self._scaler.fit_transform(vector_2d)
        
        return normalized.flatten()
    
    def create_sliding_windows(self, data: pd.DataFrame, min_gap_days: int = 1) -> List[Dict[str, any]]:
        """
        Create sliding 7-day windows across entire dataset.
        
        Args:
            data (pd.DataFrame): Complete dataset with price and indicator data
            min_gap_days (int): Minimum gap between windows (default: 1 day)
            
        Returns:
            List[Dict]: List of window dictionaries with features and metadata
        """
        if len(data) < self.lookback_days:
            raise ValueError(f"Insufficient data: need at least {self.lookback_days} days, got {len(data)}")
        
        windows = []
        
        # Create windows from first possible position to last possible position
        start_idx = self.lookback_days - 1  # First valid window end index
        end_idx = len(data) - 1  # Last valid window end index
        
        current_idx = start_idx
        
        while current_idx <= end_idx:
            try:
                # Create features for this window
                features = self.create_window_features(data, current_idx)
                
                # Create feature vector
                feature_vector = self.create_feature_vector(features)
                
                # Create window record
                window = {
                    'window_end_idx': current_idx,
                    'window_start_idx': current_idx - self.lookback_days + 1,
                    'window_end_date': data.index[current_idx],
                    'window_start_date': data.index[current_idx - self.lookback_days + 1],
                    'features': features,
                    'feature_vector': feature_vector,
                    'vector_length': len(feature_vector)
                }
                
                windows.append(window)
                
            except Exception as e:
                print(f"⚠ Warning: Failed to create window at index {current_idx}: {e}")
            
            # Move to next window position
            current_idx += min_gap_days
        
        print(f"✓ Created {len(windows)} sliding windows from {len(data)} days of data")
        return windows
    
    def get_current_window(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        Get the most recent 7-day window (current market conditions).
        
        Args:
            data (pd.DataFrame): Complete dataset with latest data
            
        Returns:
            Dict: Current window with features and metadata
        """
        if len(data) < self.lookback_days:
            raise ValueError(f"Insufficient data for current window: need {self.lookback_days} days, got {len(data)}")
        
        # Use the last available window
        current_end_idx = len(data) - 1
        features = self.create_window_features(data, current_end_idx)
        feature_vector = self.create_feature_vector(features)
        
        current_window = {
            'window_end_idx': current_end_idx,
            'window_start_idx': current_end_idx - self.lookback_days + 1,
            'window_end_date': data.index[current_end_idx],
            'window_start_date': data.index[current_end_idx - self.lookback_days + 1],
            'features': features,
            'feature_vector': feature_vector,
            'vector_length': len(feature_vector),
            'is_current': True
        }
        
        return current_window
    
    def get_feature_names(self) -> List[str]:
        """
        Get ordered list of feature names for the feature vector.
        
        Returns:
            List[str]: Ordered feature names
        """
        # This would need to be generated based on actual feature calculation
        # For now, return a template list
        feature_names = []
        
        if 'price_returns' in self.feature_types:
            feature_names.extend([f'daily_return_day_{i}' for i in range(self.lookback_days)])
        
        if 'intraday_volatility' in self.feature_types:
            feature_names.extend([f'intraday_vol_day_{i}' for i in range(self.lookback_days)])
        
        if 'cumulative_returns' in self.feature_types:
            feature_names.append('cumulative_return')
            feature_names.append('price_momentum')
            feature_names.append('window_volatility')
        
        if 'technical_indicators' in self.feature_types:
            indicators = ['rsi', 'macd_signal', 'bb_position', 'volume_roc', 'atr_percentile']
            for indicator in indicators:
                feature_names.extend([f'{indicator}_day_{i}' for i in range(self.lookback_days)])
                feature_names.extend([f'{indicator}_trend', f'{indicator}_mean'])
        
        return feature_names
    
    def get_window_summary(self, window: Dict[str, any]) -> Dict[str, any]:
        """
        Get summary information about a window.
        
        Args:
            window (Dict): Window dictionary from create_window_features()
            
        Returns:
            Dict: Window summary information
        """
        features = window.get('features', {})
        
        summary = {
            'date_range': f"{window['window_start_date']} to {window['window_end_date']}",
            'vector_length': window.get('vector_length', 0),
            'feature_count': len([k for k in features.keys() if k not in ['window_start_date', 'window_end_date', 'window_start_idx', 'window_end_idx']])
        }
        
        # Add key feature summaries
        if 'daily_returns' in features:
            returns = features['daily_returns']
            summary['total_return'] = sum(returns) * 100  # Convert to percentage
            summary['avg_daily_return'] = np.mean(returns) * 100
        
        if 'window_volatility' in features:
            summary['volatility'] = features['window_volatility'][0]
        
        # Add current indicator values
        for indicator in ['rsi', 'macd_signal', 'bb_position', 'volume_roc', 'atr_percentile']:
            values_key = f'{indicator}_values'
            if values_key in features and features[values_key]:
                summary[f'current_{indicator}'] = features[values_key][-1]  # Last day value
        
        return summary 