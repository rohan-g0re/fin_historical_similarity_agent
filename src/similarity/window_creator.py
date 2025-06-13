"""
7-Day Window Creator for Pattern Matching

Creates sliding 7-day windows with comprehensive feature vectors combining:
- Price movement patterns (daily returns, cumulative returns, volatility)
- Technical indicator patterns (all 5 core indicators across 7 days)
- Normalized features optimized for similarity calculations

Each window represents a "market fingerprint" for pattern matching.

This module implements the core windowing methodology that transforms continuous
financial time series into discrete, comparable market pattern segments.

Key Design Principles:

1. **Temporal Coherence**: 7-day windows capture weekly market cycles while
   maintaining sufficient granularity for pattern detection

2. **Feature Completeness**: Each window combines price action, volatility,
   volume, and technical indicators for comprehensive characterization

3. **Mathematical Normalization**: All features are scaled and normalized to
   enable meaningful similarity comparisons across different symbols and time periods

4. **Vector Optimization**: Features are engineered specifically for cosine
   similarity calculations, emphasizing relative patterns over absolute values

5. **Computational Efficiency**: Sliding window approach enables efficient
   processing of large historical datasets

The 7-day timeframe provides optimal balance between:
- Pattern granularity (enough detail to capture market dynamics)
- Statistical robustness (sufficient data points for reliable calculations)
- Market relevance (weekly cycles are meaningful in financial markets)
- Computational efficiency (manageable vector dimensions for similarity analysis)
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
    
    This class implements the sophisticated window creation engine that transforms
    continuous financial time series into discrete, comparable market patterns.
    Each window represents a comprehensive "market fingerprint" containing:
    
    **Price Pattern Features:**
    - Daily returns sequence (7 values): Captures price momentum and direction
    - Intraday volatility sequence (7 values): Measures daily trading ranges
    - Cumulative return (1 value): Overall period performance
    - Price momentum (1 value): Mid-period vs end-period comparison
    
    **Technical Indicator Features:**
    - RSI values sequence (7 values): Momentum oscillator progression
    - MACD Signal sequence (7 values): Trend strength evolution
    - Bollinger Position sequence (7 values): Volatility-adjusted price position
    - Volume ROC sequence (7 values): Participation level changes
    - ATR Percentile sequence (7 values): Volatility regime context
    
    **Derived Pattern Features:**
    - Indicator trends (5 values): Direction of change for each indicator
    - Indicator means (5 values): Average levels within the window
    - Window volatility (1 value): Overall price volatility for the period
    
    The complete feature vector contains ~62 dimensions that comprehensively
    characterize market behavior patterns suitable for similarity analysis.
    
    **Normalization Strategy:**
    - Price features: Relative (percentage-based) to handle different price levels
    - Indicator features: Already normalized by design (RSI 0-100, BB Position 0-1)
    - Optional z-score or min-max scaling for cross-temporal stability
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the window creator with configuration and indicator calculator.
        
        Sets up the complete window creation infrastructure including:
        - Configuration loading for customizable parameters
        - Technical indicators calculator for feature generation
        - Normalization scaler setup for feature standardization
        - Feature type selection for flexible window composition
        
        Args:
            config_manager (ConfigManager, optional): Configuration manager instance.
                                                     Creates new one if None to ensure
                                                     independent operation capability.
        """
        # Initialize configuration system for centralized parameter management
        self.config = config_manager if config_manager else ConfigManager()
        self.window_config = self.config.get_window_config()
        
        # Core window parameters - these define the fundamental windowing approach
        self.lookback_days = self.window_config.get('lookback_days', 7)  # Standard 7-day windows
        
        # Feature type configuration - enables flexible feature composition
        # Different feature types capture different aspects of market behavior
        self.feature_types = self.window_config.get('feature_types', [
            'price_returns',        # Daily price movement patterns
            'intraday_volatility',  # Daily trading range patterns
            'cumulative_returns',   # Overall period performance
            'technical_indicators'  # All 5 core technical indicators
        ])
        
        # Normalization method selection for cross-temporal stability
        self.normalization = self.window_config.get('normalization', 'z_score')
        
        # Initialize technical indicators calculator for feature generation
        self.indicators = TechnicalIndicators(config_manager)
        
        # Set up normalization infrastructure
        self._scaler = None
        self._setup_scaler()
    
    def _setup_scaler(self) -> None:
        """
        Setup the appropriate normalization scaler based on configuration.
        
        Different normalization methods serve different purposes:
        - Z-score (StandardScaler): Centers features around mean=0, std=1
          Good for normally distributed features and preserving relative patterns
        - Min-Max (MinMaxScaler): Scales features to 0-1 range
          Good for bounded features and ensuring equal weight in distance calculations
        - None: No additional normalization beyond inherent feature scaling
          Relies on features being pre-normalized (like RSI, Bollinger Position)
        """
        if self.normalization == 'z_score':
            self._scaler = StandardScaler()  # Mean=0, Std=1 normalization
        elif self.normalization == 'min_max':
            self._scaler = MinMaxScaler()    # 0-1 range normalization
        else:
            self._scaler = None              # No additional normalization
    
    def _calculate_price_features(self, data: pd.DataFrame, window_start: int, window_end: int) -> Dict[str, List[float]]:
        """
        Calculate price-based features for a 7-day window.
        
        Extracts and calculates all price-related features that capture the
        essential characteristics of price movement within the window period.
        These features form the foundation of the market pattern fingerprint.
        
        **Feature Categories:**
        
        1. **Daily Returns** (7 values): Day-by-day percentage price changes
           - Captures short-term momentum and volatility patterns
           - Essential for identifying similar price movement sequences
        
        2. **Intraday Volatility** (7 values): Daily high-low ranges as % of close
           - Measures daily trading activity and uncertainty
           - Important for risk assessment and market condition comparison
        
        3. **Cumulative Return** (1 value): Total return over the 7-day period
           - Overall performance summary for the window
           - Helps classify trending vs consolidating periods
        
        4. **Price Momentum** (1 value): 3-day vs 7-day price comparison
           - Captures acceleration/deceleration patterns within the window
           - Useful for identifying momentum shifts
        
        5. **Window Volatility** (1 value): Standard deviation of daily returns
           - Summary measure of price uncertainty within the window
           - Important for risk-adjusted pattern comparison
        
        Args:
            data (pd.DataFrame): Complete dataset with price data
            window_start (int): Start index of window (inclusive)
            window_end (int): End index of window (exclusive)
            
        Returns:
            Dict[str, List[float]]: Dictionary of price features with descriptive names
                                   Each feature is a list to maintain sequence information
        """
        # Extract the specific window data for analysis
        window_data = data.iloc[window_start:window_end]
        features = {}
        
        # Daily returns (7 values) - core price movement sequence
        if 'price_returns' in self.feature_types:
            if 'daily_return' in window_data.columns:
                # Use pre-calculated daily returns if available
                returns = window_data['daily_return'].fillna(0).tolist()
            else:
                # Calculate daily returns on-the-fly if needed
                returns = window_data['close'].pct_change().fillna(0).tolist()
            features['daily_returns'] = returns
        
        # Intraday volatility (7 values) - daily trading range patterns
        if 'intraday_volatility' in self.feature_types:
            if 'intraday_volatility' in window_data.columns:
                # Use pre-calculated intraday volatility if available
                volatility = window_data['intraday_volatility'].fillna(0).tolist()
            else:
                # Calculate intraday volatility as (High-Low)/Close ratio
                # This normalizes the trading range by the price level
                volatility = ((window_data['high'] - window_data['low']) / window_data['close']).fillna(0).tolist()
            features['intraday_volatility'] = volatility
        
        # Cumulative returns (1 value) - overall window performance
        if 'cumulative_returns' in self.feature_types:
            start_price = window_data['close'].iloc[0]
            end_price = window_data['close'].iloc[-1]
            # Calculate total return over the window period
            cumulative_return = (end_price - start_price) / start_price if start_price != 0 else 0
            features['cumulative_return'] = [cumulative_return]
        
        # Additional derived price features for enhanced pattern recognition
        
        # Price momentum (1 value) - acceleration/deceleration within window
        if len(window_data) >= 7:
            # Compare mid-window price to end-window price
            price_3d = window_data['close'].iloc[3]  # Mid-window (day 4 of 7)
            price_7d = window_data['close'].iloc[-1] # End-window (day 7 of 7)
            # Calculate momentum as percentage change from mid to end
            momentum = (price_7d - price_3d) / price_3d if price_3d != 0 else 0
            features['price_momentum'] = [momentum]
        
        # Window volatility (1 value) - summary risk measure for the period
        if 'daily_returns' in features and len(features['daily_returns']) > 1:
            # Standard deviation of daily returns within the window
            volatility_window = np.std(features['daily_returns'])
            features['window_volatility'] = [volatility_window]
        
        return features
    
    def _calculate_indicator_features(self, data: pd.DataFrame, window_start: int, window_end: int) -> Dict[str, List[float]]:
        """
        Calculate technical indicator features for a 7-day window.
        
        Extracts all technical indicator values and derived features that capture
        the market regime and sentiment characteristics within the window.
        These features provide the analytical depth needed for sophisticated
        pattern matching beyond simple price movements.
        
        **Feature Structure for Each Indicator:**
        
        1. **Raw Values** (7 values each): Complete sequence of indicator values
           - RSI: Momentum oscillator values (0-100)
           - MACD Signal: Trend strength values (unbounded)
           - Bollinger Position: Volatility-adjusted price position (0-1)
           - Volume ROC: Participation change percentages
           - ATR Percentile: Volatility regime ranking (0-100)
        
        2. **Trend Features** (1 value each): Direction of change within window
           - Calculated as (Last Value - First Value) for each indicator
           - Captures whether the indicator is strengthening or weakening
        
        3. **Mean Features** (1 value each): Average level within window
           - Provides regime context (high/low RSI, high/low volatility, etc.)
           - Important for market condition classification
        
        This creates ~35 features from technical indicators alone, providing
        comprehensive market condition characterization.
        
        Args:
            data (pd.DataFrame): Complete dataset with calculated indicators
            window_start (int): Start index of window (inclusive)
            window_end (int): End index of window (exclusive)
            
        Returns:
            Dict[str, List[float]]: Dictionary of indicator features with descriptive names
                                   Includes raw sequences, trends, and means for each indicator
        """
        if 'technical_indicators' not in self.feature_types:
            return {}
        
        # Extract the specific window data for indicator analysis
        window_data = data.iloc[window_start:window_end]
        features = {}
        
        # Core technical indicator columns that should be present
        indicator_columns = ['rsi', 'macd_signal', 'bb_position', 'volume_roc', 'atr_percentile']
        
        for indicator in indicator_columns:
            if indicator in window_data.columns:
                # Extract the complete 7-day sequence for this indicator
                # Forward fill missing values and use neutral defaults for any remaining gaps
                values = window_data[indicator].ffill().fillna(50.0).tolist()
                features[f'{indicator}_values'] = values
                
                # Calculate derived features for enhanced pattern recognition
                if len(values) > 1:
                    # Trend of indicator over the window (directional change)
                    # Positive = strengthening, Negative = weakening
                    indicator_trend = values[-1] - values[0]
                    features[f'{indicator}_trend'] = [indicator_trend]
                    
                    # Average indicator value for regime classification
                    # Provides context about typical levels during this period
                    indicator_mean = np.mean(values)
                    features[f'{indicator}_mean'] = [indicator_mean]
        
        return features
    
    def create_window_features(self, data: pd.DataFrame, window_end_date: Union[str, pd.Timestamp, int]) -> Dict[str, List[float]]:
        """
        Create comprehensive feature vector for a single 7-day window.
        
        This is the core method that orchestrates the complete feature extraction
        process for a single market window. It combines price patterns and
        technical indicators into a unified feature dictionary suitable for
        vector creation and similarity analysis.
        
        **Complete Feature Set (typical ~62 dimensions):**
        
        **Price Features (~12 dimensions):**
        - Daily returns (7): Day-by-day price movements
        - Intraday volatility (7): Daily trading ranges
        - Cumulative return (1): Overall period performance  
        - Price momentum (1): Mid-to-end acceleration
        - Window volatility (1): Period risk summary
        
        **Technical Indicator Features (~50 dimensions):**
        - RSI sequence (7) + trend (1) + mean (1) = 9 features
        - MACD Signal sequence (7) + trend (1) + mean (1) = 9 features
        - Bollinger Position sequence (7) + trend (1) + mean (1) = 9 features
        - Volume ROC sequence (7) + trend (1) + mean (1) = 9 features
        - ATR Percentile sequence (7) + trend (1) + mean (1) = 9 features
        
        The resulting feature vector provides comprehensive market characterization
        suitable for cosine similarity-based pattern matching.
        
        Args:
            data (pd.DataFrame): Complete dataset with price and indicator data
            window_end_date (Union[str, pd.Timestamp, int]): End date/index of window
                                                            Supports multiple input formats
            
        Returns:
            Dict[str, List[float]]: Complete feature dictionary for the window
                                   Ready for vector creation and similarity analysis
            
        Raises:
            ValueError: If insufficient data or invalid window specification
        """
        # Convert window_end_date to index for consistent processing
        if isinstance(window_end_date, (str, pd.Timestamp)):
            # Handle string or timestamp input
            if isinstance(window_end_date, str):
                window_end_date = pd.to_datetime(window_end_date)
            
            # Find the corresponding index in the dataset
            try:
                # Use nearest matching for robust date handling
                window_end_idx = data.index.get_indexer([window_end_date], method='nearest')[0]
                if window_end_idx == -1:
                    raise ValueError(f"Date {window_end_date} not found in data")
            except Exception:
                raise ValueError(f"Invalid date format or date not found: {window_end_date}")
        else:
            # Handle integer index input
            window_end_idx = int(window_end_date)
        
        # Calculate window bounds with validation
        # Window is [start_idx, end_idx) - start inclusive, end exclusive
        window_start_idx = window_end_idx - self.lookback_days + 1
        
        # Validate window bounds to ensure sufficient data
        if window_start_idx < 0:
            raise ValueError(f"Insufficient data for window: need {self.lookback_days} days, "
                           f"start index would be {window_start_idx}")
        
        if window_end_idx >= len(data):
            raise ValueError(f"Window end index {window_end_idx} exceeds data length {len(data)}")
        
        # Extract features from both price and indicator data
        features = {}
        
        # Calculate price-based features (returns, volatility, momentum)
        price_features = self._calculate_price_features(data, window_start_idx, window_end_idx)
        features.update(price_features)
        
        # Calculate technical indicator features (all 5 indicators + derived)
        indicator_features = self._calculate_indicator_features(data, window_start_idx, window_end_idx)
        features.update(indicator_features)
        
        # Add metadata for tracking and debugging
        features['window_start_date'] = [data.index[window_start_idx].strftime('%Y-%m-%d')]
        features['window_end_date'] = [data.index[window_end_idx - 1].strftime('%Y-%m-%d')]
        features['window_length'] = [self.lookback_days]
        
        return features
    
    def create_feature_vector(self, features: Dict[str, List[float]]) -> np.ndarray:
        """
        Convert feature dictionary to flat numerical vector for similarity calculations.
        
        Transforms the hierarchical feature dictionary into a flat numerical array
        suitable for mathematical similarity calculations. This conversion is critical
        for the pattern matching system as it enables cosine similarity and other
        vector-based distance metrics.
        
        **Vector Construction Process:**
        
        1. **Feature Flattening**: Convert all feature lists to single flat array
        2. **Metadata Filtering**: Remove non-numerical metadata (dates, etc.)
        3. **NaN Handling**: Replace any remaining missing values with zeros
        4. **Consistency Validation**: Ensure all vectors have same dimensionality
        
        **Vector Properties:**
        - Fixed dimensionality (~62 elements) for all windows
        - Numerical values only (no strings or dates)
        - NaN-free for reliable mathematical operations
        - Maintains feature order for consistent comparison
        
        The resulting vector enables direct application of similarity algorithms
        while preserving all the pattern information from the original window.
        
        Args:
            features (Dict[str, List[float]]): Feature dictionary from create_window_features
            
        Returns:
            np.ndarray: Flat numerical vector ready for similarity calculations
                       Fixed-size array with consistent dimensionality
        """
        vector_elements = []
        
        # Define the expected feature order for consistency across all windows
        # This ensures that the same vector position always represents the same feature
        feature_order = [
            # Price features (in logical sequence)
            'daily_returns', 'intraday_volatility', 'cumulative_return', 
            'price_momentum', 'window_volatility',
            
            # Technical indicator raw sequences
            'rsi_values', 'macd_signal_values', 'bb_position_values', 
            'volume_roc_values', 'atr_percentile_values',
            
            # Technical indicator trends
            'rsi_trend', 'macd_signal_trend', 'bb_position_trend',
            'volume_roc_trend', 'atr_percentile_trend',
            
            # Technical indicator means
            'rsi_mean', 'macd_signal_mean', 'bb_position_mean',
            'volume_roc_mean', 'atr_percentile_mean'
        ]
        
        # Build vector by processing features in consistent order
        for feature_name in feature_order:
            if feature_name in features:
                feature_values = features[feature_name]
                
                # Handle different feature value types
                if isinstance(feature_values, list):
                    # Flatten list features (like daily returns sequence)
                    vector_elements.extend(feature_values)
                else:
                    # Handle single values
                    vector_elements.append(float(feature_values))
        
        # Convert to numpy array for mathematical operations
        vector = np.array(vector_elements, dtype=np.float64)
        
        # Handle any remaining NaN values that could disrupt similarity calculations
        # Replace with 0.0 as neutral value that won't bias similarity scores
        vector = np.nan_to_num(vector, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return vector
    
    def normalize_feature_vector(self, feature_vector: np.ndarray, fit_data: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply normalization to feature vector for improved similarity calculations.
        
        Normalization is crucial for effective pattern matching because:
        1. Different features have different scales (RSI 0-100, returns ±10%)
        2. Larger-scale features can dominate similarity calculations
        3. Normalized features ensure equal contribution to pattern matching
        4. Cross-temporal stability requires consistent scaling
        
        **Normalization Methods:**
        
        - **Z-Score (StandardScaler)**: Centers around mean=0, scales to std=1
          * Good for normally distributed features
          * Preserves relative patterns while equalizing scales
          * Preferred for mixed feature types
        
        - **Min-Max (MinMaxScaler)**: Scales to 0-1 range
          * Good for bounded features
          * Ensures equal weight in distance calculations  
          * Preferred when feature distributions are known
        
        - **None**: No additional normalization
          * Relies on features being pre-normalized
          * Suitable when RSI, Bollinger Position already provide scaling
        
        Args:
            feature_vector (np.ndarray): Raw feature vector to normalize
            fit_data (np.ndarray, optional): Data to fit scaler parameters
                                           If None, fit on the single vector
            
        Returns:
            np.ndarray: Normalized feature vector ready for similarity analysis
        """
        if self._scaler is None:
            # No normalization requested - return vector as-is
            return feature_vector
        
        # Ensure vector is 2D for sklearn compatibility
        if len(feature_vector.shape) == 1:
            vector_2d = feature_vector.reshape(1, -1)
        else:
            vector_2d = feature_vector
        
        try:
            if fit_data is not None:
                # Fit scaler on provided data (e.g., historical vectors)
                # This enables consistent normalization across time periods
                if len(fit_data.shape) == 1:
                    fit_data_2d = fit_data.reshape(1, -1)
                else:
                    fit_data_2d = fit_data
                
                self._scaler.fit(fit_data_2d)
                normalized = self._scaler.transform(vector_2d)
            else:
                # Fit and transform on the single vector
                # Less ideal but necessary when no historical data available
                normalized = self._scaler.fit_transform(vector_2d)
            
            # Return to original dimensionality
            return normalized.flatten()
            
        except Exception as e:
            # If normalization fails, return original vector with warning
            warnings.warn(f"Normalization failed: {e}. Using unnormalized vector.")
            return feature_vector
    
    def create_sliding_windows(self, data: pd.DataFrame, min_gap_days: int = 1) -> List[Dict[str, any]]:
        """
        Create sliding windows across entire dataset for historical analysis.
        
        Generates a comprehensive set of historical windows suitable for pattern
        matching analysis. Each window represents a distinct market pattern that
        can be compared to current conditions.
        
        **Sliding Window Strategy:**
        
        1. **Coverage**: Process entire historical dataset systematically
        2. **Gap Control**: Maintain minimum gap between windows to ensure independence
        3. **Boundary Handling**: Skip periods with insufficient data
        4. **Metadata Preservation**: Include dates and context for each window
        
        **Gap Considerations:**
        - min_gap_days = 1: Maximum overlap (6 days shared between consecutive windows)
        - min_gap_days = 7: No overlap (completely independent windows)
        - min_gap_days = 30: Monthly sampling (reduces computational load)
        
        Larger gaps reduce computational requirements but may miss subtle patterns.
        Smaller gaps provide more patterns but increase computational cost.
        
        Args:
            data (pd.DataFrame): Complete dataset with price and indicator data
            min_gap_days (int): Minimum gap between window start dates
                              Controls overlap and computational requirements
            
        Returns:
            List[Dict[str, any]]: List of window dictionaries with features and metadata
                                 Each contains complete feature set plus tracking information
        """
        windows = []
        total_periods = len(data)
        
        # Calculate valid window positions considering lookback requirements
        # Start from lookback_days to ensure sufficient history for first window
        start_idx = self.lookback_days - 1
        
        print(f"📊 Creating sliding windows with {min_gap_days}-day gaps...")
        print(f"   Data period: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        print(f"   Total periods: {total_periods}, Window size: {self.lookback_days} days")
        
        # Iterate through dataset creating windows at specified intervals
        current_idx = start_idx
        while current_idx < total_periods:
            try:
                # Create feature set for this window position
                features = self.create_window_features(data, current_idx)
                
                # Convert to feature vector for similarity calculations
                feature_vector = self.create_feature_vector(features)
                
                # Create complete window record with features and metadata
                window = {
                    'features': features,
                    'feature_vector': feature_vector,
                    'vector_length': len(feature_vector),
                    'window_start_date': data.index[current_idx - self.lookback_days + 1].strftime('%Y-%m-%d'),
                    'window_end_date': data.index[current_idx].strftime('%Y-%m-%d'),
                    'window_end_index': current_idx,
                    'data_quality_score': self._assess_window_quality(features)
                }
                
                windows.append(window)
                
            except Exception as e:
                # Log window creation failures but continue processing
                window_date = data.index[current_idx].strftime('%Y-%m-%d')
                print(f"⚠ Failed to create window ending {window_date}: {e}")
            
            # Move to next window position based on gap setting
            current_idx += min_gap_days
        
        print(f"✅ Created {len(windows)} sliding windows")
        return windows
    
    def get_current_window(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        Create window for the most recent period in the dataset.
        
        Generates the "current" window representing the latest market conditions
        that will be compared against historical patterns. This window serves as
        the target for similarity-based pattern matching.
        
        **Current Window Properties:**
        - Uses the most recent complete 7-day period
        - Includes all feature types (price + indicators)
        - Serves as comparison target for historical pattern matching
        - Contains identical feature structure to historical windows
        
        Args:
            data (pd.DataFrame): Complete dataset with latest data
            
        Returns:
            Dict[str, any]: Current window with features and metadata
                           Ready for use as pattern matching target
            
        Raises:
            ValueError: If insufficient data for current window creation
        """
        if len(data) < self.lookback_days:
            raise ValueError(f"Insufficient data for current window: {len(data)} days "
                           f"(need {self.lookback_days})")
        
        # Use the last available date as window end
        last_idx = len(data) - 1
        
        print(f"🎯 Creating current window ending {data.index[last_idx].strftime('%Y-%m-%d')}")
        
        try:
            # Create features for the current period
            features = self.create_window_features(data, last_idx)
            
            # Convert to vector for similarity calculations
            feature_vector = self.create_feature_vector(features)
            
            # Create complete current window record
            current_window = {
                'features': features,
                'feature_vector': feature_vector,
                'vector_length': len(feature_vector),
                'window_start_date': data.index[last_idx - self.lookback_days + 1].strftime('%Y-%m-%d'),
                'window_end_date': data.index[last_idx].strftime('%Y-%m-%d'),
                'window_end_index': last_idx,
                'data_quality_score': self._assess_window_quality(features),
                'is_current': True  # Flag to identify this as the target window
            }
            
            print(f"✅ Current window created successfully")
            return current_window
            
        except Exception as e:
            raise ValueError(f"Failed to create current window: {e}")
    
    def _assess_window_quality(self, features: Dict[str, List[float]]) -> float:
        """
        Assess the quality of a window's features for reliable pattern matching.
        
        Evaluates various quality metrics to identify windows with reliable
        data that will produce meaningful similarity comparisons. Poor quality
        windows can be filtered out to improve pattern matching accuracy.
        
        **Quality Factors:**
        - Feature completeness (all expected features present)
        - Data validity (no excessive NaN or extreme values)
        - Reasonable value ranges (indicators within expected bounds)
        - Pattern coherence (features align logically)
        
        Args:
            features (Dict[str, List[float]]): Window features to assess
            
        Returns:
            float: Quality score from 0.0 (poor) to 1.0 (excellent)
        """
        quality_score = 1.0
        
        # Check feature completeness
        expected_features = ['daily_returns', 'rsi_values', 'macd_signal_values']
        missing_features = [f for f in expected_features if f not in features]
        if missing_features:
            quality_score *= 0.7  # Penalize missing core features
        
        # Check for excessive missing data within features
        for feature_name, feature_values in features.items():
            if isinstance(feature_values, list) and len(feature_values) > 0:
                # Count NaN/None values
                nan_count = sum(1 for v in feature_values if v is None or (isinstance(v, float) and np.isnan(v)))
                nan_ratio = nan_count / len(feature_values)
                
                if nan_ratio > 0.3:  # More than 30% missing
                    quality_score *= (1 - nan_ratio)  # Penalize based on missing ratio
        
        # Clamp score to valid range
        return max(0.0, min(1.0, quality_score))
    
    def get_feature_names(self) -> List[str]:
        """
        Get ordered list of feature names for vector interpretation.
        
        Provides the standard feature ordering used in vector creation,
        enabling interpretation and analysis of similarity results.
        This is particularly useful for:
        - Understanding which features drive similarity
        - Debugging pattern matching results  
        - Feature importance analysis
        - Vector dimension mapping
        
        Returns:
            List[str]: Ordered list of feature names matching vector positions
        """
        feature_names = []
        
        # Price features
        if 'price_returns' in self.feature_types:
            feature_names.extend([f'daily_return_day_{i+1}' for i in range(7)])
        
        if 'intraday_volatility' in self.feature_types:
            feature_names.extend([f'intraday_vol_day_{i+1}' for i in range(7)])
        
        if 'cumulative_returns' in self.feature_types:
            feature_names.append('cumulative_return')
        
        # Additional price features
        feature_names.extend(['price_momentum', 'window_volatility'])
        
        # Technical indicator features
        if 'technical_indicators' in self.feature_types:
            indicators = ['rsi', 'macd_signal', 'bb_position', 'volume_roc', 'atr_percentile']
            
            for indicator in indicators:
                # Sequence values
                feature_names.extend([f'{indicator}_day_{i+1}' for i in range(7)])
                # Derived features
                feature_names.extend([f'{indicator}_trend', f'{indicator}_mean'])
        
        return feature_names
    
    def get_window_summary(self, window: Dict[str, any]) -> Dict[str, any]:
        """
        Generate comprehensive summary of a window's characteristics.
        
        Creates a detailed summary of window features and properties for
        analysis, debugging, and reporting purposes. This summary provides
        both statistical and interpretive information about the window.
        
        Args:
            window (Dict[str, any]): Window dictionary with features and metadata
            
        Returns:
            Dict[str, any]: Comprehensive summary with statistics and interpretations
        """
        if 'features' not in window:
            return {'error': 'No features found in window'}
        
        features = window['features']
        summary = {
            'temporal_info': {
                'start_date': window.get('window_start_date', 'Unknown'),
                'end_date': window.get('window_end_date', 'Unknown'),
                'window_length': window.get('vector_length', 0)
            },
            'feature_statistics': {},
            'market_regime': {}
        }
        
        # Analyze price features
        if 'daily_returns' in features:
            returns = features['daily_returns']
            summary['feature_statistics']['returns'] = {
                'mean': np.mean(returns),
                'volatility': np.std(returns),
                'min': np.min(returns),
                'max': np.max(returns)
            }
        
        # Analyze indicator levels
        if 'rsi_values' in features:
            rsi_avg = np.mean(features['rsi_values'])
            if rsi_avg > 70:
                summary['market_regime']['momentum'] = 'OVERBOUGHT'
            elif rsi_avg < 30:
                summary['market_regime']['momentum'] = 'OVERSOLD'
            else:
                summary['market_regime']['momentum'] = 'NEUTRAL'
        
        if 'atr_percentile_values' in features:
            atr_avg = np.mean(features['atr_percentile_values'])
            if atr_avg > 75:
                summary['market_regime']['volatility'] = 'HIGH'
            elif atr_avg < 25:
                summary['market_regime']['volatility'] = 'LOW'
            else:
                summary['market_regime']['volatility'] = 'NORMAL'
        
        # Add quality assessment
        summary['quality_score'] = window.get('data_quality_score', 0.0)
        
        return summary 