"""
Technical Indicators Calculator for Pattern Matching

Implements the 5 core technical indicators optimized for similarity analysis:
1. RSI (Relative Strength Index) - Momentum
2. MACD Signal Line - Trend & Momentum  
3. Bollinger Band Position - Volatility & Price Position
4. Volume Rate of Change (VROC) - Market Participation
5. ATR Percentile - Volatility Regime

All indicators are normalized for effective pattern matching.

This module forms the mathematical foundation of the pattern matching system.
Each indicator captures a specific dimension of market behavior:

- RSI: Measures momentum exhaustion and potential reversal points
- MACD Signal: Captures trend strength and directional changes  
- Bollinger Position: Quantifies price position within volatility bands
- Volume ROC: Identifies changing market participation levels
- ATR Percentile: Measures current volatility relative to recent history

The 5-indicator approach provides comprehensive market characterization while
maintaining computational efficiency for large-scale similarity analysis.
Key design principles:

- Mathematical Robustness: Each calculation handles edge cases gracefully
- Normalization: All outputs scaled for cross-symbol comparability  
- Performance: Optimized for batch processing of large datasets
- Stability: Consistent results across different market conditions
- Interpretability: Clear business meaning for each indicator value
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union
import ta

from ..core.config_manager import ConfigManager


class TechnicalIndicators:
    """
    High-performance technical indicators calculator for pattern matching.
    
    This class implements the core mathematical engine for market analysis,
    transforming raw price and volume data into normalized technical indicators
    suitable for similarity-based pattern matching. 
    
    Design Philosophy:
    - Each indicator captures a unique dimension of market behavior
    - Normalization ensures cross-symbol and cross-timeframe comparability
    - Robust mathematical formulations handle edge cases automatically
    - Configuration-driven parameters enable fine-tuning without code changes
    - Memory-efficient calculations support large-scale analysis
    
    The 5-indicator methodology provides comprehensive market characterization:
    
    1. Momentum (RSI): Overbought/oversold conditions and reversal potential
    2. Trend (MACD Signal): Directional bias and trend strength
    3. Volatility Position (Bollinger Bands): Price position within volatility envelope
    4. Participation (Volume ROC): Market engagement and conviction levels  
    5. Volatility Regime (ATR Percentile): Current risk environment context
    
    This combination captures both price action and market microstructure,
    enabling detection of similar market conditions across different time periods.
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the technical indicators calculator with configuration.
        
        Loads all indicator parameters from configuration system, enabling
        easy parameter tuning without code modifications. Each indicator
        has its own configuration section with default fallback values.
        
        Args:
            config_manager (ConfigManager, optional): Configuration manager instance.
                                                     Creates new one if None to ensure
                                                     independent operation capability.
        """
        # Initialize configuration system for centralized parameter management
        self.config = config_manager if config_manager else ConfigManager()
        self.indicators_config = self.config.get_indicators_config()
        
        # Load indicator-specific parameters from configuration
        # Each indicator section provides customizable parameters with sensible defaults
        self.rsi_params = self.indicators_config.get('rsi', {})
        self.macd_params = self.indicators_config.get('macd', {})
        self.bb_params = self.indicators_config.get('bollinger_bands', {})
        self.vroc_params = self.indicators_config.get('volume_roc', {})
        self.atr_params = self.indicators_config.get('atr', {})
        
    def calculate_rsi(self, data: pd.DataFrame, column: str = 'close') -> pd.Series:
        """
        Calculate Relative Strength Index (RSI) - Momentum Indicator.
        
        RSI is a momentum oscillator that measures the magnitude of recent price changes
        to evaluate overbought/oversold conditions. It oscillates between 0 and 100:
        
        - RSI > 70: Traditionally considered overbought (potential sell signal)
        - RSI < 30: Traditionally considered oversold (potential buy signal)
        - RSI 30-70: Neutral momentum zone
        
        Mathematical Foundation:
        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss over the specified period
        
        For pattern matching, RSI provides crucial momentum context that helps
        identify similar market conditions. Two periods with similar RSI values
        likely have comparable momentum characteristics.
        
        Args:
            data (pd.DataFrame): OHLCV data with required price column
            column (str): Column to calculate RSI on (default: 'close')
            
        Returns:
            pd.Series: RSI values (0-100, already normalized for cross-symbol comparison)
            
        Raises:
            ValueError: If RSI calculation fails due to insufficient or invalid data
        """
        # Get RSI period from configuration with sensible default
        period = self.rsi_params.get('period', 14)  # 14 days is traditional RSI period
        
        try:
            # Use ta library for robust RSI calculation
            # ta library handles the complex smoothing mathematics automatically
            rsi = ta.momentum.RSIIndicator(
                close=data[column],
                window=period
            ).rsi()
            
            # Handle edge cases that commonly occur in financial time series
            # Initial NaN values are inevitable due to the lookback period requirement
            rsi = rsi.fillna(50.0)  # Fill initial NaN with neutral RSI (50 = no momentum bias)
            
            # Ensure values stay within valid mathematical bounds
            # Although RSI should be 0-100 by definition, floating point errors can occur
            rsi = np.clip(rsi, 0, 100)  # Enforce strict 0-100 range
            
            return rsi
            
        except Exception as e:
            # Provide context for debugging while maintaining system stability
            raise ValueError(f"RSI calculation failed: {e}")
    
    def calculate_macd_signal(self, data: pd.DataFrame, column: str = 'close') -> pd.Series:
        """
        Calculate MACD Signal Line - Trend & Momentum Indicator.
        
        MACD (Moving Average Convergence Divergence) Signal line is the exponential
        moving average of the MACD line, providing smoother trend signals with less noise.
        
        Mathematical Foundation:
        MACD Line = EMA(12) - EMA(26)  [Fast EMA minus Slow EMA]
        Signal Line = EMA(9) of MACD Line  [Smoothed version we use]
        
        Why Signal Line instead of MACD Line:
        - Reduces noise while preserving trend information
        - More suitable for pattern matching due to smoother characteristics
        - Better captures sustained trend changes vs temporary fluctuations
        
        Interpretation:
        - Rising Signal Line: Strengthening uptrend
        - Falling Signal Line: Strengthening downtrend  
        - Flat Signal Line: Sideways/consolidation phase
        
        For pattern matching, MACD Signal provides trend context that helps identify
        periods with similar directional bias and trend strength.
        
        Args:
            data (pd.DataFrame): OHLCV data with required price column
            column (str): Column to calculate MACD on (default: 'close')
            
        Returns:
            pd.Series: MACD Signal Line values (unbounded but typically small numbers)
            
        Raises:
            ValueError: If MACD calculation fails due to insufficient or invalid data
        """
        # Get MACD parameters from configuration with traditional defaults
        fast_period = self.macd_params.get('fast_period', 12)   # Fast EMA period
        slow_period = self.macd_params.get('slow_period', 26)   # Slow EMA period  
        signal_period = self.macd_params.get('signal_period', 9) # Signal smoothing period
        
        try:
            # Calculate MACD components using robust ta library implementation
            macd_indicator = ta.trend.MACD(
                close=data[column],
                window_fast=fast_period,
                window_slow=slow_period,
                window_sign=signal_period
            )
            
            # Get MACD signal line (smoothed version preferred for pattern matching)
            # Signal line is less noisy than raw MACD line while preserving trend information
            macd_signal = macd_indicator.macd_signal()
            
            # Handle missing values that occur at the beginning due to EMA calculation requirements
            # Forward fill is appropriate since MACD represents momentum persistence
            macd_signal = macd_signal.ffill().fillna(0)  # Fill with 0 if no data available
            
            return macd_signal
            
        except Exception as e:
            # Provide detailed error context for debugging
            raise ValueError(f"MACD Signal calculation failed: {e}")
    
    def calculate_bollinger_position(self, data: pd.DataFrame, column: str = 'close') -> pd.Series:
        """
        Calculate Bollinger Band Position - Volatility & Price Position Indicator.
        
        Bollinger Band Position quantifies where the current price sits within the
        volatility-adjusted price envelope, providing a normalized measure of
        price extremes relative to recent volatility.
        
        Mathematical Foundation:
        Upper Band = SMA(n) + (k × StdDev(n))
        Lower Band = SMA(n) - (k × StdDev(n))
        Position = (Price - Lower Band) / (Upper Band - Lower Band)
        
        Normalization Benefits:
        - Range: 0 to 1 (0 = lower band, 0.5 = middle band, 1 = upper band)
        - Cross-symbol comparability: Accounts for different price levels and volatilities
        - Statistical meaning: Represents price percentile within recent volatility range
        
        Interpretation:
        - Position > 0.8: Price near upper volatility extreme (potential resistance)
        - Position < 0.2: Price near lower volatility extreme (potential support)
        - Position ≈ 0.5: Price near volatility-adjusted mean (equilibrium)
        
        For pattern matching, Bollinger Position captures the volatility context
        of price action, helping identify periods with similar risk/reward profiles.
        
        Args:
            data (pd.DataFrame): OHLCV data with required price column
            column (str): Column to calculate position for (default: 'close')
            
        Returns:
            pd.Series: Bollinger Band Position (0-1, normalized for comparison)
            
        Raises:
            ValueError: If Bollinger Band calculation fails due to insufficient data
        """
        # Get Bollinger Band parameters from configuration with standard defaults
        period = self.bb_params.get('period', 20)      # Standard deviation lookback period
        std_dev = self.bb_params.get('std_dev', 2)     # Standard deviation multiplier
        
        try:
            # Calculate Bollinger Bands using robust ta library implementation
            bb_indicator = ta.volatility.BollingerBands(
                close=data[column],
                window=period,
                window_dev=std_dev
            )
            
            # Get upper and lower band values
            upper_band = bb_indicator.bollinger_hband()
            lower_band = bb_indicator.bollinger_lband()
            
            # Calculate normalized position within the bands
            # This creates a 0-1 scale where 0.5 represents the middle band
            band_width = upper_band - lower_band
            
            # Avoid division by zero in extremely low volatility periods
            # When bands collapse (very low volatility), position becomes undefined
            band_width = band_width.replace(0, np.nan)
            
            # Calculate position as percentage of band width
            position = (data[column] - lower_band) / band_width
            
            # Handle edge cases and ensure proper normalization
            position = position.fillna(0.5)  # Fill NaN with middle band (neutral position)
            position = np.clip(position, 0, 1)  # Enforce strict 0-1 range
            
            return position
            
        except Exception as e:
            # Provide context for troubleshooting Bollinger Band calculation issues
            raise ValueError(f"Bollinger Band Position calculation failed: {e}")
    
    def calculate_volume_roc(self, data: pd.DataFrame, column: str = 'volume') -> pd.Series:
        """
        Calculate Volume Rate of Change (VROC) - Market Participation Indicator.
        
        Volume Rate of Change measures the percentage change in trading volume
        over a specified period, providing insight into changing market participation
        and conviction levels.
        
        Mathematical Foundation:
        VROC = ((Current Volume - Volume N periods ago) / Volume N periods ago) × 100
        
        Why Volume ROC Matters for Pattern Matching:
        - High positive VROC: Increasing market participation (growing interest)
        - High negative VROC: Decreasing market participation (waning interest)
        - Low VROC: Stable participation levels (consistent engagement)
        
        Volume context is crucial because:
        - Price moves on high volume are more reliable than low volume moves
        - Volume precedes price in many market scenarios
        - Similar volume patterns often indicate similar market psychology
        
        Interpretation:
        - VROC > +50%: Significantly increased participation
        - VROC < -50%: Significantly decreased participation  
        - VROC ±20%: Normal participation fluctuation range
        
        Args:
            data (pd.DataFrame): OHLCV data with volume column
            column (str): Volume column name (default: 'volume')
            
        Returns:
            pd.Series: Volume Rate of Change (percentage, capped to prevent outliers)
            
        Raises:
            ValueError: If Volume ROC calculation fails due to missing volume data
        """
        # Get VROC period from configuration with reasonable default
        period = self.vroc_params.get('period', 10)  # 10-day lookback for volume comparison
        
        try:
            volume = data[column]
            
            # Calculate percentage change over specified period
            # pct_change automatically handles the (current-previous)/previous calculation
            vroc = volume.pct_change(periods=period) * 100  # Convert to percentage
            
            # Handle edge cases common in volume data
            vroc = vroc.fillna(0)  # Fill NaN values with 0% change (no change from baseline)
            
            # Cap extreme values to prevent outliers from dominating similarity calculations
            # Volume can have extreme spikes (news events, earnings) that would skew matching
            # ±500% represents very extreme but not impossible volume changes
            vroc = np.clip(vroc, -500, 500)  # Cap at ±500% change
            
            return vroc
            
        except Exception as e:
            # Provide specific error context for volume-related issues
            raise ValueError(f"Volume ROC calculation failed: {e}")
    
    def calculate_atr_percentile(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate ATR Percentile - Volatility Regime Indicator.
        
        Average True Range (ATR) Percentile measures current volatility relative
        to recent historical volatility, providing context about the current
        risk environment and market regime.
        
        Mathematical Foundation:
        1. Calculate ATR: Average of True Range over specified period
        2. True Range = Max(High-Low, |High-PrevClose|, |Low-PrevClose|)
        3. ATR Percentile = Percentile rank of current ATR vs recent ATR history
        
        Why ATR Percentile vs Raw ATR:
        - Normalized 0-100 scale enables cross-symbol comparison
        - Relative measure more meaningful than absolute volatility
        - Percentile ranking provides regime classification context
        
        Volatility Regime Interpretation:
        - ATR Percentile > 75: High volatility regime (expect larger moves)
        - ATR Percentile < 25: Low volatility regime (expect smaller moves)
        - ATR Percentile 25-75: Normal volatility regime (average conditions)
        
        For pattern matching, ATR Percentile ensures we compare periods with
        similar volatility characteristics, as market behavior often depends
        heavily on the prevailing volatility regime.
        
        Args:
            data (pd.DataFrame): OHLCV data with High, Low, Close columns
            
        Returns:
            pd.Series: ATR Percentile (0-100, volatility regime indicator)
            
        Raises:
            ValueError: If ATR calculation fails due to missing OHLC data
        """
        # Get ATR parameters from configuration with standard defaults
        atr_period = self.atr_params.get('period', 14)           # ATR calculation period
        percentile_window = self.atr_params.get('percentile_window', 100)  # Percentile lookback window
        
        try:
            # Calculate ATR using robust ta library implementation
            # ATR automatically handles True Range calculation complexities
            atr_indicator = ta.volatility.AverageTrueRange(
                high=data['high'],
                low=data['low'],
                close=data['close'],
                window=atr_period
            )
            
            atr_values = atr_indicator.average_true_range()
            
            # Calculate rolling percentile rank to create regime indicator
            # This transforms absolute ATR values into relative percentile rankings
            atr_percentile = atr_values.rolling(
                window=percentile_window,
                min_periods=20  # Minimum periods needed for meaningful percentile
            ).rank(pct=True) * 100  # Convert to 0-100 scale
            
            # Handle edge cases in percentile calculation
            atr_percentile = atr_percentile.fillna(50.0)  # Fill NaN with neutral percentile
            atr_percentile = np.clip(atr_percentile, 0, 100)  # Ensure valid percentile range
            
            return atr_percentile
            
        except Exception as e:
            # Provide specific error context for ATR calculation issues
            raise ValueError(f"ATR Percentile calculation failed: {e}")
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all 5 core technical indicators for comprehensive analysis.
        
        This is the primary method that orchestrates the complete technical
        analysis workflow, computing all indicators needed for pattern matching.
        The method ensures consistent calculation order and handles any
        interdependencies between indicators.
        
        The 5-indicator framework provides comprehensive market characterization:
        1. RSI: Momentum and reversal potential
        2. MACD Signal: Trend direction and strength
        3. Bollinger Position: Volatility-adjusted price position
        4. Volume ROC: Market participation changes
        5. ATR Percentile: Volatility regime context
        
        Together, these indicators create a multi-dimensional "fingerprint"
        of market conditions suitable for similarity-based pattern matching.
        
        Args:
            data (pd.DataFrame): Complete OHLCV dataset with all required columns
            
        Returns:
            pd.DataFrame: Original data enhanced with all 5 technical indicators
                         Ready for window creation and pattern matching
            
        Raises:
            ValueError: If any indicator calculation fails or required data is missing
        """
        print("🔄 Calculating technical indicators...")
        
        # Create copy to avoid modifying original data
        # This preserves the input data integrity for other uses
        enhanced_data = data.copy()
        
        # Validate required columns are present
        # Each indicator has specific column requirements
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns for indicator calculation: {missing_columns}")
        
        try:
            # Calculate each indicator in logical order
            # Some indicators may benefit from others being calculated first
            
            print("  • Calculating RSI (momentum)...")
            enhanced_data['rsi'] = self.calculate_rsi(data)
            
            print("  • Calculating MACD Signal (trend)...")
            enhanced_data['macd_signal'] = self.calculate_macd_signal(data)
            
            print("  • Calculating Bollinger Position (volatility position)...")
            enhanced_data['bb_position'] = self.calculate_bollinger_position(data)
            
            print("  • Calculating Volume ROC (participation)...")
            enhanced_data['volume_roc'] = self.calculate_volume_roc(data)
            
            print("  • Calculating ATR Percentile (volatility regime)...")
            enhanced_data['atr_percentile'] = self.calculate_atr_percentile(data)
            
            # Validate all indicators were calculated successfully
            indicator_columns = ['rsi', 'macd_signal', 'bb_position', 'volume_roc', 'atr_percentile']
            for col in indicator_columns:
                if col not in enhanced_data.columns:
                    raise ValueError(f"Failed to calculate indicator: {col}")
                
                # Check for excessive NaN values that might indicate calculation issues
                nan_percentage = enhanced_data[col].isna().sum() / len(enhanced_data) * 100
                if nan_percentage > 50:  # More than 50% NaN values suggests serious issues
                    print(f"⚠ Warning: High NaN percentage in {col}: {nan_percentage:.1f}%")
            
            print(f"✅ Technical indicators calculated successfully ({len(enhanced_data)} periods)")
            
            return enhanced_data
            
        except Exception as e:
            # Provide comprehensive error context for troubleshooting
            raise ValueError(f"Technical indicators calculation failed: {e}")
    
    def get_indicator_summary(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Generate comprehensive summary statistics for all calculated indicators.
        
        Provides detailed statistical analysis of indicator behavior including
        central tendencies, distributions, and extreme values. This summary
        is valuable for:
        - Data quality assessment
        - Parameter optimization
        - Market regime analysis
        - Historical context understanding
        
        Args:
            data (pd.DataFrame): Data with calculated technical indicators
            
        Returns:
            Dict[str, Dict[str, float]]: Nested dictionary with detailed statistics
                                       for each indicator including mean, std, min, max,
                                       percentiles, and quality metrics
        """
        indicator_columns = ['rsi', 'macd_signal', 'bb_position', 'volume_roc', 'atr_percentile']
        summary = {}
        
        for indicator in indicator_columns:
            if indicator in data.columns:
                series = data[indicator].dropna()  # Remove NaN for accurate statistics
                
                summary[indicator] = {
                    # Central tendency measures
                    'mean': float(series.mean()),
                    'median': float(series.median()),
                    'std': float(series.std()),
                    
                    # Range and extremes
                    'min': float(series.min()),
                    'max': float(series.max()),
                    'range': float(series.max() - series.min()),
                    
                    # Distribution percentiles
                    'p25': float(series.quantile(0.25)),
                    'p75': float(series.quantile(0.75)),
                    'p95': float(series.quantile(0.95)),
                    'p05': float(series.quantile(0.05)),
                    
                    # Data quality metrics
                    'count': int(len(series)),
                    'missing_pct': float((len(data) - len(series)) / len(data) * 100),
                    
                    # Trend analysis
                    'trend_slope': float(np.polyfit(range(len(series)), series, 1)[0]),
                }
                
                # Add indicator-specific interpretations
                if indicator == 'rsi':
                    summary[indicator]['overbought_pct'] = float((series > 70).sum() / len(series) * 100)
                    summary[indicator]['oversold_pct'] = float((series < 30).sum() / len(series) * 100)
                
                elif indicator == 'bb_position':
                    summary[indicator]['upper_extreme_pct'] = float((series > 0.8).sum() / len(series) * 100)
                    summary[indicator]['lower_extreme_pct'] = float((series < 0.2).sum() / len(series) * 100)
                
                elif indicator == 'atr_percentile':
                    summary[indicator]['high_vol_pct'] = float((series > 75).sum() / len(series) * 100)
                    summary[indicator]['low_vol_pct'] = float((series < 25).sum() / len(series) * 100)
        
        return summary
    
    def get_indicator_zones(self, data: pd.DataFrame) -> Dict[str, str]:
        """
        Classify current market conditions based on latest indicator values.
        
        Translates numerical indicator values into business-friendly zone
        classifications that provide immediate insight into current market state.
        These zones are used for:
        - Basic filtering in pattern matching
        - Market regime classification
        - Risk assessment
        - Trading signal generation
        
        Args:
            data (pd.DataFrame): Data with calculated indicators
            
        Returns:
            Dict[str, str]: Current zone classification for each indicator
                           with human-readable descriptions
        """
        if data.empty:
            return {}
        
        # Get latest values for zone classification
        latest = data.iloc[-1]
        zones = {}
        
        # RSI Zone Classification
        if 'rsi' in latest:
            rsi_value = latest['rsi']
            if rsi_value > 70:
                zones['rsi_zone'] = 'OVERBOUGHT'
            elif rsi_value < 30:
                zones['rsi_zone'] = 'OVERSOLD'
            else:
                zones['rsi_zone'] = 'NEUTRAL'
        
        # MACD Trend Classification (based on recent change)
        if 'macd_signal' in data.columns and len(data) >= 2:
            current_macd = latest['macd_signal']
            previous_macd = data.iloc[-2]['macd_signal']
            
            if current_macd > previous_macd:
                zones['trend_zone'] = 'UPTREND'
            elif current_macd < previous_macd:
                zones['trend_zone'] = 'DOWNTREND'
            else:
                zones['trend_zone'] = 'SIDEWAYS'
        
        # Bollinger Position Zone
        if 'bb_position' in latest:
            bb_pos = latest['bb_position']
            if bb_pos > 0.8:
                zones['bb_zone'] = 'UPPER_EXTREME'
            elif bb_pos < 0.2:
                zones['bb_zone'] = 'LOWER_EXTREME'
            else:
                zones['bb_zone'] = 'MIDDLE_RANGE'
        
        # Volume Activity Classification
        if 'volume_roc' in latest:
            vol_roc = latest['volume_roc']
            if vol_roc > 50:
                zones['volume_zone'] = 'HIGH_ACTIVITY'
            elif vol_roc < -50:
                zones['volume_zone'] = 'LOW_ACTIVITY'
            else:
                zones['volume_zone'] = 'NORMAL_ACTIVITY'
        
        # Volatility Regime Classification
        if 'atr_percentile' in latest:
            atr_pct = latest['atr_percentile']
            if atr_pct > 75:
                zones['volatility_zone'] = 'HIGH_VOLATILITY'
            elif atr_pct < 25:
                zones['volatility_zone'] = 'LOW_VOLATILITY'
            else:
                zones['volatility_zone'] = 'NORMAL_VOLATILITY'
        
        return zones 