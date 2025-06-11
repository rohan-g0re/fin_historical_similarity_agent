"""
Technical Indicators Calculator for Pattern Matching

Implements the 5 core technical indicators optimized for similarity analysis:
1. RSI (Relative Strength Index) - Momentum
2. MACD Signal Line - Trend & Momentum  
3. Bollinger Band Position - Volatility & Price Position
4. Volume Rate of Change (VROC) - Market Participation
5. ATR Percentile - Volatility Regime

All indicators are normalized for effective pattern matching.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union
import ta

from ..core.config_manager import ConfigManager


class TechnicalIndicators:
    """
    High-performance technical indicators calculator for pattern matching.
    
    Designed specifically for similarity analysis with:
    - Normalized outputs for consistent comparisons
    - Optimized calculations for large datasets
    - Robust handling of edge cases and missing data
    - Configuration-driven parameters
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the technical indicators calculator.
        
        Args:
            config_manager (ConfigManager, optional): Configuration manager instance.
                                                     Creates new one if None.
        """
        self.config = config_manager if config_manager else ConfigManager()
        self.indicators_config = self.config.get_indicators_config()
        
        # Load indicator parameters from config
        self.rsi_params = self.indicators_config.get('rsi', {})
        self.macd_params = self.indicators_config.get('macd', {})
        self.bb_params = self.indicators_config.get('bollinger_bands', {})
        self.vroc_params = self.indicators_config.get('volume_roc', {})
        self.atr_params = self.indicators_config.get('atr', {})
        
    def calculate_rsi(self, data: pd.DataFrame, column: str = 'close') -> pd.Series:
        """
        Calculate Relative Strength Index (RSI) - Momentum Indicator.
        
        RSI measures the magnitude of recent price changes to evaluate
        overbought/oversold conditions. Range: 0-100 (already normalized).
        
        Args:
            data (pd.DataFrame): OHLCV data
            column (str): Column to calculate RSI on (default: 'close')
            
        Returns:
            pd.Series: RSI values (0-100)
        """
        period = self.rsi_params.get('period', 14)
        
        try:
            # Use ta library for RSI calculation
            rsi = ta.momentum.RSIIndicator(
                close=data[column],
                window=period
            ).rsi()
            
            # Handle edge cases
            rsi = rsi.fillna(50.0)  # Fill initial NaN values with neutral (50)
            rsi = np.clip(rsi, 0, 100)  # Ensure values stay in valid range
            
            return rsi
            
        except Exception as e:
            raise ValueError(f"RSI calculation failed: {e}")
    
    def calculate_macd_signal(self, data: pd.DataFrame, column: str = 'close') -> pd.Series:
        """
        Calculate MACD Signal Line - Trend & Momentum Indicator.
        
        MACD Signal line is the EMA of the MACD line, providing smoother
        trend signals. We use the signal line for better pattern matching.
        
        Args:
            data (pd.DataFrame): OHLCV data
            column (str): Column to calculate MACD on (default: 'close')
            
        Returns:
            pd.Series: MACD Signal Line values
        """
        fast_period = self.macd_params.get('fast_period', 12)
        slow_period = self.macd_params.get('slow_period', 26)
        signal_period = self.macd_params.get('signal_period', 9)
        
        try:
            # Calculate MACD components
            macd_indicator = ta.trend.MACD(
                close=data[column],
                window_fast=fast_period,
                window_slow=slow_period,
                window_sign=signal_period
            )
            
            # Get MACD signal line (smoothed version)
            macd_signal = macd_indicator.macd_signal()
            
            # Forward fill NaN values
            macd_signal = macd_signal.ffill().fillna(0)
            
            return macd_signal
            
        except Exception as e:
            raise ValueError(f"MACD Signal calculation failed: {e}")
    
    def calculate_bollinger_position(self, data: pd.DataFrame, column: str = 'close') -> pd.Series:
        """
        Calculate Bollinger Band Position - Volatility & Price Position Indicator.
        
        Position = (Price - Lower Band) / (Upper Band - Lower Band)
        Range: 0-1 (normalized), where 0.5 = middle band, 1 = upper band, 0 = lower band
        
        Args:
            data (pd.DataFrame): OHLCV data
            column (str): Column to calculate position for (default: 'close')
            
        Returns:
            pd.Series: Bollinger Band Position (0-1)
        """
        period = self.bb_params.get('period', 20)
        std_dev = self.bb_params.get('std_dev', 2)
        
        try:
            # Calculate Bollinger Bands
            bb_indicator = ta.volatility.BollingerBands(
                close=data[column],
                window=period,
                window_dev=std_dev
            )
            
            upper_band = bb_indicator.bollinger_hband()
            lower_band = bb_indicator.bollinger_lband()
            
            # Calculate normalized position
            band_width = upper_band - lower_band
            
            # Avoid division by zero
            band_width = band_width.replace(0, np.nan)
            
            position = (data[column] - lower_band) / band_width
            
            # Handle edge cases
            position = position.fillna(0.5)  # Fill NaN with middle band
            position = np.clip(position, 0, 1)  # Ensure values stay in range [0,1]
            
            return position
            
        except Exception as e:
            raise ValueError(f"Bollinger Band Position calculation failed: {e}")
    
    def calculate_volume_roc(self, data: pd.DataFrame, column: str = 'volume') -> pd.Series:
        """
        Calculate Volume Rate of Change (VROC) - Market Participation Indicator.
        
        VROC = (Current Volume - Volume N periods ago) / Volume N periods ago
        Shows percentage change in volume over specified period.
        
        Args:
            data (pd.DataFrame): OHLCV data
            column (str): Volume column (default: 'volume')
            
        Returns:
            pd.Series: Volume Rate of Change (percentage)
        """
        period = self.vroc_params.get('period', 10)
        
        try:
            volume = data[column]
            
            # Calculate rate of change
            vroc = volume.pct_change(periods=period) * 100
            
            # Handle edge cases
            vroc = vroc.fillna(0)  # Fill NaN values with 0% change
            
            # Cap extreme values to prevent outliers from dominating similarity
            vroc = np.clip(vroc, -500, 500)  # Cap at ±500%
            
            return vroc
            
        except Exception as e:
            raise ValueError(f"Volume ROC calculation failed: {e}")
    
    def calculate_atr_percentile(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate ATR Percentile - Volatility Regime Indicator.
        
        ATR Percentile shows where current ATR ranks within historical window.
        Range: 0-100, where 100 = highest volatility in window, 0 = lowest.
        
        Args:
            data (pd.DataFrame): OHLCV data (needs high, low, close)
            
        Returns:
            pd.Series: ATR Percentile (0-100)
        """
        atr_period = self.atr_params.get('period', 14)
        percentile_window = self.atr_params.get('percentile_window', 252)  # 1 year
        
        try:
            # Calculate ATR (Average True Range)
            atr = ta.volatility.AverageTrueRange(
                high=data['high'],
                low=data['low'],
                close=data['close'],
                window=atr_period
            ).average_true_range()
            
            # Calculate rolling percentile
            atr_percentile = atr.rolling(
                window=percentile_window,
                min_periods=atr_period
            ).rank(pct=True) * 100
            
            # Handle edge cases
            atr_percentile = atr_percentile.fillna(50.0)  # Fill NaN with median
            atr_percentile = np.clip(atr_percentile, 0, 100)  # Ensure valid range
            
            return atr_percentile
            
        except Exception as e:
            raise ValueError(f"ATR Percentile calculation failed: {e}")
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all 5 core technical indicators for pattern matching.
        
        Args:
            data (pd.DataFrame): OHLCV data with required columns
            
        Returns:
            pd.DataFrame: Original data with additional indicator columns
            
        Raises:
            ValueError: If required columns are missing or calculation fails
        """
        # Validate required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns for indicators: {missing_cols}")
        
        if len(data) < 50:  # Minimum data for reliable indicator calculation
            raise ValueError(f"Insufficient data for indicators: {len(data)} days (need at least 50)")
        
        # Make a copy to avoid modifying original data
        result = data.copy()
        
        try:
            # Calculate each indicator
            print("📊 Calculating technical indicators...")
            
            # 1. RSI (Momentum)
            result['rsi'] = self.calculate_rsi(data)
            print("✓ RSI calculated")
            
            # 2. MACD Signal (Trend & Momentum)
            result['macd_signal'] = self.calculate_macd_signal(data)
            print("✓ MACD Signal calculated")
            
            # 3. Bollinger Band Position (Volatility & Price Position)
            result['bb_position'] = self.calculate_bollinger_position(data)
            print("✓ Bollinger Band Position calculated")
            
            # 4. Volume ROC (Market Participation)
            result['volume_roc'] = self.calculate_volume_roc(data)
            print("✓ Volume ROC calculated")
            
            # 5. ATR Percentile (Volatility Regime)
            result['atr_percentile'] = self.calculate_atr_percentile(data)
            print("✓ ATR Percentile calculated")
            
            print("✅ All technical indicators calculated successfully")
            
            return result
            
        except Exception as e:
            raise ValueError(f"Technical indicators calculation failed: {e}")
    
    def get_indicator_summary(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all calculated indicators.
        
        Args:
            data (pd.DataFrame): Data with calculated indicators
            
        Returns:
            Dict: Summary statistics for each indicator
        """
        indicators = ['rsi', 'macd_signal', 'bb_position', 'volume_roc', 'atr_percentile']
        
        summary = {}
        
        for indicator in indicators:
            if indicator in data.columns:
                series = data[indicator].dropna()
                summary[indicator] = {
                    'mean': float(series.mean()),
                    'std': float(series.std()),
                    'min': float(series.min()),
                    'max': float(series.max()),
                    'current': float(series.iloc[-1]) if len(series) > 0 else np.nan,
                    'count_valid': int(len(series))
                }
        
        return summary
    
    def get_indicator_zones(self, data: pd.DataFrame) -> Dict[str, str]:
        """
        Classify current indicator values into zones for basic filtering.
        
        Args:
            data (pd.DataFrame): Data with calculated indicators
            
        Returns:
            Dict: Zone classification for each indicator
        """
        zones = {}
        
        if 'rsi' in data.columns:
            current_rsi = data['rsi'].iloc[-1]
            if current_rsi > self.rsi_params.get('overbought', 70):
                zones['rsi_zone'] = 'overbought'
            elif current_rsi < self.rsi_params.get('oversold', 30):
                zones['rsi_zone'] = 'oversold'
            else:
                zones['rsi_zone'] = 'neutral'
        
        if 'bb_position' in data.columns:
            current_bb = data['bb_position'].iloc[-1]
            if current_bb > 0.8:
                zones['bb_zone'] = 'upper'
            elif current_bb < 0.2:
                zones['bb_zone'] = 'lower'
            else:
                zones['bb_zone'] = 'middle'
        
        if 'atr_percentile' in data.columns:
            current_atr = data['atr_percentile'].iloc[-1]
            if current_atr > 75:
                zones['volatility_regime'] = 'high'
            elif current_atr < 25:
                zones['volatility_regime'] = 'low'
            else:
                zones['volatility_regime'] = 'medium'
        
        if 'macd_signal' in data.columns:
            # Determine trend based on recent MACD signal direction
            recent_macd = data['macd_signal'].tail(5)
            if len(recent_macd) >= 2:
                if recent_macd.iloc[-1] > recent_macd.iloc[-2]:
                    zones['trend_direction'] = 'up'
                elif recent_macd.iloc[-1] < recent_macd.iloc[-2]:
                    zones['trend_direction'] = 'down'
                else:
                    zones['trend_direction'] = 'sideways'
        
        return zones 