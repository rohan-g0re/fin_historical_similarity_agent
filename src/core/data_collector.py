"""
Financial Data Collector for Pattern Matching Agent

Handles acquisition, caching, and preprocessing of financial data optimized 
for similarity analysis. Provides clean, standardized data structures for 
downstream processing modules.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm

from .config_manager import ConfigManager


class FinancialDataCollector:
    """
    High-performance financial data collector with intelligent caching.
    
    Optimized for pattern matching analysis with:
    - Efficient data structures for similarity calculations
    - Intelligent caching to minimize API calls
    - Data validation and quality checks
    - Standardized format for downstream processing
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the data collector.
        
        Args:
            config_manager (ConfigManager, optional): Configuration manager instance.
                                                     Creates new one if None.
        """
        self.config = config_manager if config_manager else ConfigManager()
        self.data_config = self.config.get_data_collection_config()
        self.storage_config = self.config.get_storage_config()
        
        # Set up data directories
        self.data_dir = Path(self.storage_config.get('data_directory', 'data'))
        self.cache_dir = self.data_dir / 'cache'
        self._ensure_directories()
        
        # Cache settings
        self.use_cache = self.storage_config.get('cache_data', True)
        self.cache_duration = timedelta(days=self.storage_config.get('cache_duration_days', 1))
        
        # Data quality thresholds
        self.min_data_points = 100  # Minimum days of data required
        self.max_missing_consecutive = 5  # Max consecutive missing days allowed
    
    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, symbol: str) -> Path:
        """
        Get cache file path for a symbol.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Path: Cache file path
        """
        return self.cache_dir / f"{symbol.upper()}_data.pkl"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """
        Check if cached data is still valid.
        
        Args:
            cache_path (Path): Path to cache file
            
        Returns:
            bool: True if cache is valid and recent
        """
        if not cache_path.exists():
            return False
        
        if not self.use_cache:
            return False
        
        # Check if cache is recent enough
        cache_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return datetime.now() - cache_time < self.cache_duration
    
    def _load_from_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load data from cache if available and valid.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            pd.DataFrame or None: Cached data or None if not available
        """
        cache_path = self._get_cache_path(symbol)
        
        if not self._is_cache_valid(cache_path):
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                print(f"✓ Loaded {symbol} from cache ({len(cached_data)} days)")
                return cached_data
        except Exception as e:
            print(f"⚠ Cache load failed for {symbol}: {e}")
            return None
    
    def _save_to_cache(self, symbol: str, data: pd.DataFrame) -> None:
        """
        Save data to cache.
        
        Args:
            symbol (str): Stock symbol
            data (pd.DataFrame): Data to cache
        """
        if not self.use_cache:
            return
        
        cache_path = self._get_cache_path(symbol)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"⚠ Cache save failed for {symbol}: {e}")
    
    def _fetch_from_yfinance(self, symbol: str, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance API.
        
        Args:
            symbol (str): Stock symbol
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: Raw OHLCV data
            
        Raises:
            ValueError: If data fetch fails or insufficient data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval='1d',
                auto_adjust=True,  # Adjust for splits/dividends
                prepost=False,     # Regular hours only
                actions=False      # Don't include corporate actions
            )
            
            if data.empty:
                raise ValueError(f"No data returned for {symbol}")
            
            # Standardize column names (yfinance uses title case)
            data.columns = data.columns.str.lower()
            
            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns for {symbol}: {missing_cols}")
            
            return data
        
        except Exception as e:
            raise ValueError(f"Failed to fetch data for {symbol}: {e}")
    
    def _validate_data_quality(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Validate and clean data quality for analysis.
        
        Args:
            data (pd.DataFrame): Raw OHLCV data
            symbol (str): Stock symbol for error reporting
            
        Returns:
            pd.DataFrame: Validated and cleaned data
            
        Raises:
            ValueError: If data quality is insufficient for analysis
        """
        original_length = len(data)
        
        # Check minimum data points
        if len(data) < self.min_data_points:
            raise ValueError(f"Insufficient data for {symbol}: {len(data)} days (need {self.min_data_points})")
        
        # Remove rows with all NaN values
        data = data.dropna(how='all')
        
        # Check for excessive missing data
        missing_ratio = data.isnull().sum().max() / len(data)
        if missing_ratio > 0.1:  # More than 10% missing
            raise ValueError(f"Too much missing data for {symbol}: {missing_ratio:.1%}")
        
        # Forward fill small gaps (max 3 consecutive days)
        data = data.ffill(limit=3)
        
        # Drop remaining rows with NaN
        data = data.dropna()
        
        # Check for suspicious data (prices <= 0, volume < 0)
        invalid_prices = (data[['open', 'high', 'low', 'close']] <= 0).any(axis=1)
        invalid_volume = data['volume'] < 0
        
        if invalid_prices.any() or invalid_volume.any():
            print(f"⚠ Removing {invalid_prices.sum() + invalid_volume.sum()} invalid data points for {symbol}")
            data = data[~(invalid_prices | invalid_volume)]
        
        # Final data length check
        if len(data) < self.min_data_points:
            raise ValueError(f"Insufficient clean data for {symbol}: {len(data)} days after cleaning")
        
        data_loss = (original_length - len(data)) / original_length
        if data_loss > 0.05:  # More than 5% data loss
            print(f"⚠ Data quality warning for {symbol}: {data_loss:.1%} data loss during cleaning")
        
        return data
    
    def _add_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features optimized for similarity analysis.
        
        Args:
            data (pd.DataFrame): OHLCV data
            
        Returns:
            pd.DataFrame: Data with additional derived features
        """
        # Daily returns (key for pattern matching)
        data['daily_return'] = data['close'].pct_change()
        
        # Intraday volatility (normalized by close)
        data['intraday_volatility'] = (data['high'] - data['low']) / data['close']
        
        # Volume relative to recent average (20-day)
        data['volume_ma20'] = data['volume'].rolling(window=20, min_periods=1).mean()
        data['volume_relative'] = data['volume'] / data['volume_ma20']
        
        # Price gaps (open vs previous close)
        data['price_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
        
        # 7-day rolling features for window analysis
        data['return_7d'] = data['close'].pct_change(periods=7)
        data['volatility_7d'] = data['daily_return'].rolling(window=7, min_periods=1).std()
        data['volume_trend_7d'] = data['volume'].rolling(window=7, min_periods=1).mean()
        
        return data
    
    def collect_stock_data(self, symbol: str, start_date: Optional[str] = None, 
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Collect comprehensive stock data optimized for pattern analysis.
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            start_date (str, optional): Start date (YYYY-MM-DD). Uses config default if None.
            end_date (str, optional): End date (YYYY-MM-DD). Uses current date if None.
            
        Returns:
            pd.DataFrame: Complete dataset with OHLCV + derived features
            
        Raises:
            ValueError: If data collection or validation fails
        """
        symbol = symbol.upper()
        
        # Use config defaults if not provided
        if start_date is None:
            start_date = self.data_config.get('start_date', '2000-01-01')
        if end_date is None:
            end_date = self.data_config.get('end_date')  # None = current date
        
        print(f"📈 Collecting data for {symbol} ({start_date} to {end_date or 'current'})")
        
        # Try to load from cache first
        cached_data = self._load_from_cache(symbol)
        if cached_data is not None:
            # Check if cached data covers requested range
            if (cached_data.index.min().strftime('%Y-%m-%d') <= start_date and 
                (end_date is None or cached_data.index.max().strftime('%Y-%m-%d') >= end_date)):
                return cached_data
        
        # Fetch fresh data
        try:
            raw_data = self._fetch_from_yfinance(symbol, start_date, end_date)
            print(f"✓ Downloaded {len(raw_data)} days of data for {symbol}")
            
            # Validate data quality
            clean_data = self._validate_data_quality(raw_data, symbol)
            print(f"✓ Data validation passed for {symbol}")
            
            # Add derived features for analysis
            final_data = self._add_derived_features(clean_data)
            print(f"✓ Added derived features for {symbol}")
            
            # Cache the processed data
            self._save_to_cache(symbol, final_data)
            
            return final_data
            
        except Exception as e:
            raise ValueError(f"Data collection failed for {symbol}: {e}")
    
    def collect_multiple_stocks(self, symbols: List[str], start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Collect data for multiple stocks with progress tracking.
        
        Args:
            symbols (List[str]): List of stock symbols
            start_date (str, optional): Start date for all stocks
            end_date (str, optional): End date for all stocks
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping symbols to their data
        """
        results = {}
        failed_symbols = []
        
        print(f"📊 Collecting data for {len(symbols)} stocks...")
        
        for symbol in tqdm(symbols, desc="Fetching stock data"):
            try:
                data = self.collect_stock_data(symbol, start_date, end_date)
                results[symbol.upper()] = data
            except Exception as e:
                print(f"❌ Failed to collect {symbol}: {e}")
                failed_symbols.append(symbol)
        
        print(f"✅ Successfully collected {len(results)}/{len(symbols)} stocks")
        if failed_symbols:
            print(f"❌ Failed stocks: {failed_symbols}")
        
        return results
    
    def get_data_summary(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        Get comprehensive summary of collected data.
        
        Args:
            data (pd.DataFrame): Stock data
            
        Returns:
            Dict: Data summary statistics
        """
        return {
            'date_range': {
                'start': data.index.min().strftime('%Y-%m-%d'),
                'end': data.index.max().strftime('%Y-%m-%d'),
                'total_days': len(data)
            },
            'price_summary': {
                'min_close': data['close'].min(),
                'max_close': data['close'].max(),
                'current_close': data['close'].iloc[-1],
                'total_return': ((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100
            },
            'data_quality': {
                'missing_values': data.isnull().sum().sum(),
                'zero_volume_days': (data['volume'] == 0).sum(),
                'price_gaps_large': (abs(data['price_gap']) > 0.1).sum() if 'price_gap' in data.columns else 0
            },
            'available_features': list(data.columns)
        } 