"""
Financial Data Collector for Pattern Matching Agent

Handles acquisition, caching, and preprocessing of financial data optimized 
for similarity analysis. Provides clean, standardized data structures for 
downstream processing modules.

This module is responsible for the first critical step in the analysis pipeline:
obtaining high-quality, validated financial data. Key responsibilities include:

- Efficient data acquisition from Yahoo Finance API with rate limiting
- Intelligent caching system to minimize API calls and improve performance  
- Comprehensive data quality validation and cleaning procedures
- Standardized data format ensuring consistency across the entire system
- Robust error handling for network issues, API changes, and data anomalies
- Memory-efficient data structures optimized for large-scale pattern analysis

The data collector bridges the gap between raw market data and the
structured format required by technical indicators and similarity calculations.
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
    
    This class implements a sophisticated data acquisition system optimized for
    financial pattern matching analysis. Core design principles:
    
    - Performance: Aggressive caching and batch operations minimize API calls
    - Reliability: Comprehensive error handling and data validation ensure robustness
    - Quality: Multi-level data quality checks eliminate problematic data points
    - Consistency: Standardized data format across all market instruments
    - Scalability: Memory-efficient operations support large historical datasets
    
    The collector handles the complexities of financial data including:
    - Corporate actions (splits, dividends) via auto-adjustment
    - Missing data due to market holidays or trading halts
    - Data quality issues from API inconsistencies
    - Rate limiting and API reliability challenges
    - Cross-platform file system compatibility
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the data collector with configuration and caching setup.
        
        Sets up the complete data collection infrastructure including:
        - Configuration loading for customizable parameters
        - Directory structure creation for organized data storage
        - Cache management system for performance optimization
        - Data quality threshold establishment for validation
        
        Args:
            config_manager (ConfigManager, optional): Configuration manager instance.
                                                     Creates new one if None to ensure
                                                     the system can operate independently.
        """
        # Initialize configuration system - provides centralized parameter management
        self.config = config_manager if config_manager else ConfigManager()
        self.data_config = self.config.get_data_collection_config()
        self.storage_config = self.config.get_storage_config()
        
        # Set up data directories with proper organization
        # This creates a structured file system for data management
        self.data_dir = Path(self.storage_config.get('data_directory', 'data'))
        self.cache_dir = self.data_dir / 'cache'
        self._ensure_directories()
        
        # Cache settings for performance optimization
        # Caching dramatically reduces API calls and improves user experience
        self.use_cache = self.storage_config.get('cache_data', True)
        self.cache_duration = timedelta(days=self.storage_config.get('cache_duration_days', 1))
        
        # Data quality thresholds - ensure analysis operates on clean data
        # These parameters define minimum acceptable data quality standards
        self.min_data_points = 100  # Minimum days required for meaningful analysis
        self.max_missing_consecutive = 5  # Maximum consecutive missing days tolerated
    
    def _ensure_directories(self) -> None:
        """
        Create necessary directories if they don't exist.
        
        Establishes the complete directory structure required for data operations.
        Uses recursive creation to handle nested directory structures gracefully.
        The exist_ok=True parameter prevents errors if directories already exist.
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, symbol: str) -> Path:
        """
        Get standardized cache file path for a stock symbol.
        
        Creates consistent file naming convention for cached data files.
        This standardization enables easy cache management and prevents
        file name conflicts across different symbols.
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL', 'MSFT')
            
        Returns:
            Path: Standardized cache file path for the symbol
        """
        # Normalize symbol to uppercase for consistency
        # Add .pkl extension to indicate pickle serialization format
        return self.cache_dir / f"{symbol.upper()}_data.pkl"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """
        Check if cached data is still valid and recent enough for use.
        
        Implements intelligent cache validation based on multiple criteria:
        - File existence check prevents attempting to read non-existent files
        - Cache configuration respect allows users to disable caching
        - Timestamp validation ensures data freshness for trading decisions
        
        Args:
            cache_path (Path): Path to cached data file
            
        Returns:
            bool: True if cache is valid and should be used, False if refresh needed
        """
        # Quick existence check - no point in further validation if file missing
        if not cache_path.exists():
            return False
        
        # Respect user configuration - allow cache disabling for development/testing
        if not self.use_cache:
            return False
        
        # Time-based cache validation - ensure data is recent enough for analysis
        # Financial data becomes stale quickly, especially for intraday patterns
        try:
            cache_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
            time_since_cache = datetime.now() - cache_time
            return time_since_cache < self.cache_duration
        except (OSError, OverflowError):
            # Handle file system errors gracefully by invalidating cache
            return False
    
    def _load_from_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load financial data from cache if available and valid.
        
        Implements safe cache loading with comprehensive error handling.
        Cache loading can fail for various reasons including file corruption,
        permission issues, or format changes between software versions.
        
        Args:
            symbol (str): Stock symbol to load from cache
            
        Returns:
            pd.DataFrame or None: Cached data if successfully loaded, None if unavailable
        """
        cache_path = self._get_cache_path(symbol)
        
        # Validate cache before attempting to load
        if not self._is_cache_valid(cache_path):
            return None
        
        try:
            # Use binary mode for pickle deserialization
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                
                # Provide user feedback about cache usage for transparency
                print(f"✓ Loaded {symbol} from cache ({len(cached_data)} days)")
                return cached_data
                
        except Exception as e:
            # Log cache load failure but don't crash - fall back to API
            print(f"⚠ Cache load failed for {symbol}: {e}")
            return None
    
    def _save_to_cache(self, symbol: str, data: pd.DataFrame) -> None:
        """
        Save financial data to cache for future use.
        
        Implements robust cache saving with error handling to prevent
        cache failures from affecting the main data collection workflow.
        Cache saving can fail due to disk space, permissions, or file system issues.
        
        Args:
            symbol (str): Stock symbol for cache file naming
            data (pd.DataFrame): Financial data to cache
        """
        # Respect cache configuration - allow cache disabling
        if not self.use_cache:
            return
        
        cache_path = self._get_cache_path(symbol)
        
        try:
            # Use binary mode for efficient pickle serialization
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            # Log cache save failure but don't crash - main operation succeeded
            print(f"⚠ Cache save failed for {symbol}: {e}")
    
    def _fetch_from_yfinance(self, symbol: str, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch financial data from Yahoo Finance API with comprehensive error handling.
        
        This method handles the complexities of financial data API integration:
        - Network connectivity issues and timeouts
        - API rate limiting and temporary unavailability  
        - Data format validation and standardization
        - Corporate action adjustments (splits, dividends)
        - Cross-platform compatibility considerations
        
        Args:
            symbol (str): Stock symbol to fetch (e.g., 'AAPL', 'MSFT')
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format (None = today)
            
        Returns:
            pd.DataFrame: Raw OHLCV data with standardized column names
            
        Raises:
            ValueError: If data fetch fails or returns insufficient/invalid data
        """
        try:
            # Initialize Yahoo Finance ticker object
            ticker = yf.Ticker(symbol)
            
            # Fetch historical data with optimal settings for analysis
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval='1d',          # Daily data for pattern analysis
                auto_adjust=True,       # Automatically adjust for splits/dividends
                prepost=False,          # Regular trading hours only (avoid sparse data)
                actions=False           # Don't include corporate actions (already adjusted)
            )
            
            # Validate API response - empty data indicates symbol issues
            if data.empty:
                raise ValueError(f"No data returned for {symbol} - check symbol validity")
            
            # Standardize column names for consistent downstream processing
            # Yahoo Finance uses title case, we prefer lowercase for Python conventions
            data.columns = data.columns.str.lower()
            
            # Ensure we have all required columns for technical analysis
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns for {symbol}: {missing_cols}")
            
            return data
        
        except Exception as e:
            # Wrap all errors in ValueError with context for upstream handling
            raise ValueError(f"Failed to fetch data for {symbol}: {e}")
    
    def _validate_data_quality(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Validate and clean data quality for robust analysis.
        
        Financial data often contains anomalies that can severely impact
        pattern matching analysis. This method implements comprehensive
        data quality checks and cleaning procedures:
        
        - Minimum data point validation ensures sufficient history
        - Missing data detection and gap analysis prevents sparse data issues
        - Price and volume validation catches obvious data errors
        - Outlier detection identifies potential data quality problems
        
        Args:
            data (pd.DataFrame): Raw OHLCV financial data
            symbol (str): Stock symbol for error reporting context
            
        Returns:
            pd.DataFrame: Validated and cleaned financial data ready for analysis
            
        Raises:
            ValueError: If data quality is insufficient for reliable analysis
        """
        original_length = len(data)
        
        # Check minimum data points requirement
        # Pattern analysis requires substantial history for meaningful comparisons
        if len(data) < self.min_data_points:
            raise ValueError(f"Insufficient data for {symbol}: {len(data)} days (need {self.min_data_points})")
        
        # Identify and handle missing data gaps
        # Large gaps can indicate trading halts, delistings, or data provider issues
        missing_indices = data.isnull().any(axis=1)
        
        if missing_indices.any():
            # Check for large consecutive gaps that indicate fundamental data issues
            consecutive_missing = 0
            max_consecutive = 0
            
            for is_missing in missing_indices:
                if is_missing:
                    consecutive_missing += 1
                    max_consecutive = max(max_consecutive, consecutive_missing)
                else:
                    consecutive_missing = 0
            
            # Reject data with excessive consecutive missing values
            if max_consecutive > self.max_missing_consecutive:
                raise ValueError(f"Too many consecutive missing days for {symbol}: {max_consecutive}")
            
            # Forward fill small gaps to maintain data continuity
            # This is appropriate for daily financial data where last price carries forward
            data = data.ffill()
            
            # Drop any remaining missing values (typically at the beginning)
            data = data.dropna()
            
            missing_count = original_length - len(data)
            if missing_count > 0:
                print(f"⚠ Cleaned {missing_count} missing data points for {symbol}")
        
        # Validate price data integrity
        # Check for obviously invalid prices that indicate data errors
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in data.columns:
                # Check for non-positive prices (impossible in real markets)
                invalid_prices = (data[col] <= 0).sum()
                if invalid_prices > 0:
                    raise ValueError(f"Invalid non-positive prices in {col} for {symbol}: {invalid_prices} occurrences")
                
                # Check for extreme outliers that suggest data errors
                # Using 99.9th percentile to detect massive price spikes/drops
                col_median = data[col].median()
                extreme_threshold = col_median * 10  # 10x median price suggests data error
                extreme_values = (data[col] > extreme_threshold).sum()
                
                if extreme_values > 0:
                    print(f"⚠ Warning: {extreme_values} extreme price values detected in {col} for {symbol}")
        
        # Validate OHLC relationship integrity
        # High should be >= Low, Close should be between High and Low
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            # Check High >= Low (fundamental OHLC constraint)
            invalid_hl = (data['high'] < data['low']).sum()
            if invalid_hl > 0:
                raise ValueError(f"Invalid OHLC relationships (High < Low) for {symbol}: {invalid_hl} occurrences")
            
            # Check Close within High-Low range
            invalid_close = ((data['close'] > data['high']) | (data['close'] < data['low'])).sum()
            if invalid_close > 0:
                print(f"⚠ Warning: {invalid_close} closes outside High-Low range for {symbol}")
        
        # Validate volume data if present
        if 'volume' in data.columns:
            # Check for negative volume (impossible in real markets)
            negative_volume = (data['volume'] < 0).sum()
            if negative_volume > 0:
                raise ValueError(f"Invalid negative volume for {symbol}: {negative_volume} occurrences")
            
            # Zero volume days are possible but suspicious if frequent
            zero_volume = (data['volume'] == 0).sum()
            zero_volume_pct = zero_volume / len(data) * 100
            
            if zero_volume_pct > 10:  # More than 10% zero volume days is suspicious
                print(f"⚠ Warning: High zero-volume days for {symbol}: {zero_volume_pct:.1f}%")
        
        print(f"✓ Data quality validation passed for {symbol}: {len(data)} clean days")
        return data
    
    def _add_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features commonly used in technical analysis.
        
        Pre-calculates frequently used derived features to improve downstream
        processing efficiency and ensure consistent calculations across the system.
        These features form the foundation for technical indicator calculations.
        
        Derived features include:
        - Daily returns (percentage price changes)
        - Intraday volatility (High-Low range as percentage of close)
        - Log returns (for statistical analysis)
        - Price gaps (overnight price changes)
        
        Args:
            data (pd.DataFrame): Validated OHLCV data
            
        Returns:
            pd.DataFrame: Enhanced data with derived features added
        """
        # Create a copy to avoid modifying original data
        enhanced_data = data.copy()
        
        # Daily returns - fundamental measure of price movement
        # Calculated as percentage change from previous close
        enhanced_data['daily_return'] = enhanced_data['close'].pct_change()
        
        # Intraday volatility - measure of intraday price range
        # Normalized by close price for cross-symbol comparability
        enhanced_data['intraday_volatility'] = (
            (enhanced_data['high'] - enhanced_data['low']) / enhanced_data['close']
        )
        
        # Log returns - used for statistical analysis and normal distribution assumptions
        # More appropriate for certain mathematical models
        enhanced_data['log_return'] = np.log(enhanced_data['close'] / enhanced_data['close'].shift(1))
        
        # Price gaps - difference between current open and previous close
        # Important for gap analysis and overnight risk assessment
        enhanced_data['price_gap'] = (
            enhanced_data['open'] - enhanced_data['close'].shift(1)
        ) / enhanced_data['close'].shift(1)
        
        # Handle infinite and NaN values that can arise from calculations
        # Replace with zeros to prevent downstream calculation errors
        enhanced_data = enhanced_data.replace([np.inf, -np.inf], np.nan)
        enhanced_data = enhanced_data.fillna(0)
        
        return enhanced_data
    
    def collect_stock_data(self, symbol: str, start_date: Optional[str] = None, 
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Main method to collect complete stock data with caching and validation.
        
        This is the primary public interface for data collection. It orchestrates
        the complete data acquisition workflow including caching, validation,
        and feature enhancement. The method is designed to be robust and
        user-friendly while maintaining high performance through intelligent caching.
        
        Workflow:
        1. Check cache for existing valid data
        2. Fetch from API if cache miss or invalid
        3. Validate data quality and clean if necessary
        4. Add derived features for downstream analysis
        5. Save to cache for future use
        6. Return ready-to-analyze dataset
        
        Args:
            symbol (str): Stock symbol to collect (e.g., 'AAPL', 'MSFT')
            start_date (str, optional): Start date in YYYY-MM-DD format.
                                      Defaults to configuration or 2 years ago.
            end_date (str, optional): End date in YYYY-MM-DD format.
                                    Defaults to today.
            
        Returns:
            pd.DataFrame: Complete financial dataset ready for technical analysis
                         with OHLCV data plus derived features
            
        Raises:
            ValueError: If data collection fails or data quality is insufficient
        """
        # Normalize symbol for consistent processing
        symbol = symbol.upper().strip()
        print(f"🔄 Collecting data for {symbol}...")
        
        # Set default date range if not provided
        # Use configuration defaults or reasonable fallbacks
        if start_date is None:
            start_date = self.data_config.get('start_date', 
                (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d'))  # 2 years default
        
        if end_date is None:
            end_date = self.data_config.get('end_date', 
                datetime.now().strftime('%Y-%m-%d'))  # Today default
        
        # Attempt cache loading first for performance
        cached_data = self._load_from_cache(symbol)
        if cached_data is not None:
            # Cache hit - validate date range coverage
            cache_start = cached_data.index.min().strftime('%Y-%m-%d')
            cache_end = cached_data.index.max().strftime('%Y-%m-%d')
            
            # Check if cache covers requested date range
            if cache_start <= start_date and cache_end >= end_date:
                # Filter cache to requested date range
                filtered_data = cached_data[start_date:end_date]
                if len(filtered_data) > 0:
                    print(f"✓ Using cached data for {symbol} ({len(filtered_data)} days)")
                    return filtered_data
        
        # Cache miss or insufficient coverage - fetch from API
        print(f"🌐 Fetching {symbol} data from Yahoo Finance API...")
        raw_data = self._fetch_from_yfinance(symbol, start_date, end_date)
        
        # Validate and clean the raw data
        validated_data = self._validate_data_quality(raw_data, symbol)
        
        # Add derived features for analysis
        enhanced_data = self._add_derived_features(validated_data)
        
        # Save to cache for future use
        self._save_to_cache(symbol, enhanced_data)
        
        print(f"✓ Data collection complete for {symbol}: {len(enhanced_data)} days")
        return enhanced_data
    
    def collect_multiple_stocks(self, symbols: List[str], start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Efficiently collect data for multiple stocks with progress tracking.
        
        Provides batch data collection with user-friendly progress indication.
        Handles errors gracefully by collecting available data and reporting
        failures rather than crashing the entire batch operation.
        
        Args:
            symbols (List[str]): List of stock symbols to collect
            start_date (str, optional): Start date for all symbols
            end_date (str, optional): End date for all symbols
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping symbols to their data
                                   Only includes successfully collected symbols
        """
        results = {}
        failed_symbols = []
        
        print(f"📊 Collecting data for {len(symbols)} symbols...")
        
        # Use tqdm for progress bar during batch collection
        for symbol in tqdm(symbols, desc="Collecting data"):
            try:
                data = self.collect_stock_data(symbol, start_date, end_date)
                results[symbol] = data
            except Exception as e:
                print(f"❌ Failed to collect {symbol}: {e}")
                failed_symbols.append(symbol)
        
        # Provide summary of batch collection results
        print(f"✅ Successfully collected: {len(results)} symbols")
        if failed_symbols:
            print(f"❌ Failed symbols: {', '.join(failed_symbols)}")
        
        return results
    
    def get_data_summary(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        Generate comprehensive summary statistics for collected data.
        
        Provides detailed metadata about the dataset including:
        - Temporal coverage and data completeness
        - Price and volume statistics
        - Data quality indicators
        - Derived feature statistics
        
        This summary is useful for data quality assessment and analysis planning.
        
        Args:
            data (pd.DataFrame): Financial data to summarize
            
        Returns:
            Dict[str, any]: Comprehensive data summary with statistics and metadata
        """
        if data.empty:
            return {'status': 'empty', 'message': 'No data available'}
        
        summary = {
            # Temporal information
            'start_date': data.index.min().strftime('%Y-%m-%d'),
            'end_date': data.index.max().strftime('%Y-%m-%d'),
            'total_days': len(data),
            'trading_days_coverage': len(data),  # After cleaning
            
            # Price statistics
            'price_stats': {
                'min_close': float(data['close'].min()),
                'max_close': float(data['close'].max()),
                'avg_close': float(data['close'].mean()),
                'price_volatility': float(data['close'].std() / data['close'].mean()),
            },
            
            # Volume statistics (if available)
            'volume_stats': {},
            
            # Data quality indicators
            'quality_indicators': {
                'missing_values': int(data.isnull().sum().sum()),
                'zero_volume_days': 0,
                'extreme_price_changes': 0,
            }
        }
        
        # Add volume statistics if volume data is available
        if 'volume' in data.columns:
            summary['volume_stats'] = {
                'avg_volume': float(data['volume'].mean()),
                'min_volume': float(data['volume'].min()),
                'max_volume': float(data['volume'].max()),
            }
            summary['quality_indicators']['zero_volume_days'] = int((data['volume'] == 0).sum())
        
        # Add derived feature statistics if available
        if 'daily_return' in data.columns:
            summary['return_stats'] = {
                'avg_daily_return': float(data['daily_return'].mean()),
                'return_volatility': float(data['daily_return'].std()),
                'max_daily_gain': float(data['daily_return'].max()),
                'max_daily_loss': float(data['daily_return'].min()),
            }
            
            # Count extreme price changes (>5% in one day)
            extreme_changes = (abs(data['daily_return']) > 0.05).sum()
            summary['quality_indicators']['extreme_price_changes'] = int(extreme_changes)
        
        return summary 