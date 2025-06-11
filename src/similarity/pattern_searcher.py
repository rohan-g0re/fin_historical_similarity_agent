"""
Pattern Searcher for Historical Market Analysis

Combines similarity calculation with basic filtering to find historical patterns
that match current market conditions. Implements:
- Volatility regime filtering
- Trend direction matching
- RSI zone filtering
- Comprehensive pattern search with ranking
- Performance optimization for large datasets

Main entry point for the financial agent's pattern matching capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

from ..core.config_manager import ConfigManager
from ..core.data_collector import FinancialDataCollector
from ..indicators.technical_indicators import TechnicalIndicators
from .window_creator import WindowCreator
from .similarity_calculator import SimilarityCalculator


class PatternSearcher:
    """
    Complete pattern searching system for historical market analysis.
    
    Integrates all components to provide:
    - Data collection and indicator calculation
    - 7-day window creation
    - Basic filtering by market regime
    - Similarity-based pattern matching
    - Ranked results with analysis
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the pattern searcher.
        
        Args:
            config_manager (ConfigManager, optional): Configuration manager instance.
                                                     Creates new one if None.
        """
        self.config = config_manager if config_manager else ConfigManager()
        self.filtering_config = self.config.config.get('filtering', {})
        
        # Initialize all components
        self.data_collector = FinancialDataCollector(self.config)
        self.indicators = TechnicalIndicators(self.config)
        self.window_creator = WindowCreator(self.config)
        self.similarity_calculator = SimilarityCalculator(self.config)
        
        # Filtering parameters
        self.volatility_tolerance = self.filtering_config.get('volatility_regime_tolerance', 20)
        self.trend_direction_match = self.filtering_config.get('trend_direction_match', True)
        self.rsi_zone_match = self.filtering_config.get('rsi_zone_match', True)
    
    def search_similar_patterns(self, symbol: str, current_date: Optional[str] = None,
                              start_date: Optional[str] = None,
                              apply_filters: bool = True) -> Dict[str, any]:
        """
        Complete pattern search for similar historical periods.
        
        Args:
            symbol (str): Stock symbol to analyze
            current_date (str, optional): Date to use as "current" (defaults to latest)
            start_date (str, optional): Start date for historical search
            apply_filters (bool): Whether to apply basic filters
            
        Returns:
            Dict: Complete search results with similar patterns
        """
        print(f"🔍 Starting pattern search for {symbol}")
        print("=" * 60)
        
        # Prepare data
        data = self.prepare_data(symbol, start_date)
        
        # Get current window
        if current_date:
            current_window = self.window_creator.create_window_features(data, current_date)
            current_window['feature_vector'] = self.window_creator.create_feature_vector(current_window)
        else:
            current_window = self.window_creator.get_current_window(data)
        
        print(f"📊 Current window: {current_window['window_start_date']} to {current_window['window_end_date']}")
        
        # Create historical windows
        historical_windows = self.create_historical_windows(data)
        
        if not historical_windows:
            return {
                'symbol': symbol,
                'current_window': current_window,
                'similar_patterns': [],
                'search_summary': {
                    'total_historical_windows': 0,
                    'filtered_windows': 0,
                    'similar_patterns_found': 0
                }
            }
        
        # Apply basic filters if requested
        if apply_filters:
            filtered_windows = self.apply_basic_filters(current_window, historical_windows)
        else:
            filtered_windows = historical_windows
            print(f"⚠ Skipping basic filters - using all {len(historical_windows)} windows")
        
        # Find similar patterns
        similar_patterns = self.similarity_calculator.find_similar_patterns(
            current_window, filtered_windows
        )
        
        # Enhanced analysis
        enhanced_patterns = self._enhance_pattern_analysis(symbol, data, similar_patterns)
        
        # Create summary
        search_summary = {
            'total_historical_windows': len(historical_windows),
            'filtered_windows': len(filtered_windows),
            'similar_patterns_found': len(similar_patterns),
            'data_period': f"{data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}",
            'current_window_summary': self.window_creator.get_window_summary(current_window)
        }
        
        print("=" * 60)
        print(f"✅ Pattern search complete for {symbol}")
        print(f"📈 Found {len(similar_patterns)} similar patterns")
        
        return {
            'symbol': symbol,
            'current_window': current_window,
            'similar_patterns': enhanced_patterns,
            'search_summary': search_summary
        }
    
    def prepare_data(self, symbol: str, start_date: Optional[str] = None, 
                    end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Prepare complete dataset with indicators for pattern analysis.
        
        Args:
            symbol (str): Stock symbol
            start_date (str, optional): Start date for data collection
            end_date (str, optional): End date for data collection
            
        Returns:
            pd.DataFrame: Complete dataset with price data and technical indicators
        """
        print(f"🔄 Preparing data for {symbol}...")
        
        # Collect raw data
        raw_data = self.data_collector.collect_stock_data(symbol, start_date, end_date)
        
        # Calculate technical indicators
        data_with_indicators = self.indicators.calculate_all_indicators(raw_data)
        
        print(f"✓ Data preparation complete: {len(data_with_indicators)} days with indicators")
        return data_with_indicators
    
    def create_historical_windows(self, data: pd.DataFrame, 
                                gap_days: int = 7) -> List[Dict[str, any]]:
        """
        Create all possible historical windows from dataset.
        
        Args:
            data (pd.DataFrame): Complete dataset with indicators
            gap_days (int): Gap between windows to avoid overlap
            
        Returns:
            List[Dict]: List of historical windows with features
        """
        print(f"📊 Creating historical windows with {gap_days}-day gaps...")
        
        historical_windows = self.window_creator.create_sliding_windows(
            data, min_gap_days=gap_days
        )
        
        return historical_windows
    
    def apply_basic_filters(self, target_window: Dict[str, any], 
                          historical_windows: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        Apply basic filtering to historical windows based on market regime.
        
        Args:
            target_window (Dict): Current/target window for comparison
            historical_windows (List[Dict]): All historical windows
            
        Returns:
            List[Dict]: Filtered historical windows
        """
        if not historical_windows:
            return []
        
        # Get target window characteristics
        target_features = target_window.get('features', {})
        target_zones = self._get_window_zones(target_features)
        
        print(f"📋 Target window characteristics:")
        for zone, value in target_zones.items():
            print(f"   {zone}: {value}")
        
        filtered_windows = []
        
        for window in historical_windows:
            window_features = window.get('features', {})
            window_zones = self._get_window_zones(window_features)
            
            # Apply filters
            if self._passes_filters(target_zones, window_zones):
                filtered_windows.append(window)
        
        filter_ratio = len(filtered_windows) / len(historical_windows) if historical_windows else 0
        print(f"🔍 Basic filtering: {len(filtered_windows)}/{len(historical_windows)} windows passed ({filter_ratio:.1%})")
        
        return filtered_windows
    
    def _get_window_zones(self, features: Dict[str, any]) -> Dict[str, str]:
        """Extract zone classifications from window features."""
        zones = {}
        
        # RSI zone
        if 'rsi_values' in features and features['rsi_values']:
            current_rsi = features['rsi_values'][-1]  # Last day RSI
            if current_rsi > 70:
                zones['rsi_zone'] = 'overbought'
            elif current_rsi < 30:
                zones['rsi_zone'] = 'oversold'
            else:
                zones['rsi_zone'] = 'neutral'
        
        # Bollinger Band zone
        if 'bb_position_values' in features and features['bb_position_values']:
            current_bb = features['bb_position_values'][-1]
            if current_bb > 0.8:
                zones['bb_zone'] = 'upper'
            elif current_bb < 0.2:
                zones['bb_zone'] = 'lower'
            else:
                zones['bb_zone'] = 'middle'
        
        # Volatility regime
        if 'atr_percentile_values' in features and features['atr_percentile_values']:
            current_atr = features['atr_percentile_values'][-1]
            if current_atr > 75:
                zones['volatility_regime'] = 'high'
            elif current_atr < 25:
                zones['volatility_regime'] = 'low'
            else:
                zones['volatility_regime'] = 'medium'
        
        # Trend direction
        if 'macd_signal_values' in features and features['macd_signal_values']:
            macd_values = features['macd_signal_values']
            if len(macd_values) >= 2:
                if macd_values[-1] > macd_values[-2]:
                    zones['trend_direction'] = 'up'
                elif macd_values[-1] < macd_values[-2]:
                    zones['trend_direction'] = 'down'
                else:
                    zones['trend_direction'] = 'sideways'
        
        return zones
    
    def _passes_filters(self, target_zones: Dict[str, str], 
                       window_zones: Dict[str, str]) -> bool:
        """Check if a window passes the basic filters."""
        
        # RSI zone filter
        if self.rsi_zone_match and 'rsi_zone' in target_zones and 'rsi_zone' in window_zones:
            if target_zones['rsi_zone'] != window_zones['rsi_zone']:
                return False
        
        # Trend direction filter
        if self.trend_direction_match and 'trend_direction' in target_zones and 'trend_direction' in window_zones:
            if target_zones['trend_direction'] != window_zones['trend_direction']:
                return False
        
        # Volatility regime filter (with tolerance)
        if 'volatility_regime' in target_zones and 'volatility_regime' in window_zones:
            target_vol = target_zones['volatility_regime']
            window_vol = window_zones['volatility_regime']
            
            # Allow some flexibility in volatility matching
            if target_vol == 'high' and window_vol == 'low':
                return False
            if target_vol == 'low' and window_vol == 'high':
                return False
        
        return True
    
    def _enhance_pattern_analysis(self, symbol: str, data: pd.DataFrame, 
                                patterns: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        Enhance pattern analysis with additional insights.
        
        Args:
            symbol (str): Stock symbol
            data (pd.DataFrame): Complete dataset
            patterns (List[Dict]): Similar patterns found
            
        Returns:
            List[Dict]: Enhanced patterns with additional analysis
        """
        enhanced_patterns = []
        
        for pattern in patterns:
            enhanced_pattern = pattern.copy()
            
            # Add similarity level description
            similarity_score = pattern['similarity_score']
            enhanced_pattern['similarity_level'] = self.similarity_calculator.get_similarity_level(similarity_score)
            
            # Add what happened next analysis
            window_end_idx = pattern.get('window_end_idx')
            if window_end_idx is not None and window_end_idx < len(data) - 20:  # Need at least 20 days after
                next_periods = self._analyze_what_happened_next(data, window_end_idx)
                enhanced_pattern['what_happened_next'] = next_periods
            
            # Add period context
            window_start_date = pattern.get('window_start_date')
            window_end_date = pattern.get('window_end_date')
            if window_start_date and window_end_date:
                enhanced_pattern['period_context'] = self._get_period_context(window_start_date, window_end_date)
            
            enhanced_patterns.append(enhanced_pattern)
        
        return enhanced_patterns
    
    def _analyze_what_happened_next(self, data: pd.DataFrame, window_end_idx: int) -> Dict[str, any]:
        """Analyze price movement after similar pattern occurred."""
        start_price = data['close'].iloc[window_end_idx]
        
        next_periods = {}
        
        # Define periods to analyze
        periods = {
            '1_week': 5,
            '2_weeks': 10,
            '1_month': 20,
            '3_months': 60
        }
        
        for period_name, days in periods.items():
            end_idx = window_end_idx + days
            if end_idx < len(data):
                end_price = data['close'].iloc[end_idx]
                return_pct = ((end_price - start_price) / start_price) * 100
                next_periods[period_name] = {
                    'return_pct': round(return_pct, 2),
                    'end_price': round(end_price, 2),
                    'direction': 'up' if return_pct > 0 else 'down' if return_pct < 0 else 'flat'
                }
        
        return next_periods
    
    def _get_period_context(self, start_date: str, end_date: str) -> Dict[str, str]:
        """Get contextual information about the time period."""
        try:
            if isinstance(start_date, str):
                start_dt = datetime.strptime(start_date.split()[0], '%Y-%m-%d')
            else:
                start_dt = start_date
                
            year = start_dt.year
            month = start_dt.month
            
            # Basic context
            context = {
                'year': str(year),
                'quarter': f"Q{(month-1)//3 + 1}",
                'season': self._get_season(month)
            }
            
            # Add notable periods (simplified)
            if year == 2008 and month >= 9:
                context['market_event'] = 'Financial Crisis'
            elif year == 2020 and month >= 3 and month <= 5:
                context['market_event'] = 'COVID-19 Crash'
            elif year == 2018 and month >= 10:
                context['market_event'] = 'Tech Selloff'
            
            return context
            
        except Exception:
            return {'year': 'Unknown', 'quarter': 'Unknown', 'season': 'Unknown'}
    
    def _get_season(self, month: int) -> str:
        """Get season from month number."""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    def get_pattern_summary(self, search_results: Dict[str, any]) -> Dict[str, any]:
        """
        Generate summary statistics for pattern search results.
        
        Args:
            search_results (Dict): Results from search_similar_patterns()
            
        Returns:
            Dict: Summary statistics and insights
        """
        patterns = search_results.get('similar_patterns', [])
        
        if not patterns:
            return {
                'total_patterns': 0,
                'average_similarity': 0.0,
                'similarity_distribution': {},
                'top_similarity': 0.0,
                'predictions_summary': {}
            }
        
        # Basic statistics
        similarities = [p['similarity_score'] for p in patterns]
        
        # Similarity distribution
        similarity_levels = {}
        for pattern in patterns:
            level = pattern.get('similarity_level', 'Unknown')
            similarity_levels[level] = similarity_levels.get(level, 0) + 1
        
        # Predictions analysis
        predictions_summary = {}
        if patterns and 'what_happened_next' in patterns[0]:
            periods = ['1_week', '2_weeks', '1_month', '3_months']
            for period in periods:
                returns = []
                for pattern in patterns:
                    next_data = pattern.get('what_happened_next', {})
                    if period in next_data:
                        returns.append(next_data[period]['return_pct'])
                
                if returns:
                    predictions_summary[period] = {
                        'avg_return': round(np.mean(returns), 2),
                        'positive_outcomes': sum(1 for r in returns if r > 0),
                        'negative_outcomes': sum(1 for r in returns if r < 0),
                        'success_rate': round((sum(1 for r in returns if r > 0) / len(returns)) * 100, 1)
                    }
        
        return {
            'total_patterns': len(patterns),
            'average_similarity': round(np.mean(similarities), 3),
            'similarity_distribution': similarity_levels,
            'top_similarity': round(max(similarities), 3),
            'predictions_summary': predictions_summary
        } 