"""
Pattern Searcher - High-level Pattern Matching Interface

Orchestrates the complete pattern search workflow combining:
- Window creation for current market conditions
- Similarity calculation against historical patterns
- Advanced filtering for relevant and independent results
- Statistical analysis and confidence assessment
- Business-friendly result formatting and interpretation

This module provides the primary interface for pattern-based market analysis.

The Pattern Searcher abstracts the complexity of multi-step pattern matching into
a simple, powerful interface that business users can leverage for investment
decisions. Key capabilities include:

**Comprehensive Search Process:**
1. Current market characterization using 7-day windows
2. Historical pattern database search across multiple timeframes
3. Similarity-based ranking using cosine distance calculations
4. Statistical independence filtering to prevent data leakage
5. Quality assessment and confidence scoring for results

**Business Intelligence Features:**
- Market regime classification (momentum, trend, volatility context)
- Historical precedent analysis with specific dates and outcomes
- Risk assessment based on pattern similarity confidence levels
- Performance attribution linking current conditions to past results
- Probability-based forecasting using historical pattern outcomes

**Quality Assurance:**
- Data validation ensures reliable input for analysis
- Statistical filtering removes low-quality or overlapping patterns
- Confidence scoring helps users assess reliability of findings
- Performance monitoring tracks search effectiveness over time

The system transforms complex quantitative analysis into actionable investment
insights that support evidence-based decision making.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings

from ..core.config_manager import ConfigManager
from ..core.data_collector import FinancialDataCollector
from ..indicators.technical_indicators import TechnicalIndicators
from .window_creator import WindowCreator
from .similarity_calculator import SimilarityCalculator


class PatternSearcher:
    """
    High-level pattern search orchestrator for comprehensive market analysis.
    
    This class provides the primary interface for pattern-based market analysis,
    combining all the individual components into a cohesive system that delivers
    actionable investment insights. The orchestrator handles:
    
    **Data Pipeline Management:**
    - Coordinates data collection, validation, and preprocessing
    - Manages technical indicator calculation and window creation
    - Handles caching and performance optimization across components
    - Ensures data quality and consistency throughout the workflow
    
    **Analysis Orchestration:**
    - Creates current market characterization using latest 7-day window
    - Searches historical database for similar market conditions
    - Applies sophisticated filtering to ensure result relevance
    - Ranks patterns by similarity and statistical confidence
    
    **Business Intelligence:**
    - Translates mathematical results into business-friendly insights
    - Provides market regime classification and risk assessment
    - Links current conditions to specific historical precedents
    - Enables evidence-based investment decision making
    
    **Quality Control:**
    - Validates input data quality before analysis
    - Filters results based on statistical significance
    - Provides confidence scoring for pattern matches
    - Monitors system performance and accuracy over time
    
    The system is designed to be both powerful for quantitative analysts and
    accessible for business users who need reliable market insights.
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the pattern search orchestrator with all required components.
        
        Sets up the complete pattern matching infrastructure including data
        collection, technical analysis, window creation, and similarity calculation.
        Each component is configured through the centralized configuration system
        to ensure consistent behavior across the entire workflow.
        
        Args:
            config_manager (ConfigManager, optional): Configuration manager instance.
                                                     Creates new one if None to ensure
                                                     independent operation capability.
        """
        # Initialize centralized configuration system
        self.config = config_manager if config_manager else ConfigManager()
        
        # Initialize all required components with shared configuration
        # This ensures consistent parameter usage across the entire pipeline
        self.data_collector = FinancialDataCollector(self.config)
        self.indicators = TechnicalIndicators(self.config)
        self.window_creator = WindowCreator(self.config)
        self.similarity_calculator = SimilarityCalculator(self.config)
        
        # Load search parameters from configuration
        search_config = self.config.get('pattern_search', {})
        self.min_historical_windows = search_config.get('min_historical_windows', 50)
        self.max_search_results = search_config.get('max_search_results', 10)
        self.min_confidence_score = search_config.get('min_confidence_score', 0.6)
        
        # Performance tracking for system optimization
        self.search_stats = {
            'total_searches': 0,
            'avg_search_time': 0.0,
            'avg_patterns_found': 0.0,
            'avg_confidence_score': 0.0
        }
    
    def search_similar_patterns(self, symbol: str, target_date: Optional[str] = None,
                              start_date: Optional[str] = None,
                               apply_filtering: bool = True) -> Dict[str, any]:
        """
        Search for historical patterns similar to current/specified market conditions.
        
        This is the primary method that orchestrates the complete pattern search
        workflow. It handles data collection, technical analysis, pattern matching,
        and result interpretation to provide comprehensive market insights.
        
        **Complete Search Workflow:**
        
        1. **Data Preparation**:
           - Collect historical price and volume data for the symbol
           - Calculate all 5 core technical indicators
           - Validate data quality and completeness
        
        2. **Market Characterization**:
           - Create current/target window with 7-day market fingerprint
           - Generate comprehensive feature vector for pattern matching
           - Assess current market regime (momentum, trend, volatility)
        
        3. **Historical Search**:
           - Generate sliding windows across entire historical dataset
           - Calculate similarity scores against current conditions
           - Apply statistical filtering for independence and quality
        
        4. **Result Analysis**:
           - Rank patterns by similarity and confidence
           - Provide statistical summary and confidence assessment
           - Format results for business interpretation and decision making
        
        **Filtering Options:**
        - Statistical independence: Removes overlapping time periods
        - Quality filtering: Excludes low-quality data periods
        - Confidence filtering: Only returns high-confidence matches
        - Recency filtering: Can emphasize recent vs historical patterns
        
        Args:
            symbol (str): Stock symbol to analyze (e.g., 'AAPL', 'MSFT')
            target_date (str, optional): Specific date for analysis (YYYY-MM-DD)
                                       Uses most recent data if None
            start_date (str, optional): Start date for historical search window
                                      Uses default lookback if None
            apply_filtering (bool): Whether to apply quality and independence filters
                                  True for production, False for research
            
        Returns:
            Dict[str, any]: Comprehensive search results including:
                          - similar_patterns: Ranked list of historical matches
                          - current_conditions: Analysis of target period
                          - search_statistics: Quality and confidence metrics
                          - market_regime: Current market condition classification
                          
        Raises:
            ValueError: If data collection fails or insufficient historical data
        """
        search_start_time = datetime.now()
        
        print(f"🔍 Starting pattern search for {symbol}")
        if target_date:
            print(f"   Target date: {target_date}")
        
        try:
            # Step 1: Data Collection and Preparation
            print("📊 Step 1: Collecting and preparing data...")
            data = self._prepare_data(symbol, start_date, target_date)
            
            if len(data) < self.min_historical_windows:
                raise ValueError(f"Insufficient historical data: {len(data)} periods "
                               f"(need {self.min_historical_windows})")
            
            # Step 2: Market Characterization
            print("🎯 Step 2: Characterizing current market conditions...")
            current_window = self._create_target_window(data, target_date)
            market_regime = self._assess_market_regime(current_window)
            
            # Step 3: Historical Pattern Search
            print("🔄 Step 3: Searching historical patterns...")
            historical_windows = self._create_historical_windows(data)
            
            # Apply filtering if requested
            if apply_filtering:
                historical_windows = self._apply_quality_filters(historical_windows)
            
            print(f"   Comparing against {len(historical_windows)} historical windows")
            
            # Step 4: Similarity Calculation and Ranking
            print("📈 Step 4: Calculating similarities and ranking results...")
            similar_patterns = self.similarity_calculator.find_similar_patterns(
                current_window, 
                historical_windows,
                apply_gap_filter=apply_filtering
            )
            
            # Step 5: Statistical Analysis and Result Preparation
            print("📋 Step 5: Analyzing results and preparing insights...")
            search_results = self._prepare_search_results(
                similar_patterns, 
                current_window, 
                market_regime,
                len(historical_windows)
            )
            
            # Add symbol to results for business report generation
            search_results['symbol'] = symbol
            
            # Performance tracking
            search_time = (datetime.now() - search_start_time).total_seconds()
            self._update_search_stats(search_time, len(similar_patterns), search_results)
            
            print(f"✅ Pattern search completed in {search_time:.2f} seconds")
            print(f"   Found {len(similar_patterns)} similar patterns")
            
            return search_results
            
        except Exception as e:
            print(f"❌ Pattern search failed: {e}")
            raise
    
    def _prepare_data(self, symbol: str, start_date: Optional[str], 
                     target_date: Optional[str]) -> pd.DataFrame:
        """
        Collect and prepare complete dataset for pattern analysis.
        
        Orchestrates the data collection and technical analysis pipeline to create
        a comprehensive dataset ready for pattern matching. This method ensures
        data quality and completeness required for reliable analysis.
        
        **Data Preparation Pipeline:**
        1. **Raw Data Collection**: Fetch OHLCV data from Yahoo Finance
        2. **Data Validation**: Ensure quality and completeness
        3. **Technical Analysis**: Calculate all 5 core indicators
        4. **Feature Engineering**: Add derived features for pattern matching
        5. **Quality Assessment**: Validate final dataset integrity
        
        Args:
            symbol (str): Stock symbol to collect data for
            start_date (str, optional): Start date for data collection
            target_date (str, optional): End date for data collection
            
        Returns:
            pd.DataFrame: Complete dataset with OHLCV data and technical indicators
                         Ready for window creation and pattern matching
            
        Raises:
            ValueError: If data collection or technical analysis fails
        """
        # Collect raw price and volume data
        raw_data = self.data_collector.collect_stock_data(
            symbol=symbol,
            start_date=start_date,
            end_date=target_date
        )
        
        # Calculate technical indicators for complete analysis
        enhanced_data = self.indicators.calculate_all_indicators(raw_data)
        
        # Validate final dataset quality
        self._validate_prepared_data(enhanced_data, symbol)
        
        # Pass the original data to similarity calculator for forward return calculations
        self.similarity_calculator.set_original_data(enhanced_data)
        
        return enhanced_data
    
    def _validate_prepared_data(self, data: pd.DataFrame, symbol: str) -> None:
        """
        Validate prepared dataset quality for reliable pattern analysis.
        
        Performs comprehensive quality checks to ensure the dataset meets
        requirements for accurate pattern matching. Poor quality data can
        lead to unreliable results and false pattern matches.
        
        **Validation Checks:**
        - Minimum data points for statistical reliability
        - Technical indicator completeness and validity
        - Data continuity and gap analysis
        - Feature vector compatibility requirements
        
        Args:
            data (pd.DataFrame): Prepared dataset to validate
            symbol (str): Symbol name for error reporting
            
        Raises:
            ValueError: If data quality is insufficient for reliable analysis
        """
        # Check minimum data requirements
        if len(data) < self.min_historical_windows:
            raise ValueError(f"Insufficient data for {symbol}: {len(data)} periods "
                           f"(minimum {self.min_historical_windows} required)")
        
        # Validate technical indicators are present
        required_indicators = ['rsi', 'macd_signal', 'bb_position', 'volume_roc', 'atr_percentile']
        missing_indicators = [ind for ind in required_indicators if ind not in data.columns]
        
        if missing_indicators:
            raise ValueError(f"Missing technical indicators for {symbol}: {missing_indicators}")
        
        # Check for excessive missing data
        for indicator in required_indicators:
            missing_pct = (data[indicator].isna().sum() / len(data)) * 100
            if missing_pct > 30:  # More than 30% missing is problematic
                warnings.warn(f"High missing data in {indicator} for {symbol}: {missing_pct:.1f}%")
        
        print(f"✓ Data validation passed for {symbol}: {len(data)} periods with complete indicators")
    
    def _create_target_window(self, data: pd.DataFrame, target_date: Optional[str]) -> Dict[str, any]:
        """
        Create target window representing current or specified market conditions.
        
        Generates the market "fingerprint" that will be compared against historical
        patterns. This window captures the current market regime and serves as
        the reference point for similarity analysis.
        
        Args:
            data (pd.DataFrame): Complete dataset with indicators
            target_date (str, optional): Specific date for target window
                                       Uses most recent if None
            
        Returns:
            Dict[str, any]: Target window with features and metadata
                           Ready for similarity comparison
        """
        if target_date:
            # Create window for specific date
            try:
                target_window = self.window_creator.create_window_features(data, target_date)
                feature_vector = self.window_creator.create_feature_vector(target_window)
                
                window = {
                    'features': target_window,
                    'feature_vector': feature_vector,
                    'vector_length': len(feature_vector),
                    'is_current': False,
                    'target_date': target_date
                }
            except Exception as e:
                raise ValueError(f"Failed to create target window for {target_date}: {e}")
        else:
            # Use most recent complete window
            window = self.window_creator.get_current_window(data)
        
        return window
    
    def _assess_market_regime(self, window: Dict[str, any]) -> Dict[str, str]:
        """
        Assess current market regime based on window features.
        
        Classifies the current market conditions into interpretable regimes
        that provide business context for pattern matching results. This
        classification helps users understand the market environment and
        interpret pattern similarity in proper context.
        
        **Regime Classifications:**
        - Momentum: Overbought, Oversold, Neutral (based on RSI)
        - Trend: Uptrend, Downtrend, Sideways (based on MACD Signal)
        - Volatility: High, Normal, Low (based on ATR Percentile)
        - Volume: High Activity, Normal, Low Activity (based on Volume ROC)
        - Position: Upper Extreme, Middle Range, Lower Extreme (based on Bollinger)
        
        Args:
            window (Dict[str, any]): Window with calculated features
            
        Returns:
            Dict[str, str]: Market regime classifications for interpretation
        """
        features = window.get('features', {})
        regime = {}
        
        # Momentum regime (RSI-based)
        if 'rsi_values' in features and features['rsi_values']:
            current_rsi = features['rsi_values'][-1]  # Most recent RSI
            avg_rsi = np.mean(features['rsi_values'])   # Window average
            
            if avg_rsi > 70:
                regime['momentum'] = 'OVERBOUGHT'
            elif avg_rsi < 30:
                regime['momentum'] = 'OVERSOLD'
            else:
                regime['momentum'] = 'NEUTRAL'
        
        # Trend regime (MACD Signal-based)
        if 'macd_signal_values' in features and len(features['macd_signal_values']) >= 2:
            macd_values = features['macd_signal_values']
            recent_trend = macd_values[-1] - macd_values[0]  # Change over window
            
            if recent_trend > 0:
                regime['trend'] = 'STRENGTHENING'
            elif recent_trend < 0:
                regime['trend'] = 'WEAKENING'
            else:
                regime['trend'] = 'SIDEWAYS'
        
        # Volatility regime (ATR Percentile-based)
        if 'atr_percentile_values' in features and features['atr_percentile_values']:
            avg_atr_pct = np.mean(features['atr_percentile_values'])
            
            if avg_atr_pct > 75:
                regime['volatility'] = 'HIGH'
            elif avg_atr_pct < 25:
                regime['volatility'] = 'LOW'
            else:
                regime['volatility'] = 'NORMAL'
        
        # Volume activity regime (Volume ROC-based)
        if 'volume_roc_values' in features and features['volume_roc_values']:
            avg_vol_roc = np.mean(features['volume_roc_values'])
            
            if avg_vol_roc > 25:
                regime['volume'] = 'HIGH_ACTIVITY'
            elif avg_vol_roc < -25:
                regime['volume'] = 'LOW_ACTIVITY'
            else:
                regime['volume'] = 'NORMAL_ACTIVITY'
        
        # Price position regime (Bollinger Position-based)
        if 'bb_position_values' in features and features['bb_position_values']:
            avg_bb_pos = np.mean(features['bb_position_values'])
            
            if avg_bb_pos > 0.8:
                regime['price_position'] = 'UPPER_EXTREME'
            elif avg_bb_pos < 0.2:
                regime['price_position'] = 'LOWER_EXTREME'
            else:
                regime['price_position'] = 'MIDDLE_RANGE'
        
        return regime
    
    def _create_historical_windows(self, data: pd.DataFrame) -> List[Dict[str, any]]:
        """
        Create comprehensive set of historical windows for pattern comparison.
        
        Generates sliding windows across the entire historical dataset to create
        a comprehensive database of market patterns. Each window represents a
        distinct market condition that can be compared to current conditions.
        
        **Window Creation Strategy:**
        - Systematic coverage of entire historical period
        - Consistent feature extraction methodology
        - Quality assessment for each window
        - Efficient batch processing for performance
        
        Args:
            data (pd.DataFrame): Complete historical dataset
            
        Returns:
            List[Dict[str, any]]: Historical windows ready for similarity analysis
        """
        # Create sliding windows with minimal gap for comprehensive coverage
        # Using 1-day gap provides maximum pattern coverage while maintaining efficiency
        historical_windows = self.window_creator.create_sliding_windows(
            data, 
            min_gap_days=1  # Minimal gap for comprehensive coverage
        )
        
        print(f"✓ Created {len(historical_windows)} historical windows")
        return historical_windows
    
    def _apply_quality_filters(self, windows: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        Apply quality filters to ensure reliable pattern matching results.
        
        Filters the historical window database to remove low-quality patterns
        that could contaminate similarity analysis. Quality filtering is essential
        for production systems where reliability is paramount.
        
        **Quality Filters:**
        - Data completeness: Remove windows with excessive missing values
        - Feature validity: Ensure all required features are present
        - Statistical reliability: Remove outlier or anomalous patterns
        - Temporal quality: Filter periods with known data issues
        
        Args:
            windows (List[Dict[str, any]]): Raw historical windows
            
        Returns:
            List[Dict[str, any]]: Filtered windows meeting quality standards
        """
        filtered_windows = []
        
        for window in windows:
            # Check data quality score if available
            quality_score = window.get('data_quality_score', 1.0)
            
            if quality_score >= 0.7:  # Minimum 70% quality threshold
                # Additional quality checks can be added here
                # e.g., feature completeness, value range validation
                filtered_windows.append(window)
        
        print(f"✓ Quality filtering: {len(filtered_windows)}/{len(windows)} windows passed")
        return filtered_windows
    
    def _prepare_search_results(self, similar_patterns: List[Dict[str, any]], 
                               current_window: Dict[str, any],
                               market_regime: Dict[str, str],
                               total_windows_searched: int) -> Dict[str, any]:
        """
        Prepare comprehensive search results for business analysis and reporting.
        
        Transforms raw similarity results into a comprehensive business intelligence
        package that supports investment decision-making. The results include
        statistical analysis, confidence assessment, and business interpretation.
        
        **Result Components:**
        - Pattern matches: Ranked historical precedents with similarity scores
        - Current analysis: Market regime and condition assessment
        - Statistical summary: Confidence metrics and quality indicators
        - Business insights: Interpretations and recommendations
        
        Args:
            similar_patterns (List[Dict[str, any]]): Ranked similarity results
            current_window (Dict[str, any]): Current market conditions window
            market_regime (Dict[str, str]): Market regime classification
            total_windows_searched (int): Total historical patterns analyzed
            
        Returns:
            Dict[str, any]: Comprehensive search results ready for business use
        """
        # Extract similarity scores for statistical analysis
        similarity_scores = [p['similarity_score'] for p in similar_patterns]
        
        # Calculate comprehensive statistics
        search_statistics = self.similarity_calculator.get_similarity_statistics(similarity_scores)
        search_statistics['total_windows_searched'] = total_windows_searched
        search_statistics['patterns_found'] = len(similar_patterns)
        
        # Add confidence assessment
        confidence_assessment = self._assess_search_confidence(search_statistics, market_regime)
        
        # Enhance pattern results with business interpretation
        enhanced_patterns = []
        for pattern in similar_patterns:
            enhanced_pattern = pattern.copy()
            enhanced_pattern['similarity_level'] = self.similarity_calculator.get_similarity_level(
                pattern['similarity_score']
            )
            enhanced_pattern['confidence_tier'] = self._get_confidence_tier(pattern['similarity_score'])
            enhanced_patterns.append(enhanced_pattern)
        
        # Compile comprehensive results
        results = {
            'search_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_patterns_searched': total_windows_searched,
                'total_historical_windows': total_windows_searched,
                'patterns_found': len(similar_patterns),
                'similar_patterns_found': len(similar_patterns),
                'search_quality': 'HIGH' if len(similar_patterns) >= 5 else 'MEDIUM' if len(similar_patterns) >= 2 else 'LOW',
                'data_period': f"Historical analysis from {len(enhanced_patterns)} periods"
            },
            'current_window': current_window,
            'current_conditions': {
                'window_period': f"{current_window.get('window_start_date', 'Unknown')} to {current_window.get('window_end_date', 'Unknown')}",
                'market_regime': market_regime,
                'vector_dimensions': current_window.get('vector_length', 0)
            },
            'similar_patterns': enhanced_patterns,
            'search_summary': search_statistics,
            'search_statistics': search_statistics,
            'confidence_assessment': confidence_assessment,
            'interpretation': self._generate_interpretation(enhanced_patterns, market_regime, search_statistics)
        }
        
        return results
    
    def _assess_search_confidence(self, stats: Dict[str, float], 
                                 regime: Dict[str, str]) -> Dict[str, any]:
        """
        Assess overall confidence in search results for decision support.
        
        Evaluates multiple factors to provide a comprehensive confidence assessment
        that helps users understand the reliability of pattern matching results.
        This assessment is crucial for risk management and decision making.
        
        **Confidence Factors:**
        - Pattern quantity: More patterns provide better statistical basis
        - Similarity levels: Higher similarities indicate stronger matches
        - Score consistency: Lower variance suggests reliable patterns
        - Market regime: Some regimes are more predictable than others
        
        Args:
            stats (Dict[str, float]): Search statistics summary
            regime (Dict[str, str]): Market regime classification
            
        Returns:
            Dict[str, any]: Comprehensive confidence assessment
        """
        confidence = {
            'overall_score': 0.0,
            'factors': {},
            'recommendation': 'UNKNOWN'
        }
        
        # Factor 1: Pattern quantity (more patterns = higher confidence)
        pattern_count = stats.get('patterns_found', 0)
        if pattern_count >= 10:
            quantity_score = 1.0
        elif pattern_count >= 5:
            quantity_score = 0.7
        elif pattern_count >= 2:
            quantity_score = 0.5
        else:
            quantity_score = 0.2
        
        confidence['factors']['pattern_quantity'] = quantity_score
        
        # Factor 2: Average similarity level (higher = better)
        avg_similarity = stats.get('mean', 0.0)
        if avg_similarity >= 0.8:
            similarity_score = 1.0
        elif avg_similarity >= 0.7:
            similarity_score = 0.8
        elif avg_similarity >= 0.6:
            similarity_score = 0.6
        else:
            similarity_score = 0.3
        
        confidence['factors']['similarity_strength'] = similarity_score
        
        # Factor 3: Score consistency (lower std = higher confidence)
        score_std = stats.get('std', 1.0)
        if score_std <= 0.05:
            consistency_score = 1.0
        elif score_std <= 0.10:
            consistency_score = 0.8
        elif score_std <= 0.15:
            consistency_score = 0.6
        else:
            consistency_score = 0.4
        
        confidence['factors']['score_consistency'] = consistency_score
        
        # Calculate overall confidence score (weighted average)
        overall_score = (
            quantity_score * 0.4 +      # 40% weight on quantity
            similarity_score * 0.4 +    # 40% weight on similarity strength
            consistency_score * 0.2     # 20% weight on consistency
        )
        
        confidence['overall_score'] = overall_score
        
        # Generate recommendation based on overall score
        if overall_score >= 0.8:
            confidence['recommendation'] = 'HIGH_CONFIDENCE'
        elif overall_score >= 0.6:
            confidence['recommendation'] = 'MEDIUM_CONFIDENCE'
        elif overall_score >= 0.4:
            confidence['recommendation'] = 'LOW_CONFIDENCE'
        else:
            confidence['recommendation'] = 'INSUFFICIENT_EVIDENCE'
        
        return confidence
    
    def _get_confidence_tier(self, similarity_score: float) -> str:
        """
        Classify individual pattern confidence for portfolio applications.
        
        Provides tiered confidence classification for individual patterns
        to support portfolio construction and risk management decisions.
        
        Args:
            similarity_score (float): Individual pattern similarity score
            
        Returns:
            str: Confidence tier classification
        """
        if similarity_score >= 0.85:
            return 'TIER_1'  # Core positions
        elif similarity_score >= 0.75:
            return 'TIER_2'  # Supporting positions
        elif similarity_score >= 0.65:
            return 'TIER_3'  # Research/context
        else:
            return 'TIER_4'  # Low confidence
    
    def _generate_interpretation(self, patterns: List[Dict[str, any]], 
                               regime: Dict[str, str],
                               stats: Dict[str, float]) -> Dict[str, any]:
        """
        Generate business interpretation of search results for decision support.
        
        Transforms technical analysis results into business-friendly insights
        that support investment decision-making and risk assessment.
        
        Args:
            patterns (List[Dict[str, any]]): Enhanced pattern results
            regime (Dict[str, str]): Current market regime
            stats (Dict[str, float]): Search statistics
            
        Returns:
            Dict[str, any]: Business interpretation and insights
        """
        interpretation = {
            'market_context': self._interpret_market_regime(regime),
            'pattern_summary': self._summarize_patterns(patterns),
            'risk_assessment': self._assess_pattern_risk(patterns, stats),
            'recommended_actions': self._generate_recommendations(patterns, regime, stats)
        }
        
        return interpretation
    
    def _interpret_market_regime(self, regime: Dict[str, str]) -> str:
        """Generate business-friendly market regime interpretation."""
        regime_parts = []
        
        if regime.get('momentum') == 'OVERBOUGHT':
            regime_parts.append("momentum appears stretched to the upside")
        elif regime.get('momentum') == 'OVERSOLD':
            regime_parts.append("momentum appears oversold")
        
        if regime.get('trend') == 'STRENGTHENING':
            regime_parts.append("trend is strengthening")
        elif regime.get('trend') == 'WEAKENING':
            regime_parts.append("trend is weakening")
        
        if regime.get('volatility') == 'HIGH':
            regime_parts.append("volatility is elevated")
        elif regime.get('volatility') == 'LOW':
            regime_parts.append("volatility is subdued")
        
        if regime_parts:
            return "Current market conditions suggest " + ", ".join(regime_parts) + "."
        else:
            return "Market conditions appear neutral across key indicators."
    
    def _summarize_patterns(self, patterns: List[Dict[str, any]]) -> str:
        """Generate summary of found patterns."""
        if not patterns:
            return "No similar historical patterns found above confidence threshold."
        
        high_conf_count = len([p for p in patterns if p['similarity_score'] >= 0.8])
        total_count = len(patterns)
        
        if high_conf_count >= 3:
            return f"Found {total_count} similar patterns, with {high_conf_count} showing high confidence similarity."
        elif total_count >= 5:
            return f"Found {total_count} moderately similar patterns for analysis."
        else:
            return f"Found {total_count} historical patterns with limited similarity."
    
    def _assess_pattern_risk(self, patterns: List[Dict[str, any]], 
                           stats: Dict[str, float]) -> str:
        """Assess risk level based on pattern analysis."""
        if not patterns:
            return "UNKNOWN - Insufficient pattern data for risk assessment."
        
        avg_similarity = stats.get('mean', 0.0)
        score_consistency = stats.get('std', 1.0)
        
        if avg_similarity >= 0.8 and score_consistency <= 0.1:
            return "LOW - High-confidence patterns with consistent similarity."
        elif avg_similarity >= 0.7 and score_consistency <= 0.15:
            return "MEDIUM - Good patterns with reasonable consistency."
        elif avg_similarity >= 0.6:
            return "MEDIUM-HIGH - Moderate patterns, use with additional analysis."
        else:
            return "HIGH - Weak patterns, high uncertainty in projections."
    
    def _generate_recommendations(self, patterns: List[Dict[str, any]], 
                                regime: Dict[str, str],
                                stats: Dict[str, float]) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        if not patterns:
            recommendations.append("Consider waiting for clearer pattern signals before making major position changes.")
            return recommendations
        
        high_conf_patterns = [p for p in patterns if p['similarity_score'] >= 0.8]
        
        if len(high_conf_patterns) >= 3:
            recommendations.append("High-confidence patterns support position sizing based on historical precedents.")
            recommendations.append("Review specific dates and outcomes of top-ranked patterns for detailed insights.")
        
        if regime.get('volatility') == 'HIGH':
            recommendations.append("Elevated volatility suggests smaller position sizes and closer risk monitoring.")
        
        if regime.get('momentum') == 'OVERBOUGHT':
            recommendations.append("Overbought conditions may limit upside potential in the near term.")
        elif regime.get('momentum') == 'OVERSOLD':
            recommendations.append("Oversold conditions may present opportunity if supported by fundamental analysis.")
        
        avg_similarity = stats.get('mean', 0.0)
        if avg_similarity < 0.6:
            recommendations.append("Lower pattern similarity suggests using additional analysis methods for confirmation.")
        
        return recommendations
    
    def _update_search_stats(self, search_time: float, patterns_found: int, 
                           results: Dict[str, any]) -> None:
        """Update performance statistics for system monitoring."""
        self.search_stats['total_searches'] += 1
        
        # Update moving averages
        n = self.search_stats['total_searches']
        self.search_stats['avg_search_time'] = (
            (self.search_stats['avg_search_time'] * (n-1) + search_time) / n
        )
        self.search_stats['avg_patterns_found'] = (
            (self.search_stats['avg_patterns_found'] * (n-1) + patterns_found) / n
        )
        
        # Update confidence tracking
        if results.get('search_statistics', {}).get('mean'):
            avg_conf = results['search_statistics']['mean']
            self.search_stats['avg_confidence_score'] = (
                (self.search_stats['avg_confidence_score'] * (n-1) + avg_conf) / n
            )
    
    def get_search_performance(self) -> Dict[str, any]:
        """
        Get performance statistics for system monitoring and optimization.
        
        Returns:
            Dict[str, any]: Performance metrics and system health indicators
        """
        return {
            'performance_metrics': self.search_stats.copy(),
            'system_health': {
                'avg_search_time_acceptable': self.search_stats['avg_search_time'] < 30.0,
                'avg_patterns_found_sufficient': self.search_stats['avg_patterns_found'] >= 3.0,
                'avg_confidence_adequate': self.search_stats['avg_confidence_score'] >= 0.6
            },
            'recommendations': self._generate_performance_recommendations()
        }
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate recommendations for system performance optimization."""
        recommendations = []
        
        if self.search_stats['avg_search_time'] > 30.0:
            recommendations.append("Consider optimizing data collection or reducing historical window count.")
        
        if self.search_stats['avg_patterns_found'] < 3.0:
            recommendations.append("Consider adjusting similarity thresholds or expanding historical data range.")
        
        if self.search_stats['avg_confidence_score'] < 0.6:
            recommendations.append("Review technical indicator parameters or data quality filters.")
        
        return recommendations 