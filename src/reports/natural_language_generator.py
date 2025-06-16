#!/usr/bin/env python3
"""
Natural Language Report Generator Module

This module transforms technical financial analysis data into comprehensive, business-friendly
natural language reports. It serves as the critical bridge between complex quantitative analysis
and actionable investment insights.

Core Functionality:
- Converts similarity analysis results into narrative format
- Generates executive summaries with key investment insights
- Creates risk assessments with specific mitigation strategies
- Provides historical context and pattern analysis
- Generates business-friendly recommendations and outlook

Report Structure:
1. Executive Summary - High-level findings and investment thesis
2. Current Market Analysis - Technical indicator interpretation
3. Historical Pattern Analysis - Seasonal and decade-based insights
4. Risk Assessment - Comprehensive risk evaluation and mitigation
5. Future Outlook - Multi-timeframe projections based on historical patterns
6. Detailed Comparisons - Individual historical pattern analysis
7. Technical Summary - Raw technical data and methodology

Business Intelligence Features:
- Confidence scoring based on pattern similarity
- Risk categorization with position sizing recommendations
- Seasonal and temporal pattern recognition
- Market regime classification and interpretation
- Historical success rate analysis across multiple timeframes
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import statistics


class NaturalLanguageReportGenerator:
    """
    Core Natural Language Report Generation Engine
    
    This class orchestrates the transformation of technical analysis data into 
    comprehensive business-friendly reports. It combines quantitative analysis
    with qualitative interpretation to provide actionable investment insights.
    
    Architecture:
    - Modular report generation with distinct sections
    - Risk-based categorization and assessment framework
    - Statistical analysis of historical pattern outcomes
    - Confidence scoring based on similarity metrics
    - Business-friendly language generation with specific recommendations
    
    Report Generation Process:
    1. Data validation and preprocessing
    2. Statistical analysis of historical patterns
    3. Risk assessment and confidence scoring
    4. Narrative generation with business context
    5. Recommendation synthesis and formatting
    """
    
    def __init__(self):
        """
        Initialize the report generator with predefined risk and confidence frameworks.
        
        Sets up:
        - Risk level thresholds for categorizing downside probability
        - Confidence level mappings based on similarity scores
        - Business-friendly descriptions for quantitative measures
        """
        # Risk assessment framework based on downside probability thresholds
        # Used to categorize investment risk levels with appropriate descriptions
        self.risk_levels = {
            'low': {'threshold': 20, 'description': 'low risk'},
            'moderate': {'threshold': 50, 'description': 'moderate risk'},
            'high': {'threshold': 80, 'description': 'high risk'},
            'very_high': {'threshold': 100, 'description': 'very high risk'}
        }
        
        # Confidence level framework based on similarity score thresholds
        # Maps quantitative similarity to qualitative confidence descriptions
        self.confidence_levels = {
            'very_high': {'threshold': 0.95, 'description': 'very high confidence'},
            'high': {'threshold': 0.90, 'description': 'high confidence'},
            'moderate': {'threshold': 0.80, 'description': 'moderate confidence'},
            'low': {'threshold': 0.70, 'description': 'low confidence'},
            'very_low': {'threshold': 0.60, 'description': 'very low confidence'}
        }

    def generate_full_report(self, analysis_data: Dict[str, Any]) -> str:
        """
        Generate Complete Business-Friendly Investment Report
        
        This is the main orchestration method that coordinates the generation of a 
        comprehensive natural language report from technical analysis data. It follows
        a structured approach to present complex quantitative findings in an accessible,
        actionable format for business decision-making.
        
        Report Architecture:
        - Executive Summary: High-level investment thesis and key findings
        - Current Market Analysis: Technical indicator interpretation and market conditions
        - Historical Pattern Analysis: Seasonal patterns and historical context
        - Risk Assessment: Comprehensive risk evaluation with mitigation strategies
        - Future Outlook: Multi-timeframe projections based on historical precedents
        - Detailed Comparisons: Individual analysis of most similar historical periods
        - Technical Summary: Methodology overview and raw technical data
        
        Args:
            analysis_data: Complete analysis results from pattern_searcher including:
                - symbol: Stock symbol analyzed
                - current_window: Current market data and technical indicators
                - similar_patterns: List of historically similar periods with outcomes
                - search_summary: Metadata about the analysis process
                
        Returns:
            Formatted natural language report ready for business presentation
            
        Business Value:
        - Transforms complex quantitative analysis into actionable insights
        - Provides specific risk assessment and position sizing recommendations
        - Offers historical context for informed decision-making
        - Delivers clear investment thesis with supporting evidence
        """
        report_sections = []
        
        # Header Section: Basic identification and analysis period
        # Sets professional context and provides key metadata
        report_sections.append(self._generate_header(analysis_data))
        
        # Executive Summary: Critical for decision-makers who need key insights quickly
        # Synthesizes main findings into investment thesis and recommendations
        report_sections.append(self._generate_executive_summary(analysis_data))
        
        # Current Market Analysis: Interprets technical indicators in business language
        # Translates RSI, MACD, volatility metrics into market condition assessment
        report_sections.append(self._generate_current_analysis(analysis_data))
        
        # Historical Pattern Analysis: Provides evidence base for projections
        # Analyzes seasonal patterns, decade trends, and pattern reliability
        report_sections.append(self._generate_pattern_analysis(analysis_data))
        
        # Risk Assessment: Critical for position sizing and risk management
        # Quantifies downside probability and provides specific mitigation strategies
        report_sections.append(self._generate_risk_assessment(analysis_data))
        
        # Future Outlook: Multi-timeframe projections based on historical evidence
        # Provides specific probability-based forecasts with supporting rationale
        report_sections.append(self._generate_outlook(analysis_data))
        
        # Detailed Historical Comparisons: Deep dive into most similar periods
        # Offers specific historical precedents with actual outcomes
        report_sections.append(self._generate_detailed_comparisons(analysis_data))
        
        # Technical Summary: Methodology transparency and raw data
        # Provides technical context and validates analytical approach
        report_sections.append(self._generate_technical_summary(analysis_data))
        
        # Footer: Professional disclaimers and metadata
        # Ensures proper risk disclosure and report attribution
        report_sections.append(self._generate_footer())
        
        # Combine all sections with proper spacing for readability
        return "\n\n".join(report_sections)

    def _generate_header(self, data: Dict[str, Any]) -> str:
        """
        Generate Professional Report Header
        
        Creates the formal header section that establishes report context and 
        provides essential metadata for the analysis. This section serves as 
        the professional identifier and sets expectations for the analysis type.
        
        Args:
            data: Analysis data containing symbol and current window information
            
        Returns:
            Formatted header string with symbol, dates, and analysis type
            
        Business Purpose:
        - Provides clear identification of the analyzed security
        - Documents the specific time period under analysis
        - Establishes the analytical methodology being employed
        - Creates professional presentation format for stakeholders
        """
        # Extract core identification data with safe defaults
        symbol = data.get('symbol', 'Unknown')
        current_window = data.get('current_window', {})
        start_date = current_window.get('window_start_date', 'Unknown')
        end_date = current_window.get('window_end_date', 'Unknown')
        
        # Format professional header with clear visual separation
        return f"""
═══════════════════════════════════════════════════════════════════════════════
                        FINANCIAL PATTERN ANALYSIS REPORT
                                    {symbol}
═══════════════════════════════════════════════════════════════════════════════

Report Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
Analysis Period: {self._format_date(start_date)} to {self._format_date(end_date)}
Analysis Type: Historical Pattern Matching & Similarity Analysis
        """

    def _generate_executive_summary(self, data: Dict[str, Any]) -> str:
        """
        Generate Executive Summary - The Most Critical Report Section
        
        This method creates the executive summary that synthesizes complex analysis
        into actionable investment insights. It's designed for senior decision-makers
        who need rapid understanding of key findings and investment implications.
        
        Process:
        1. Evaluates pattern availability and quality
        2. Analyzes outcomes from most similar historical periods
        3. Synthesizes risk assessment and confidence levels
        4. Generates clear investment thesis and recommendations
        
        Args:
            data: Complete analysis data with patterns and search results
            
        Returns:
            Executive summary section with key findings and investment thesis
            
        Business Critical Elements:
        - Investment thesis based on historical evidence
        - Risk level assessment with quantitative backing
        - Success rate analysis from similar historical periods
        - Confidence assessment based on pattern similarity
        - Clear actionable recommendations for decision-makers
        """
        # Extract core data for executive analysis
        symbol = data.get('symbol', 'Unknown')
        similar_patterns = data.get('similar_patterns', [])
        search_summary = data.get('search_summary', {})
        
        # Handle case where no patterns are found - critical risk disclosure
        if not similar_patterns:
            return f"""
📊 EXECUTIVE SUMMARY
═══════════════════════════════════════════════════════════════════════════════

{symbol} analysis comple~ted but no significant historical patterns were found that 
match the current market conditions. This suggests the current situation may be 
unique or that insufficient historical data is available for comparison.

Recommendation: Exercise caution and consider additional analysis methods.
            """
        
        # Analyze the highest-quality pattern for confidence assessment
        top_pattern = similar_patterns[0]
        similarity_score = top_pattern.get('similarity_score', 0)
        confidence_desc = self._get_confidence_description(similarity_score)
        
        # Perform comprehensive outcome analysis on top patterns
        # Focus on top 3 patterns for most reliable statistical inference
        outcomes = self._analyze_pattern_outcomes(similar_patterns[:3])
        
        # Extract metadata for context and scope
        total_patterns = len(similar_patterns)
        data_period = search_summary.get('data_period', 'Unknown period')
        
        # Generate comprehensive executive summary with key business insights
        return f"""
📊 EXECUTIVE SUMMARY
═══════════════════════════════════════════════════════════════════════════════

Our analysis of {symbol} has identified {total_patterns} historical periods with 
similar market patterns from {data_period}. The most similar pattern shows 
{confidence_desc} (similarity score: {similarity_score:.1%}).

Key Findings:
• Historical patterns suggest {outcomes['primary_trend']} over the next month
• Risk level is assessed as {outcomes['risk_level']} based on historical volatility
• {outcomes['success_rate']:.0f}% of similar periods showed positive outcomes in the short term
• Market conditions are {outcomes['market_regime']} compared to historical norms

Investment Thesis: {outcomes['investment_thesis']}
        """

    def _generate_current_analysis(self, data: Dict[str, Any]) -> str:
        """
        Generate Current Market Conditions Analysis
        
        Translates technical indicators into business-friendly market condition assessment.
        This section provides immediate context for the current trading environment and
        helps stakeholders understand the backdrop against which historical patterns
        are being compared.
        
        Technical Indicators Analyzed:
        - RSI (Relative Strength Index): Market momentum and overbought/oversold conditions
        - ATR Percentile: Volatility environment and risk assessment
        - MACD Trend: Directional bias and trend strength
        - Volume ROC: Market participation and conviction levels
        
        Args:
            data: Analysis data containing current window with technical features
            
        Returns:
            Current market analysis section with business interpretations
            
        Business Value:
        - Provides immediate market context for decision-making
        - Translates complex technical indicators into actionable insights
        - Establishes baseline conditions for pattern comparison
        - Offers risk environment assessment for position sizing
        """
        # Extract current market data and technical features
        current_window = data.get('current_window', {})
        features = current_window.get('features', {})
        
        # Generate business-friendly analysis for each major technical dimension
        
        # RSI Analysis: Momentum and overbought/oversold assessment
        # Critical for understanding market sentiment and potential reversal points
        rsi_analysis = self._analyze_rsi(features.get('rsi_values', []))
        
        # Volatility Analysis: Risk environment and market uncertainty
        # Essential for position sizing and risk management decisions
        volatility_analysis = self._analyze_volatility(features.get('atr_percentile_values', []))
        
        # Trend Analysis: Directional bias and momentum sustainability
        # Key for understanding prevailing market forces and continuation probability
        trend_analysis = self._analyze_trend(features)
        
        # Volume Analysis: Market participation and conviction assessment
        # Important for validating price movements and institutional interest
        volume_analysis = self._analyze_volume(features.get('volume_roc_values', []))
        
        # Synthesize overall market assessment and outlook
        return f"""
📈 CURRENT MARKET CONDITIONS ANALYSIS
═══════════════════════════════════════════════════════════════════════════════

Market Momentum & Sentiment:
{rsi_analysis}

Volatility Environment:
{volatility_analysis}

Trend Dynamics:
{trend_analysis}

Volume Characteristics:
{volume_analysis}

Overall Assessment: The current market environment shows {self._get_overall_market_assessment(features)} 
characteristics, suggesting {self._get_market_outlook(features)} in the near term.
        """

    def _generate_pattern_analysis(self, data: Dict[str, Any]) -> str:
        """
        Generate Historical Pattern Analysis Section
        
        Provides comprehensive analysis of historically similar market periods,
        offering crucial context for understanding the reliability and precedent
        of current market conditions. This section transforms quantitative 
        similarity scores into qualitative business insights.
        
        Analysis Dimensions:
        - Pattern Quality: Distribution of similarity scores and confidence levels
        - Seasonal Context: When similar patterns historically occurred
        - Historical Context: Decade-based distribution and market regime analysis
        - Reliability Assessment: Statistical consistency and pattern dependability
        
        Args:
            data: Analysis data containing similar_patterns list
            
        Returns:
            Historical pattern analysis section with business context
            
        Business Intelligence:
        - Validates current analysis against historical precedent
        - Identifies seasonal biases and temporal patterns
        - Provides confidence assessment for decision-making
        - Offers market regime context for risk assessment
        """
        # Extract historical pattern data for analysis
        similar_patterns = data.get('similar_patterns', [])
        
        # Handle case of insufficient historical precedent - critical for risk assessment
        if not similar_patterns:
            return """
🔍 HISTORICAL PATTERN ANALYSIS
═══════════════════════════════════════════════════════════════════════════════

No similar historical patterns were identified with sufficient confidence levels.
This suggests the current market conditions may be unusual or unprecedented.
            """
        
        # Generate comprehensive pattern analysis across multiple dimensions
        
        # Pattern Quality Analysis: Distribution of similarity scores and pattern count
        # Essential for understanding the statistical foundation of the analysis
        pattern_summary = self._summarize_patterns(similar_patterns)
        
        # Seasonal Analysis: When similar patterns historically occurred
        # Critical for understanding cyclical market behavior and timing biases
        seasonal_analysis = self._analyze_seasonal_patterns(similar_patterns)
        
        # Decade Analysis: Historical distribution across different market eras
        # Important for understanding market regime changes and secular trends
        decade_analysis = self._analyze_decade_patterns(similar_patterns)
        
        # Synthesize comprehensive historical pattern analysis
        return f"""
🔍 HISTORICAL PATTERN ANALYSIS
═══════════════════════════════════════════════════════════════════════════════

Pattern Distribution:
{pattern_summary}

Seasonal Insights:
{seasonal_analysis}

Historical Context:
{decade_analysis}

Pattern Reliability: {self._assess_pattern_reliability(similar_patterns)}
        """

    def _generate_risk_assessment(self, data: Dict[str, Any]) -> str:
        """
        Generate Comprehensive Risk Assessment - Critical for Investment Decisions
        
        This method performs multi-dimensional risk analysis combining historical
        precedent with current market conditions to provide specific, actionable
        risk management guidance. It's essential for position sizing and risk
        management decisions.
        
        Risk Assessment Framework:
        1. Historical Downside Analysis: Probability and magnitude of negative outcomes
        2. Volatility Risk Assessment: Current market uncertainty and price stability
        3. Pattern Consistency Analysis: Reliability of historical precedents
        4. Integrated Risk Rating: Combined assessment with specific mitigation strategies
        
        Args:
            data: Analysis data with patterns and current market conditions
            
        Returns:
            Comprehensive risk assessment with specific recommendations
            
        Business Critical Elements:
        - Quantified downside probability based on historical evidence
        - Specific position sizing recommendations
        - Actionable risk mitigation strategies
        - Maximum historical decline analysis for worst-case scenario planning
        - Current volatility environment assessment
        """
        # Extract data for comprehensive risk analysis
        similar_patterns = data.get('similar_patterns', [])
        current_window = data.get('current_window', {})
        
        # Handle high-risk scenario of insufficient historical data
        if not similar_patterns:
            return """
⚠️  RISK ASSESSMENT
═══════════════════════════════════════════════════════════════════════════════

Risk Level: HIGH (Insufficient historical data for reliable assessment)
Primary Concerns: Lack of comparable historical patterns increases uncertainty
Recommendation: Exercise extreme caution and consider position sizing accordingly
            """
        
        # Perform multi-dimensional risk analysis
        
        # Historical Risk Metrics: Quantify downside probability and magnitude
        # Critical for understanding potential losses based on historical precedent
        risk_metrics = self._calculate_risk_metrics(similar_patterns)
        
        # Current Volatility Risk: Assess immediate market uncertainty
        # Essential for understanding current risk environment
        volatility_risk = self._assess_volatility_risk(current_window)
        
        # Pattern Consistency: Evaluate reliability of historical analysis
        # Important for confidence in risk projections
        pattern_consistency = self._assess_pattern_consistency(similar_patterns)
        
        # Integrated Risk Assessment: Combine all factors into actionable guidance
        overall_risk = self._determine_overall_risk(risk_metrics, volatility_risk, pattern_consistency)
        
        # Generate comprehensive risk assessment with specific recommendations
        return f"""
⚠️  RISK ASSESSMENT
═══════════════════════════════════════════════════════════════════════════════

Overall Risk Level: {overall_risk['level'].upper()}

Key Risk Factors:
• Downside Risk: {risk_metrics['downside_probability']:.0f}% chance of negative returns
• Maximum Historical Decline: {risk_metrics['max_decline']:.1f}% in similar periods
• Volatility Risk: {volatility_risk['assessment']}
• Pattern Consistency: {pattern_consistency['description']}

Risk Mitigation Strategies:
{overall_risk['mitigation_strategies']}

Position Sizing Recommendation: {overall_risk['position_sizing']}
        """

    def _generate_outlook(self, data: Dict[str, Any]) -> str:
        """Generate future outlook based on historical patterns."""
        similar_patterns = data.get('similar_patterns', [])
        
        if not similar_patterns:
            return """
🔮 FUTURE OUTLOOK
═══════════════════════════════════════════════════════════════════════════════

Unable to provide reliable outlook due to lack of similar historical patterns.
Recommend monitoring current developments and reassessing as new data becomes available.
            """
        
        timeframe_analysis = self._analyze_timeframe_outcomes(similar_patterns)
        
        return f"""
🔮 FUTURE OUTLOOK
═══════════════════════════════════════════════════════════════════════════════

Based on {len(similar_patterns)} similar historical periods:

1-Week Outlook:
{timeframe_analysis['1_week']}

2-Week Outlook:
{timeframe_analysis['2_weeks']}

1-Month Outlook:
{timeframe_analysis['1_month']}

3-Month Outlook:
{timeframe_analysis['3_months']}

Key Catalysts to Monitor:
{self._identify_key_catalysts(similar_patterns)}

Probability of Success: {timeframe_analysis['overall_probability']:.0f}%
        """

    def _generate_detailed_comparisons(self, data: Dict[str, Any]) -> str:
        """Generate detailed comparison with top historical patterns."""
        similar_patterns = data.get('similar_patterns', [])
        
        if not similar_patterns:
            return ""
        
        comparisons = []
        for i, pattern in enumerate(similar_patterns[:5], 1):
            comparison = self._generate_single_comparison(pattern, i)
            comparisons.append(comparison)
        
        return f"""
📋 DETAILED HISTORICAL COMPARISONS
═══════════════════════════════════════════════════════════════════════════════

Top {len(comparisons)} Most Similar Periods:

{chr(10).join(comparisons)}
        """

    def _generate_single_comparison(self, pattern: Dict[str, Any], rank: int) -> str:
        """Generate detailed comparison for a single pattern."""
        start_date = self._format_date(pattern.get('window_start_date', ''))
        end_date = self._format_date(pattern.get('window_end_date', ''))
        similarity = pattern.get('similarity_score', 0)
        
        what_happened = pattern.get('what_happened_next', {})
        context = pattern.get('period_context', {})
        
        # Format outcomes
        outcomes = []
        for timeframe, data in what_happened.items():
            if isinstance(data, dict):
                return_pct = data.get('return_pct', 0)
                direction = data.get('direction', 'unknown')
                outcomes.append(f"{timeframe.replace('_', ' ').title()}: {return_pct:+.1f}% ({direction})")
        
        context_str = f"{context.get('year', 'Unknown')} {context.get('quarter', '')}"
        if 'market_event' in context:
            context_str += f" ({context['market_event']})"
        
        return f"""
#{rank}. Period: {start_date} to {end_date}
   Similarity: {similarity:.1%} | Context: {context_str}
   Outcomes: {' | '.join(outcomes) if outcomes else 'No outcome data available'}
        """

    def _generate_technical_summary(self, data: Dict[str, Any]) -> str:
        """Generate technical summary section."""
        search_summary = data.get('search_summary', {})
        current_window = data.get('current_window', {})
        
        return f"""
🔧 TECHNICAL ANALYSIS SUMMARY
═══════════════════════════════════════════════════════════════════════════════

Analysis Methodology:
• Pattern Matching Algorithm: Multi-dimensional similarity analysis
• Features Analyzed: {current_window.get('vector_length', 'Unknown')} technical indicators
• Historical Database: {search_summary.get('total_historical_windows', 0):,} periods analyzed
• Data Quality: {search_summary.get('filtered_windows', 0):,} periods met quality criteria

Current Technical Readings:
• RSI: {self._get_current_rsi(current_window):.1f}
• MACD Signal: {self._get_current_macd(current_window):.3f}
• Bollinger Band Position: {self._get_current_bb_position(current_window):.1%}
• Volume ROC: {self._get_current_volume_roc(current_window):+.1f}%
• ATR Percentile: {self._get_current_atr_percentile(current_window):.1f}th percentile

Data Period: {search_summary.get('data_period', 'Unknown')}
Patterns Found: {len(data.get('similar_patterns', []))}
        """

    def _generate_footer(self) -> str:
        """Generate report footer with disclaimers."""
        return f"""
═══════════════════════════════════════════════════════════════════════════════
                                  DISCLAIMERS
═══════════════════════════════════════════════════════════════════════════════

IMPORTANT: This analysis is for informational purposes only and should not be 
considered as investment advice. Past performance does not guarantee future results.
Historical patterns may not repeat, and market conditions can change rapidly.

Always consult with qualified financial professionals before making investment 
decisions. Consider your risk tolerance, investment objectives, and financial 
situation before acting on this analysis.

Generated by Financial Agent v1.0 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
═══════════════════════════════════════════════════════════════════════════════
        """

    # Helper methods for analysis
    def _format_date(self, date_str: str) -> str:
        """Format date string to readable format."""
        try:
            if isinstance(date_str, str) and date_str:
                # Try parsing different date formats
                for fmt in ['%Y-%m-%d %H:%M:%S%z', '%Y-%m-%d']:
                    try:
                        dt = datetime.strptime(date_str.split(' ')[0], '%Y-%m-%d')
                        return dt.strftime('%B %d, %Y')
                    except:
                        continue
            return date_str
        except:
            return date_str

    def _get_confidence_description(self, similarity_score: float) -> str:
        """Get confidence description based on similarity score."""
        for level, config in self.confidence_levels.items():
            if similarity_score >= config['threshold']:
                return config['description']
        return 'very low confidence'

    def _analyze_pattern_outcomes(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze Historical Pattern Outcomes - Core Intelligence Engine
        
        This method performs comprehensive statistical analysis of historical pattern
        outcomes to generate actionable investment insights. It transforms raw historical
        data into business intelligence by analyzing success rates, volatility, and
        market regime characteristics.
        
        Statistical Analysis Process:
        1. Extract 1-month forward returns from historical patterns
        2. Calculate success rate (percentage of positive outcomes)
        3. Compute average returns and volatility metrics
        4. Classify trend characteristics and market regime
        5. Generate evidence-based investment thesis
        
        Args:
            patterns: List of historical patterns with 'what_happened_next' data
            
        Returns:
            Dictionary with comprehensive outcome analysis including:
            - primary_trend: Expected market direction based on historical evidence
            - risk_level: Volatility-based risk assessment
            - success_rate: Percentage of historically positive outcomes
            - market_regime: Overall market environment characterization
            - investment_thesis: Evidence-based investment recommendation
            
        Business Intelligence Generated:
        - Trend Analysis: Direction and magnitude of expected movement
        - Risk Assessment: Volatility-based risk characterization
        - Success Probability: Historical win rate for similar patterns
        - Market Regime Classification: Overall environment assessment
        - Investment Thesis: Comprehensive recommendation synthesis
        """
        # Handle insufficient data scenario with appropriate risk disclosure
        if not patterns:
            return {
                'primary_trend': 'uncertain direction',
                'risk_level': 'high',
                'success_rate': 0,
                'market_regime': 'uncertain',
                'investment_thesis': 'Insufficient data for reliable analysis'
            }
        
        # Extract 1-month forward returns for statistical analysis
        # Focus on 1-month horizon as optimal balance between noise and signal
        monthly_returns = []
        positive_outcomes = 0
        
        for pattern in patterns:
            what_happened = pattern.get('what_happened_next', {})
            month_data = what_happened.get('1_month', {})
            if month_data and 'return_pct' in month_data:
                return_pct = month_data['return_pct']
                monthly_returns.append(return_pct)
                if return_pct > 0:
                    positive_outcomes += 1
        
        # Validate sufficient outcome data for reliable analysis
        if not monthly_returns:
            return {
                'primary_trend': 'uncertain direction',
                'risk_level': 'high',
                'success_rate': 0,
                'market_regime': 'data insufficient',
                'investment_thesis': 'Insufficient outcome data for analysis'
            }
        
        # Calculate core statistical metrics for business intelligence
        avg_return = statistics.mean(monthly_returns)
        success_rate = (positive_outcomes / len(monthly_returns)) * 100
        volatility = statistics.stdev(monthly_returns) if len(monthly_returns) > 1 else 0
        
        # Classify primary trend based on historical return magnitude
        # Thresholds designed for practical investment decision-making
        if avg_return > 5:
            primary_trend = 'strong upward movement'
        elif avg_return > 0:
            primary_trend = 'modest upward movement'
        elif avg_return > -5:
            primary_trend = 'sideways movement with slight downward bias'
        else:
            primary_trend = 'downward movement'
        
        # Assess risk level based on historical return volatility
        # Critical for position sizing and risk management decisions
        if volatility > 15:
            risk_level = 'high'
        elif volatility > 8:
            risk_level = 'moderate to high'
        elif volatility > 4:
            risk_level = 'moderate'
        else:
            risk_level = 'low to moderate'
        
        # Classify market regime based on success rate
        # Provides strategic context for investment approach
        if success_rate > 70:
            market_regime = 'favorable for bullish strategies'
        elif success_rate > 50:
            market_regime = 'mixed with slight bullish bias'
        elif success_rate > 30:
            market_regime = 'challenging with bearish undertones'
        else:
            market_regime = 'predominantly bearish'
        
        # Generate evidence-based investment thesis
        # Synthesizes all analysis into actionable recommendation
        if success_rate > 60 and avg_return > 2:
            thesis = 'Historical patterns support a cautiously optimistic outlook with managed risk exposure'
        elif success_rate > 50:
            thesis = 'Mixed historical signals suggest a balanced approach with active risk management'
        else:
            thesis = 'Historical patterns indicate elevated risk; consider defensive positioning'
        
        # Return comprehensive outcome analysis for report generation
        return {
            'primary_trend': primary_trend,
            'risk_level': risk_level,
            'success_rate': success_rate,
            'market_regime': market_regime,
            'investment_thesis': thesis
        }

    def _analyze_rsi(self, rsi_values: List[float]) -> str:
        """
        Analyze RSI (Relative Strength Index) - Momentum and Sentiment Indicator
        
        Translates technical RSI readings into business-friendly momentum analysis.
        RSI is critical for understanding market sentiment, overbought/oversold conditions,
        and potential reversal points that impact investment timing decisions.
        
        RSI Interpretation Framework:
        - 70+: Overbought territory, potential selling pressure
        - 60-70: Strong bullish momentum, continued upside potential
        - 40-60: Neutral momentum, consolidation or indecision
        - 30-40: Weak momentum, downside pressure likely
        - <30: Oversold territory, potential buying opportunity
        
        Args:
            rsi_values: List of RSI values from the current window (7-day period)
            
        Returns:
            Business-friendly RSI analysis with momentum assessment and trend direction
            
        Business Value:
        - Identifies optimal entry/exit timing based on momentum extremes
        - Provides sentiment assessment for market positioning
        - Warns of potential reversal conditions for risk management
        - Offers trend sustainability analysis for position holding decisions
        """
        # Validate RSI data availability
        if not rsi_values:
            return "• RSI data unavailable - cannot assess momentum conditions"
        
        # Extract current RSI reading and calculate trend direction
        current_rsi = rsi_values[-1]
        rsi_trend = rsi_values[-1] - rsi_values[0] if len(rsi_values) > 1 else 0
        
        # Interpret RSI level using business-friendly momentum analysis
        # Thresholds based on standard technical analysis best practices
        if current_rsi > 70:
            momentum = "overbought conditions suggest potential selling pressure"
        elif current_rsi < 30:
            momentum = "oversold conditions may present buying opportunities"
        elif current_rsi > 60:
            momentum = "strong bullish momentum with room for continued upside"
        elif current_rsi < 40:
            momentum = "weak momentum suggests continued downside pressure"
        else:
            momentum = "neutral momentum suggests consolidation or indecision"
        
        # Assess RSI trend direction for momentum sustainability analysis
        trend_desc = "strengthening" if rsi_trend > 5 else "weakening" if rsi_trend < -5 else "stable"
        
        # Generate comprehensive momentum assessment with specific readings
        return f"• Current momentum shows {momentum} (RSI: {current_rsi:.1f}, trend: {trend_desc})"

    def _analyze_volatility(self, atr_values: List[float]) -> str:
        """Analyze volatility and return business description."""
        if not atr_values:
            return "• Volatility data unavailable - cannot assess risk environment"
        
        current_atr = atr_values[-1]
        
        if current_atr > 80:
            volatility_desc = "extremely high volatility creates significant risk and opportunity"
        elif current_atr > 60:
            volatility_desc = "elevated volatility suggests increased market uncertainty"
        elif current_atr > 40:
            volatility_desc = "moderate volatility provides balanced risk-reward environment"
        elif current_atr > 20:
            volatility_desc = "low volatility suggests stable, predictable price movements"
        else:
            volatility_desc = "very low volatility may indicate complacency or accumulation"
        
        return f"• Market volatility shows {volatility_desc} ({current_atr:.0f}th percentile)"

    def _analyze_trend(self, features: Dict[str, Any]) -> str:
        """Analyze trend indicators and return business description."""
        macd_values = features.get('macd_signal_values', [])
        macd_trend = features.get('macd_signal_trend', [0])[0] if features.get('macd_signal_trend') else 0
        
        if not macd_values:
            return "• Trend data unavailable - cannot assess directional bias"
        
        if macd_trend > 0.5:
            trend_desc = "strengthening uptrend suggests continued bullish momentum"
        elif macd_trend < -0.5:
            trend_desc = "weakening trend indicates potential bearish pressure"
        else:
            trend_desc = "sideways trend suggests market consolidation"
        
        return f"• Trend analysis indicates {trend_desc} (MACD trend: {macd_trend:+.2f})"

    def _analyze_volume(self, volume_roc_values: List[float]) -> str:
        """Analyze volume and return business description."""
        if not volume_roc_values:
            return "• Volume data unavailable - cannot assess participation levels"
        
        recent_volume = statistics.mean(volume_roc_values[-3:]) if len(volume_roc_values) >= 3 else volume_roc_values[-1]
        
        if recent_volume > 20:
            volume_desc = "unusually high volume suggests strong institutional interest"
        elif recent_volume > 10:
            volume_desc = "elevated volume indicates increased market participation"
        elif recent_volume > -10:
            volume_desc = "normal volume suggests typical market engagement"
        elif recent_volume > -25:
            volume_desc = "below-average volume may indicate lack of conviction"
        else:
            volume_desc = "very low volume suggests minimal market interest"
        
        return f"• Trading volume shows {volume_desc} (recent ROC: {recent_volume:+.1f}%)"

    def _get_overall_market_assessment(self, features: Dict[str, Any]) -> str:
        """Get overall market assessment."""
        # This is a simplified assessment - you could make it more sophisticated
        rsi_values = features.get('rsi_values', [])
        atr_values = features.get('atr_percentile_values', [])
        
        if not rsi_values or not atr_values:
            return "uncertain"
        
        current_rsi = rsi_values[-1]
        current_atr = atr_values[-1]
        
        if current_rsi > 60 and current_atr < 50:
            return "bullish but stable"
        elif current_rsi > 60 and current_atr > 70:
            return "bullish but volatile"
        elif current_rsi < 40 and current_atr > 70:
            return "bearish and volatile"
        elif current_rsi < 40 and current_atr < 50:
            return "bearish but stable"
        else:
            return "neutral to mixed"

    def _get_market_outlook(self, features: Dict[str, Any]) -> str:
        """Get market outlook based on features."""
        assessment = self._get_overall_market_assessment(features)
        
        if "bullish" in assessment:
            return "continued upward movement with appropriate risk management"
        elif "bearish" in assessment:
            return "defensive positioning and capital preservation"
        else:
            return "cautious monitoring and tactical positioning"

    def _summarize_patterns(self, patterns: List[Dict[str, Any]]) -> str:
        """Summarize pattern distribution."""
        if not patterns:
            return "No patterns available for analysis"
        
        total_patterns = len(patterns)
        very_high_similarity = len([p for p in patterns if p.get('similarity_score', 0) >= 0.90])
        high_similarity = len([p for p in patterns if 0.80 <= p.get('similarity_score', 0) < 0.90])
        
        return f"""• {total_patterns} similar historical periods identified
• {very_high_similarity} periods show very high similarity (90%+ match)
• {high_similarity} periods show high similarity (80-90% match)
• Pattern quality is {'excellent' if very_high_similarity > 3 else 'good' if very_high_similarity > 0 else 'moderate'}"""

    def _analyze_seasonal_patterns(self, patterns: List[Dict[str, Any]]) -> str:
        """Analyze seasonal distribution of patterns."""
        seasons = {}
        for pattern in patterns:
            context = pattern.get('period_context', {})
            season = context.get('season', 'Unknown')
            seasons[season] = seasons.get(season, 0) + 1
        
        if not seasons or all(season == 'Unknown' for season in seasons):
            return "• Seasonal data not available for analysis"
        
        dominant_season = max(seasons, key=seasons.get) if seasons else 'Unknown'
        return f"• Most similar periods occurred during {dominant_season} ({seasons.get(dominant_season, 0)} instances)"

    def _analyze_decade_patterns(self, patterns: List[Dict[str, Any]]) -> str:
        """Analyze decade distribution of patterns."""
        decades = {}
        for pattern in patterns:
            context = pattern.get('period_context', {})
            year = context.get('year')
            if year and year != 'Unknown':
                try:
                    decade = f"{int(year)//10*10}s"
                    decades[decade] = decades.get(decade, 0) + 1
                except:
                    pass
        
        if not decades:
            return "• Historical period data not available"
        
        dominant_decade = max(decades, key=decades.get)
        return f"• {len(decades)} different decades represented, most common: {dominant_decade} ({decades[dominant_decade]} instances)"

    def _assess_pattern_reliability(self, patterns: List[Dict[str, Any]]) -> str:
        """Assess overall pattern reliability."""
        if not patterns:
            return "Cannot assess reliability due to insufficient patterns"
        
        avg_similarity = statistics.mean([p.get('similarity_score', 0) for p in patterns])
        consistency = statistics.stdev([p.get('similarity_score', 0) for p in patterns]) if len(patterns) > 1 else 0
        
        if avg_similarity > 0.85 and consistency < 0.05:
            return "Very high reliability - patterns are highly consistent"
        elif avg_similarity > 0.75 and consistency < 0.10:
            return "High reliability - patterns show good consistency"
        elif avg_similarity > 0.65:
            return "Moderate reliability - patterns show reasonable consistency"
        else:
            return "Lower reliability - patterns show significant variation"

    def _calculate_risk_metrics(self, patterns: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate comprehensive risk metrics."""
        negative_outcomes = 0
        all_returns = []
        
        for pattern in patterns:
            what_happened = pattern.get('what_happened_next', {})
            month_data = what_happened.get('1_month', {})
            if month_data and 'return_pct' in month_data:
                return_pct = month_data['return_pct']
                all_returns.append(return_pct)
                if return_pct < 0:
                    negative_outcomes += 1
        
        if not all_returns:
            return {
                'downside_probability': 50.0,
                'max_decline': 0.0,
                'average_decline': 0.0
            }
        
        downside_probability = (negative_outcomes / len(all_returns)) * 100
        negative_returns = [r for r in all_returns if r < 0]
        max_decline = abs(min(negative_returns)) if negative_returns else 0
        avg_decline = abs(statistics.mean(negative_returns)) if negative_returns else 0
        
        return {
            'downside_probability': downside_probability,
            'max_decline': max_decline,
            'average_decline': avg_decline
        }

    def _assess_volatility_risk(self, current_window: Dict[str, Any]) -> Dict[str, str]:
        """Assess volatility-based risk."""
        features = current_window.get('features', {})
        atr_values = features.get('atr_percentile_values', [])
        
        if not atr_values:
            return {'assessment': 'Cannot assess - volatility data unavailable'}
        
        current_atr = atr_values[-1]
        
        if current_atr > 80:
            return {'assessment': 'High volatility increases position risk significantly'}
        elif current_atr > 60:
            return {'assessment': 'Elevated volatility requires careful position sizing'}
        elif current_atr > 40:
            return {'assessment': 'Moderate volatility allows for standard risk management'}
        else:
            return {'assessment': 'Low volatility provides favorable risk environment'}

    def _assess_pattern_consistency(self, patterns: List[Dict[str, Any]]) -> Dict[str, str]:
        """Assess pattern consistency."""
        if len(patterns) < 3:
            return {'description': 'Insufficient patterns for consistency assessment'}
        
        similarities = [p.get('similarity_score', 0) for p in patterns]
        consistency = statistics.stdev(similarities)
        
        if consistency < 0.02:
            return {'description': 'Very high consistency across similar patterns'}
        elif consistency < 0.05:
            return {'description': 'Good consistency across similar patterns'}
        elif consistency < 0.10:
            return {'description': 'Moderate consistency with some variation'}
        else:
            return {'description': 'Low consistency - patterns show significant variation'}

    def _determine_overall_risk(self, risk_metrics: Dict, volatility_risk: Dict, pattern_consistency: Dict) -> Dict[str, str]:
        """Determine overall risk assessment."""
        downside_prob = risk_metrics.get('downside_probability', 50)
        
        if downside_prob > 70:
            risk_level = 'very high'
            mitigation = """• Consider significantly reduced position size
• Implement strict stop-loss levels
• Monitor positions closely with daily reviews
• Consider hedging strategies"""
            position_sizing = "Maximum 25% of typical position size"
        elif downside_prob > 50:
            risk_level = 'high'
            mitigation = """• Use reduced position sizing
• Set clear exit criteria before entry
• Monitor key technical levels closely
• Consider partial profit-taking on any gains"""
            position_sizing = "50% of typical position size"
        elif downside_prob > 30:
            risk_level = 'moderate'
            mitigation = """• Standard risk management protocols
• Set stop-loss at key technical levels
• Monitor volume and momentum indicators
• Be prepared to adjust position if conditions change"""
            position_sizing = "75% of typical position size"
        else:
            risk_level = 'low to moderate'
            mitigation = """• Standard position sizing appropriate
• Monitor for any changes in market regime
• Maintain disciplined approach to profit-taking
• Keep stop-loss levels reasonable but not too tight"""
            position_sizing = "Normal position sizing acceptable"
        
        return {
            'level': risk_level,
            'mitigation_strategies': mitigation,
            'position_sizing': position_sizing
        }

    def _analyze_timeframe_outcomes(self, patterns: List[Dict[str, Any]]) -> Dict[str, str]:
        """Analyze outcomes across different timeframes."""
        timeframes = ['1_week', '2_weeks', '1_month', '3_months']
        analysis = {}
        
        for timeframe in timeframes:
            returns = []
            positive_count = 0
            
            for pattern in patterns:
                what_happened = pattern.get('what_happened_next', {})
                period_data = what_happened.get(timeframe, {})
                if period_data and 'return_pct' in period_data:
                    return_pct = period_data['return_pct']
                    returns.append(return_pct)
                    if return_pct > 0:
                        positive_count += 1
            
            if returns:
                avg_return = statistics.mean(returns)
                success_rate = (positive_count / len(returns)) * 100
                
                if avg_return > 5:
                    outlook = f"Strong positive outlook with {avg_return:.1f}% average return"
                elif avg_return > 2:
                    outlook = f"Modestly positive outlook with {avg_return:.1f}% average return"
                elif avg_return > -2:
                    outlook = f"Neutral outlook with minimal {avg_return:+.1f}% average movement"
                else:
                    outlook = f"Negative outlook with {avg_return:.1f}% average decline"
                
                analysis[timeframe] = f"• {outlook} ({success_rate:.0f}% historical success rate)"
            else:
                analysis[timeframe] = f"• No historical data available for {timeframe.replace('_', ' ')}"
        
        # Calculate overall probability
        all_monthly_returns = []
        for pattern in patterns:
            what_happened = pattern.get('what_happened_next', {})
            month_data = what_happened.get('1_month', {})
            if month_data and 'return_pct' in month_data:
                all_monthly_returns.append(month_data['return_pct'])
        
        overall_success = (len([r for r in all_monthly_returns if r > 0]) / len(all_monthly_returns)) * 100 if all_monthly_returns else 50
        analysis['overall_probability'] = overall_success
        
        return analysis

    def _identify_key_catalysts(self, patterns: List[Dict[str, Any]]) -> str:
        """Identify key catalysts to monitor."""
        # This could be enhanced with more sophisticated analysis
        market_events = set()
        for pattern in patterns:
            context = pattern.get('period_context', {})
            if 'market_event' in context:
                market_events.add(context['market_event'])
        
        if market_events:
            events_str = ', '.join(market_events)
            return f"• Historical patterns coincided with: {events_str}\n• Monitor for similar market developments or sector-specific news"
        else:
            return """• Monitor overall market sentiment and momentum indicators
• Watch for changes in sector rotation or institutional positioning
• Pay attention to volume patterns and any unusual market activity"""

    # Helper methods for current technical readings
    def _get_current_rsi(self, current_window: Dict[str, Any]) -> float:
        """Get current RSI value."""
        features = current_window.get('features', {})
        rsi_values = features.get('rsi_values', [])
        return rsi_values[-1] if rsi_values else 0.0

    def _get_current_macd(self, current_window: Dict[str, Any]) -> float:
        """Get current MACD signal value."""
        features = current_window.get('features', {})
        macd_values = features.get('macd_signal_values', [])
        return macd_values[-1] if macd_values else 0.0

    def _get_current_bb_position(self, current_window: Dict[str, Any]) -> float:
        """Get current Bollinger Band position."""
        features = current_window.get('features', {})
        bb_values = features.get('bb_position_values', [])
        return bb_values[-1] if bb_values else 0.0

    def _get_current_volume_roc(self, current_window: Dict[str, Any]) -> float:
        """Get current Volume ROC."""
        features = current_window.get('features', {})
        volume_values = features.get('volume_roc_values', [])
        return volume_values[-1] if volume_values else 0.0

    def _get_current_atr_percentile(self, current_window: Dict[str, Any]) -> float:
        """Get current ATR percentile."""
        features = current_window.get('features', {})
        atr_values = features.get('atr_percentile_values', [])
        return atr_values[-1] if atr_values else 0.0


def generate_report_from_json(json_file_path: str, output_file_path: Optional[str] = None) -> str:
    """
    Module-Level Report Generation Interface
    
    Primary external interface for converting JSON analysis files into comprehensive
    natural language business reports. This function serves as the main entry point
    for standalone report generation and integration with other system components.
    
    Workflow:
    1. Load and validate JSON analysis data from file
    2. Initialize the report generator with business intelligence frameworks
    3. Generate comprehensive natural language report
    4. Optionally save to file for distribution
    5. Return report string for immediate use or further processing
    
    Args:
        json_file_path: Path to JSON file containing complete analysis results from pattern_searcher
        output_file_path: Optional path for saving the formatted report (if None, returns string only)
        
    Returns:
        Complete natural language report as formatted string ready for business presentation
        
    Integration Points:
    - Used by generate_report.py for standalone report generation
    - Called by run_analysis.py for integrated analysis and reporting
    - Supports both file output and programmatic string return
    - Handles all error conditions gracefully with informative messages
    
    Error Handling:
    - File I/O errors (missing files, permissions, encoding issues)
    - JSON parsing errors (malformed analysis data)
    - Report generation errors (missing required data fields)
    - Output file errors (directory permissions, disk space)
    
    Business Value:
    - Enables standalone report generation from saved analysis results
    - Provides flexible output options for different use cases
    - Ensures consistent error handling and user feedback
    - Facilitates batch processing and automated report generation
    """
    try:
        # Load and parse JSON analysis data with error handling
        with open(json_file_path, 'r') as f:
            analysis_data = json.load(f)
        
        # Initialize report generator with business intelligence frameworks
        generator = NaturalLanguageReportGenerator()
        
        # Generate comprehensive natural language report
        report = generator.generate_full_report(analysis_data)
        
        # Optionally save report to file for distribution
        if output_file_path:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Report saved to: {output_file_path}")
        
        # Return report string for immediate use or further processing
        return report
        
    except Exception as e:
        # Generate informative error message for troubleshooting
        error_msg = f"Error generating report: {str(e)}"
        print(error_msg)
        return error_msg


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python natural_language_generator.py <json_file_path> [output_file_path]")
        sys.exit(1)
    
    json_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    report = generate_report_from_json(json_path, output_path)
    
    if not output_path:
        print(report) 