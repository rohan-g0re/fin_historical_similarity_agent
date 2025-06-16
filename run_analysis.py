#!/usr/bin/env python3
"""
Financial Agent - Interactive Command Line Interface
Run pattern analysis on any stock symbol interactively.

This is the main entry point for the Financial Pattern Analysis System.
It provides an interactive interface for:
1. Collecting financial data for any stock symbol
2. Running pattern matching analysis against historical data
3. Generating both technical JSON reports and business-friendly natural language reports
4. Displaying results in user-friendly terminal formats

The system implements a 5-indicator methodology using 7-day market windows
to find similar historical patterns using cosine similarity calculations.

Usage:
    python run_analysis.py
"""

import sys
import json
from datetime import datetime
from pathlib import Path

# Add src to path for imports - allows clean module imports from src/ directory
sys.path.append(str(Path(__file__).parent / "src"))

# Import core system components for the complete analysis pipeline
from src.core.config_manager import ConfigManager
from src.similarity.pattern_searcher import PatternSearcher
from src.reports.natural_language_generator import NaturalLanguageReportGenerator


def print_banner():
    """
    Print welcome banner to establish system identity and purpose.
    Creates professional appearance for command-line tool.
    """
    print("=" * 60)
    print("🚀 FINANCIAL AGENT - PATTERN ANALYSIS SYSTEM")
    print("=" * 60)
    print("Interactive analysis for any stock symbol")
    print()


def print_target_analysis(current_window):
    """
    Print target pattern analysis in a user-friendly format.
    
    This function analyzes the current market window and translates
    technical indicators into business-friendly interpretations:
    - RSI zones (overbought/oversold/neutral)
    - Volatility regimes (high/medium/low) 
    - Trend directions (uptrend/downtrend/sideways)
    
    Args:
        current_window (dict): The current 7-day market window with features
                              Contains technical indicators and metadata
    """
    print("🎯 TARGET PATTERN ANALYSIS")
    print("-" * 40)
    
    # Extract features and calculate zones - technical indicators are pre-calculated
    features = current_window.get('features', {})
    
    # RSI analysis - Relative Strength Index interpretation
    # RSI > 70 = overbought (potential sell signal)
    # RSI < 30 = oversold (potential buy signal) 
    # RSI 30-70 = neutral territory
    rsi_values = features.get('rsi_values', [])
    if rsi_values:
        current_rsi = rsi_values[-1]  # Get most recent RSI value
        if current_rsi > 70:
            rsi_zone = "OVERBOUGHT"  # Stock may be due for a pullback
        elif current_rsi < 30:
            rsi_zone = "OVERSOLD"   # Stock may be due for a bounce
        else:
            rsi_zone = "NEUTRAL"    # No strong momentum signal
        print(f"📊 Market Regime:")
        print(f"   • RSI Zone: {rsi_zone} (RSI: {current_rsi:.1f})")
    
    # Volatility analysis - ATR Percentile interpretation
    # High percentile = high volatility = higher risk/reward potential
    # Low percentile = low volatility = more stable price action
    atr_values = features.get('atr_percentile_values', [])
    if atr_values:
        current_atr = atr_values[-1]  # Get most recent ATR percentile
        if current_atr > 75:
            vol_regime = "HIGH"     # Expect larger price swings
        elif current_atr < 25:
            vol_regime = "LOW"      # Expect smaller price movements
        else:
            vol_regime = "MEDIUM"   # Average volatility environment
        print(f"   • Volatility: {vol_regime} ({current_atr:.1f}th percentile)")
    
    # Trend analysis - MACD Signal Line interpretation
    # Rising MACD Signal = strengthening uptrend
    # Falling MACD Signal = strengthening downtrend
    # Flat MACD Signal = sideways/consolidation
    macd_values = features.get('macd_signal_values', [])
    if len(macd_values) >= 2:
        # Compare latest value to previous to determine trend direction
        if macd_values[-1] > macd_values[-2]:
            trend = "UPTREND"       # Bullish momentum building
        elif macd_values[-1] < macd_values[-2]:
            trend = "DOWNTREND"     # Bearish momentum building
        else:
            trend = "SIDEWAYS"      # No clear directional momentum
        print(f"   • Trend: {trend}")
    
    print()


def print_search_summary(search_summary, processing_time_ms=0):
    """
    Print search summary statistics to show analysis scope and performance.
    
    Provides transparency about:
    - How much historical data was analyzed
    - How many patterns passed filtering criteria
    - System performance metrics
    
    Args:
        search_summary (dict): Summary statistics from pattern search
        processing_time_ms (float): Total processing time in milliseconds
    """
    print("📈 SEARCH SUMMARY")
    print("-" * 40)
    
    # Handle different key names and ensure safe formatting
    total_patterns = search_summary.get('total_patterns_searched', search_summary.get('total_historical_windows', 0))
    patterns_found = search_summary.get('patterns_found', search_summary.get('similar_patterns_found', 0))
    
    # Safe formatting - only use comma separator for integers
    if isinstance(total_patterns, int):
        print(f"🔍 Total Historical Windows: {total_patterns:,}")
    else:
        print(f"🔍 Total Historical Windows: {total_patterns}")
    
    if isinstance(patterns_found, int):
        print(f"✅ Similar Patterns Found: {patterns_found}")
    else:
        print(f"✅ Similar Patterns Found: {patterns_found}")
    
    # Processing time
    print(f"⚡ Processing Time: {processing_time_ms:.1f}ms")
    print()


def print_similar_periods(similar_periods, top_k=10):
    """
    Print similar periods in a formatted table for easy scanning.
    
    Creates a visual table showing:
    - Rank order by similarity score
    - Historical date ranges for context
    - Similarity scores (0-1 scale)
    - Human-readable similarity levels
    
    Args:
        similar_periods (list): List of similar pattern dictionaries
        top_k (int): Maximum number of results to display
    """
    if not similar_periods:
        print("❌ No similar periods found.")
        return
    
    print(f"🔍 TOP {min(len(similar_periods), top_k)} SIMILAR PERIODS")
    print("-" * 75)
    print(f"{'Rank':<4} {'Date Range':<23} {'Similarity':<12} {'Level':<12}")
    print("-" * 75)
    
    # Display up to top_k results in ranked order
    for period in similar_periods[:top_k]:
        rank = period.get('rank', 'N/A')
        start_date = period.get('window_start_date', 'N/A')
        end_date = period.get('window_end_date', 'N/A')
        similarity = period.get('similarity_score', 0)
        
        # Convert numerical similarity to descriptive level for business users
        # These thresholds help non-technical users understand confidence levels
        if similarity >= 0.90:
            level = "Very High"     # Extremely similar patterns - high confidence
        elif similarity >= 0.80:
            level = "High"          # Strong similarity - good confidence
        elif similarity >= 0.70:
            level = "Medium-High"   # Reasonable similarity - moderate confidence
        elif similarity >= 0.60:
            level = "Medium"        # Some similarity - use with caution
        elif similarity >= 0.50:
            level = "Low-Medium"    # Weak similarity - low confidence
        else:
            level = "Low"           # Poor similarity - be very cautious
        
        print(f"{rank:<4} {start_date} to {end_date} {similarity:<12.3f} {level:<12}")
    
    print()


def print_detailed_analysis(period, rank):
    """
    Print detailed analysis for a specific historical period.
    
    Provides deeper insight into a single pattern match including:
    - Temporal context (dates)
    - Confidence assessment (similarity score + interpretation)
    - Technical details (feature vector length)
    
    Args:
        period (dict): Single pattern match with metadata
        rank (int): Rank position of this pattern
    """
    print(f"📊 DETAILED ANALYSIS - RANK #{rank}")
    print("-" * 50)
    start_date = period.get('window_start_date', 'N/A')
    end_date = period.get('window_end_date', 'N/A')
    similarity = period.get('similarity_score', 0)
    
    # Convert similarity score to business-friendly interpretation
    if similarity >= 0.90:
        level = "Very High"
    elif similarity >= 0.80:
        level = "High"
    elif similarity >= 0.70:
        level = "Medium-High"
    elif similarity >= 0.60:
        level = "Medium"
    elif similarity >= 0.50:
        level = "Low-Medium"
    else:
        level = "Low"
    
    print(f"📅 Period: {start_date} to {end_date}")
    print(f"🎯 Similarity: {similarity:.3f} ({level})")
    print(f"🔢 Vector Length: {period.get('vector_length', 'N/A')} features")
    
    print()


def get_user_input():
    """
    Get stock symbol and preferences from user interactively.
    
    This function handles the interactive mode workflow:
    1. Validates stock symbol input
    2. Presents analysis option menu
    3. Handles user choice validation
    4. Asks about natural language report generation
    
    Returns:
        tuple: (symbol, top_k, detailed, generate_report)
               - symbol: Validated stock symbol
               - top_k: Number of results to show
               - detailed: Whether to show detailed analysis
               - generate_report: Whether to generate business report
    """
    print("🔤 Enter Stock Symbol (e.g., AAPL, MSFT, GOOGL):")
    
    # Symbol validation loop - ensures user provides valid input
    while True:
        symbol = input("Symbol: ").strip().upper()
        if symbol:
            break  # Valid symbol entered
        print("Please enter a valid stock symbol.")
    
    # Analysis options menu - provides structured choices for different use cases
    print("\n⚙️  Analysis Options:")
    print("1. Quick Analysis (Top 5 results)")          # Fast overview
    print("2. Standard Analysis (Top 10 results)")      # Balanced depth
    print("3. Detailed Analysis (Top 20 results)")      # Comprehensive view
    print("4. Custom")                                   # User-defined parameters
    
    # Option selection and validation loop
    while True:
        choice = input("Choose option (1-4): ").strip()
        if choice == "1":
            top_k, detailed = 5, False      # Quick overview for time-sensitive decisions
        elif choice == "2":
            top_k, detailed = 10, False     # Standard analysis for most use cases
        elif choice == "3":
            top_k, detailed = 20, False     # Deep analysis for research purposes
        elif choice == "4":
            # Custom option allows power users to specify exact parameters
            while True:
                try:
                    top_k = int(input("Number of results (1-50): "))
                    if 1 <= top_k <= 50:
                        break  # Valid range entered
                    print("Please enter a number between 1 and 50.")
                except ValueError:
                    print("Please enter a valid number.")
            
            # Ask if user wants detailed analysis of top result
            detailed = input("Show detailed analysis for top result? (y/n): ").strip().lower() == 'y'
        else:
            print("Please choose 1, 2, 3, or 4.")
            continue

        # Ask about natural language report generation
        # This bridges technical analysis with business communication
        generate_report = input("\n📝 Generate business-friendly natural language report? (y/n): ").strip().lower() == 'y'
        
        break  # Valid choice made, exit loop
    
    return symbol, top_k, detailed, generate_report


def run_analysis(symbol, top_k=10, show_detailed=False, generate_report=False):
    """
    Run the complete pattern analysis workflow for a given symbol.
    
    This is the core function that orchestrates the entire analysis pipeline:
    1. Initialize the pattern search system
    2. Execute the similarity search against historical data  
    3. Display results in user-friendly format
    4. Optionally generate natural language business reports
    5. Save results to JSON for further analysis
    
    Args:
        symbol (str): Stock symbol to analyze (e.g., 'AAPL')
        top_k (int): Number of top results to display
        show_detailed (bool): Whether to show detailed analysis for top result
        generate_report (bool): Whether to generate natural language report
        
    Returns:
        dict: Complete analysis results for programmatic use
    """
    print(f"🔄 Running analysis for {symbol}...")
    
    # Timing analysis for performance monitoring
    start_time = datetime.now()
    
    try:
        # Initialize the pattern searcher - this loads configuration and sets up all components
        searcher = PatternSearcher()
        
        # Execute the complete pattern search workflow
        # This involves: data collection, indicator calculation, window creation,
        # filtering, similarity calculation, and ranking
        results = searcher.search_similar_patterns(symbol)
        
        # Calculate processing performance metrics
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() * 1000  # Convert to milliseconds
        
        # Display analysis results in user-friendly terminal format
        print_target_analysis(results['current_conditions'])
        print_search_summary(results['search_metadata'], processing_time)
        
        # Display similar periods table
        similar_periods = results.get('similar_patterns', [])
        print_similar_periods(similar_periods, top_k)
        
        # Show detailed analysis for top result if requested
        if show_detailed and similar_periods:
            print_detailed_analysis(similar_periods[0], 1)
        
        # Save results to JSON file for further analysis/archival
        json_filename = save_results(symbol, results)
        print(f"💾 Results saved to: {json_filename}")
        
        # Generate natural language business report if requested
        if generate_report:
            print(f"\n📝 Generating natural language report...")
            report_filename = generate_natural_language_report(json_filename, symbol)
            if report_filename:
                print(f"📄 Business report saved to: {report_filename}")
            else:
                print("❌ Failed to generate natural language report")
        
        return results
        
    except Exception as e:
        # Comprehensive error handling with user-friendly messages
        print(f"❌ Error running analysis for {symbol}: {str(e)}")
        print("   Possible causes:")
        print("   • Invalid stock symbol")
        print("   • Network connectivity issues")
        print("   • Insufficient historical data")
        print("   • API rate limiting")
        return None


def save_results(symbol, results):
    """
    Save analysis results to JSON file with standardized naming.
    
    Creates permanent record of analysis for:
    - Future reference and comparison
    - Generating reports later via generate_report.py
    - Building historical analysis databases
    - Sharing results with stakeholders
    
    Args:
        symbol (str): Stock symbol for filename
        results (dict): Complete analysis results
        
    Returns:
        str: Filename of saved JSON file
    """
    # Create standardized filename with timestamp for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"analysis_{symbol}_{timestamp}.json"
    
    # Save with proper formatting for human readability
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)  # default=str handles datetime objects
    
    return filename


def generate_natural_language_report(json_filename, symbol):
    """
    Generate business-friendly natural language report from JSON analysis.
    
    Converts technical analysis data into narrative reports suitable for:
    - Business stakeholders without technical background
    - Investment committee presentations  
    - Client communication
    - Research documentation
    
    Args:
        json_filename (str): Path to JSON analysis file
        symbol (str): Stock symbol for context
        
    Returns:
        str or None: Filename of generated report, or None if failed
    """
    try:
        # Load the JSON analysis data
        with open(json_filename, 'r') as f:
            analysis_data = json.load(f)
        
        # Initialize the natural language generator
        report_generator = NaturalLanguageReportGenerator()
        
        # Generate the complete business-friendly report
        report = report_generator.generate_full_report(analysis_data)
        
        # Create report filename based on JSON filename
        report_filename = json_filename.replace('.json', '_BUSINESS_REPORT.txt')
        
        # Save the natural language report
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return report_filename
        
    except Exception as e:
        print(f"❌ Error generating natural language report: {str(e)}")
        return None


def main():
    """
    Main function that runs the interactive analysis workflow.
    
    Automatically starts in interactive mode when the script is executed.
    Provides guided experience for users to:
    1. Enter stock symbol
    2. Choose analysis depth
    3. Generate business reports
    4. Save results for future reference
    """
    # Display banner for professional appearance
    print_banner()
    
    # Get user preferences through interactive prompts
    symbol, top_k, detailed, generate_report = get_user_input()
    
    # Execute the analysis workflow
    results = run_analysis(
        symbol=symbol,
        top_k=top_k, 
        show_detailed=detailed,
        generate_report=generate_report
    )
    
    # Provide final status summary
    if results:
        print("\n" + "=" * 60)
        print("✅ Analysis completed successfully!")
        similar_count = len(results.get('similar_patterns', []))
        print(f"📊 Found {similar_count} similar historical patterns for {symbol}")
        if generate_report:
            print("📝 Business report generated for stakeholder communication")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Analysis failed - please check symbol and try again")
        print("=" * 60)


# Standard Python script entry point
if __name__ == "__main__":
    main() 