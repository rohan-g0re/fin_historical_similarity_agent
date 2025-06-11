#!/usr/bin/env python3
"""
Financial Agent - Command Line Interface
Run pattern analysis on any stock symbol from the terminal.

Usage:
    python run_analysis.py AAPL
    python run_analysis.py MSFT --top-k 15
    python run_analysis.py --interactive
    python run_analysis.py AAPL --generate-report
"""

import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.config_manager import ConfigManager
from src.similarity.pattern_searcher import PatternSearcher
from src.reports.natural_language_generator import NaturalLanguageReportGenerator


def print_banner():
    """Print welcome banner"""
    print("=" * 60)
    print("🚀 FINANCIAL AGENT - PATTERN ANALYSIS SYSTEM")
    print("=" * 60)
    print("Find similar historical patterns for any stock symbol")
    print()


def print_target_analysis(current_window):
    """Print target pattern analysis in a user-friendly format"""
    print("🎯 TARGET PATTERN ANALYSIS")
    print("-" * 40)
    print(f"📅 Date Range: {current_window.get('window_start_date', 'N/A')} to {current_window.get('window_end_date', 'N/A')}")
    
    # Extract features and calculate zones
    features = current_window.get('features', {})
    
    # RSI analysis
    rsi_values = features.get('rsi_values', [])
    if rsi_values:
        current_rsi = rsi_values[-1]
        if current_rsi > 70:
            rsi_zone = "OVERBOUGHT"
        elif current_rsi < 30:
            rsi_zone = "OVERSOLD"
        else:
            rsi_zone = "NEUTRAL"
        print(f"📊 Market Regime:")
        print(f"   • RSI Zone: {rsi_zone} (RSI: {current_rsi:.1f})")
    
    # Volatility analysis
    atr_values = features.get('atr_percentile_values', [])
    if atr_values:
        current_atr = atr_values[-1]
        if current_atr > 75:
            vol_regime = "HIGH"
        elif current_atr < 25:
            vol_regime = "LOW"
        else:
            vol_regime = "MEDIUM"
        print(f"   • Volatility: {vol_regime} ({current_atr:.1f}th percentile)")
    
    # Trend analysis
    macd_values = features.get('macd_signal_values', [])
    if len(macd_values) >= 2:
        if macd_values[-1] > macd_values[-2]:
            trend = "UPTREND"
        elif macd_values[-1] < macd_values[-2]:
            trend = "DOWNTREND"
        else:
            trend = "SIDEWAYS"
        print(f"   • Trend: {trend}")
    
    print()


def print_search_summary(search_summary, processing_time_ms=0):
    """Print search summary statistics"""
    print("📈 SEARCH SUMMARY")
    print("-" * 40)
    print(f"🔍 Total Historical Windows: {search_summary.get('total_historical_windows', 'N/A'):,}")
    print(f"🔍 Filtered Windows: {search_summary.get('filtered_windows', 'N/A'):,}")
    print(f"✅ Similar Patterns Found: {search_summary.get('similar_patterns_found', 'N/A')}")
    print(f"📅 Data Period: {search_summary.get('data_period', 'N/A')}")
    print(f"⚡ Processing Time: {processing_time_ms:.1f}ms")
    print()


def print_similar_periods(similar_periods, top_k=10):
    """Print similar periods in a formatted table"""
    if not similar_periods:
        print("❌ No similar periods found.")
        return
    
    print(f"🔍 TOP {min(len(similar_periods), top_k)} SIMILAR PERIODS")
    print("-" * 75)
    print(f"{'Rank':<4} {'Date Range':<23} {'Similarity':<12} {'Level':<12}")
    print("-" * 75)
    
    for period in similar_periods[:top_k]:
        rank = period.get('rank', 'N/A')
        start_date = period.get('window_start_date', 'N/A')
        end_date = period.get('window_end_date', 'N/A')
        similarity = period.get('similarity_score', 0)
        
        # Determine similarity level
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
        
        print(f"{rank:<4} {start_date} to {end_date} {similarity:<12.3f} {level:<12}")
    
    print()


def print_detailed_analysis(period, rank):
    """Print detailed analysis for a specific period"""
    print(f"📊 DETAILED ANALYSIS - RANK #{rank}")
    print("-" * 50)
    start_date = period.get('window_start_date', 'N/A')
    end_date = period.get('window_end_date', 'N/A')
    similarity = period.get('similarity_score', 0)
    
    # Determine similarity level
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
    """Get stock symbol and preferences from user interactively"""
    print("🔤 Enter Stock Symbol (e.g., AAPL, MSFT, GOOGL):")
    while True:
        symbol = input("Symbol: ").strip().upper()
        if symbol:
            break
        print("Please enter a valid stock symbol.")
    
    print("\n⚙️  Analysis Options:")
    print("1. Quick Analysis (Top 5 results)")
    print("2. Standard Analysis (Top 10 results)")
    print("3. Detailed Analysis (Top 20 results)")
    print("4. Custom")
    
    while True:
        choice = input("Choose option (1-4): ").strip()
        if choice == "1":
            top_k, detailed = 5, False
        elif choice == "2":
            top_k, detailed = 10, False
        elif choice == "3":
            top_k, detailed = 20, False
        elif choice == "4":
            while True:
                try:
                    top_k = int(input("Number of results (1-50): "))
                    if 1 <= top_k <= 50:
                        break
                    print("Please enter a number between 1 and 50.")
                except ValueError:
                    print("Please enter a valid number.")
            
            detailed = input("Show detailed analysis for top result? (y/n): ").strip().lower() == 'y'
        else:
            print("Please choose 1, 2, 3, or 4.")
            continue
        
        # Ask about natural language report
        generate_report = input("\n📝 Generate business-friendly natural language report? (y/n): ").strip().lower() == 'y'
        
        return symbol, top_k, detailed, generate_report


def run_analysis(symbol, top_k=10, show_detailed=False, generate_report=False):
    """Run the complete pattern analysis"""
    print(f"🔄 Analyzing {symbol}...")
    print("This may take a few moments...\n")
    
    try:
        # Initialize system
        config = ConfigManager()
        searcher = PatternSearcher(config)
        
        # Run analysis
        start_time = datetime.now()
        results = searcher.search_similar_patterns(
            symbol=symbol,
            apply_filters=True
        )
        end_time = datetime.now()
        
        # Limit results to requested top_k
        if 'similar_patterns' in results and len(results['similar_patterns']) > top_k:
            results['similar_patterns'] = results['similar_patterns'][:top_k]
        
        processing_time = (end_time - start_time).total_seconds()
        
        # Display results
        if 'symbol' in results:  # Success indicator
            print_target_analysis(results.get('current_window', {}))
            
            print_search_summary(results.get('search_summary', {}), processing_time * 1000)
            print_similar_periods(results.get('similar_patterns', []), top_k)
            
            # Show detailed analysis if requested
            if show_detailed and results.get('similar_patterns'):
                print_detailed_analysis(results['similar_patterns'][0], 1)
            
            # Save results to file (ALWAYS preserve this functionality)
            json_filename = save_results(symbol, results)
            
            # Generate natural language report if requested
            if generate_report:
                generate_natural_language_report(json_filename, symbol)
            
        else:
            print(f"❌ Analysis failed: No results returned")
            print("💡 Please check:")
            print("   • Stock symbol is valid and actively traded")
            print("   • Internet connection is stable")
            print("   • Sufficient historical data is available")
    
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        print("💡 Please check:")
        print("   • Stock symbol is valid and actively traded")
        print("   • Internet connection is stable")
        print("   • Try again in a few minutes")


def save_results(symbol, results):
    """Save results to a JSON file"""
    try:
        filename = f"analysis_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"💾 Results saved to: {filename}")
        return filename  # Return filename for report generation
    except Exception as e:
        print(f"⚠️  Could not save results: {e}")
        return None


def generate_natural_language_report(json_filename, symbol):
    """Generate and save a natural language business report"""
    if not json_filename:
        print("❌ Cannot generate report: JSON file not available")
        return
    
    try:
        print(f"\n📝 Generating natural language report for {symbol}...")
        
        # Initialize report generator
        generator = NaturalLanguageReportGenerator()
        
        # Load JSON data
        with open(json_filename, 'r') as f:
            analysis_data = json.load(f)
        
        # Generate report
        report = generator.generate_full_report(analysis_data)
        
        # Save report to file
        report_filename = json_filename.replace('.json', '_BUSINESS_REPORT.txt')
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 Business report saved to: {report_filename}")
        print(f"📊 Report contains executive summary, risk assessment, and actionable insights")
        
        # Optionally display a brief preview
        lines = report.split('\n')
        preview_lines = []
        for line in lines:
            preview_lines.append(line)
            if len(preview_lines) >= 15:  # Show first 15 lines as preview
                break
        
        print(f"\n📋 REPORT PREVIEW:")
        print("─" * 60)
        print('\n'.join(preview_lines))
        print("─" * 60)
        print(f"📖 Full report available in: {report_filename}")
        
    except Exception as e:
        print(f"❌ Error generating natural language report: {str(e)}")
        print("💡 JSON analysis was still saved successfully")


def main():
    """Main function to handle command line arguments and run analysis"""
    parser = argparse.ArgumentParser(
        description="Financial Agent - Find similar historical patterns for stocks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_analysis.py AAPL                    # Analyze Apple with default settings
  python run_analysis.py MSFT --top-k 15         # Show top 15 results for Microsoft
  python run_analysis.py GOOGL --detailed        # Show detailed analysis for Google
  python run_analysis.py AAPL --generate-report  # Generate business report for Apple
  python run_analysis.py --interactive           # Interactive mode with prompts
  python run_analysis.py TSLA --top-k 20 --detailed --generate-report  # Full analysis with report
        """
    )
    
    parser.add_argument(
        'symbol',
        nargs='?',
        help='Stock symbol to analyze (e.g., AAPL, MSFT, GOOGL)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of similar periods to show (default: 10, max: 50)'
    )
    
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed analysis for the top result'
    )
    
    parser.add_argument(
        '--generate-report',
        action='store_true',
        help='Generate a business-friendly natural language report'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode with prompts'
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    # Interactive mode
    if args.interactive or not args.symbol:
        symbol, top_k, detailed, generate_report = get_user_input()
        run_analysis(symbol, top_k, detailed, generate_report)
    else:
        # Command line mode
        symbol = args.symbol.upper()
        top_k = min(args.top_k, 50)  # Cap at 50
        run_analysis(symbol, top_k, args.detailed, args.generate_report)
    
    print("\n" + "=" * 60)
    print("🎉 Analysis Complete! Thank you for using Financial Agent.")
    print("=" * 60)


if __name__ == "__main__":
    main() 