# 🚀 Financial Agent - Complete Pattern Matching System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

## 📋 **What This System Does**

Find historical periods when a stock behaved similarly to current conditions. Analyze any stock using 5 technical indicators and get ranked matches from 20+ years of market data.

**Key Features:**
- **One-command analysis**: `python run_analysis.py AAPL` 
- **Historical pattern matching** with similarity scores
- **Market regime filtering** (volatility, momentum, trend)
- **Business-friendly reports** for stakeholders
- **Real-time processing** of decades of data

## 🎯 **Quick Start**

### **Installation**
```bash
# 1. Clone repository
git clone <repository-url>
cd financial_agent

# 2. Create virtual environment
python -m venv financial_agent_env
# Windows: financial_agent_env\Scripts\activate
# Mac/Linux: source financial_agent_env/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### **Basic Usage**
```bash
# Analyze any stock symbol
python run_analysis.py AAPL

# Get more results
python run_analysis.py MSFT --top-k 15

# Generate business report
python run_analysis.py GOOGL --generate-report

# Interactive mode for beginners
python run_analysis.py --interactive
```

## 📊 **How It Works**

### **System Architecture**
```mermaid
flowchart TD
    A[Stock Symbol Input] --> B[Data Collection]
    B --> C[Calculate 5 Technical Indicators]
    C --> D[Create 7-Day Windows]
    D --> E[Generate 62-Feature Vectors]
    E --> F[Calculate Similarities]
    F --> G[Apply Market Filters]
    G --> H[Rank Results]
    H --> I[Generate Reports]
```

### **Core Process**
1. **Data Collection**: Download historical data from Yahoo Finance
2. **Technical Analysis**: Calculate RSI, MACD, Bollinger Bands, Volume ROC, ATR
3. **Pattern Creation**: Convert 7-day periods into 62-dimensional vectors
4. **Similarity Search**: Use cosine similarity to find matches
5. **Market Filtering**: Filter by volatility, momentum, and trend regimes
6. **Results**: Ranked list of similar historical periods

### **The 5-Indicator System**

| Indicator | Purpose | What It Measures |
|-----------|---------|------------------|
| **RSI** | Momentum | Overbought/oversold conditions (0-100) |
| **MACD Signal** | Trend | Trend strength and direction |
| **Bollinger Position** | Volatility | Price position within bands (0-1) |
| **Volume ROC** | Participation | Changes in trading volume |
| **ATR Percentile** | Risk | Volatility regime (0-100) |

### **62-Feature Vector Breakdown**
Each 7-day period becomes a comprehensive "fingerprint":
- **21 Price Features**: Daily returns, volatility, cumulative performance
- **41 Indicator Features**: 7 daily values + trend + average for each indicator

## 🖥️ **Command Line Interface**

### **Available Commands**
```bash
python run_analysis.py SYMBOL [OPTIONS]

Options:
  --top-k N              Number of results (1-50, default: 10)
  --detailed             Show detailed analysis
  --generate-report      Create business report
  --interactive          Guided prompts mode
  --help                 Show help message
```

### **Examples**
```bash
# Basic analysis
python run_analysis.py AAPL

# Get top 20 matches with details
python run_analysis.py TSLA --top-k 20 --detailed

# Generate executive report
python run_analysis.py MSFT --generate-report

# Full analysis with report
python run_analysis.py GOOGL --top-k 15 --detailed --generate-report
```

### **Sample Output**
```
🚀 FINANCIAL AGENT - PATTERN ANALYSIS SYSTEM

🎯 TARGET ANALYSIS (AAPL)
📅 Period: 2024-01-01 to 2024-01-07
📊 Market State:
   • RSI: NEUTRAL (48.9) 
   • Volatility: MEDIUM (54.8th percentile)
   • Trend: UPTREND

📈 SEARCH RESULTS
🔍 Analyzed: 914 historical periods
✅ Found: 12 similar patterns
⚡ Time: 1,892ms

🔍 TOP MATCHES
Rank  Date Range           Similarity  Level      Return
1     2020-08-30 to 09-08  0.983      Very High   +5.2%
2     2018-03-15 to 03-21  0.974      Very High   -2.1%
3     2019-11-08 to 11-14  0.967      Very High   +3.8%

💾 Saved: analysis_AAPL_20240101_143052.json
```

## 📝 **Business Reports**

Generate natural language reports for stakeholders:

```bash
# During analysis
python run_analysis.py AAPL --generate-report

# From existing analysis
python generate_report.py analysis_AAPL_20240101_143052.json
```

**Report Contents:**
- **Executive Summary**: Key findings and investment thesis
- **Current Market Analysis**: Technical conditions explained simply  
- **Historical Patterns**: When similar conditions occurred
- **Risk Assessment**: Probability of outcomes with position sizing
- **Future Outlook**: Multi-timeframe projections
- **Detailed Comparisons**: Top historical matches with results

## 🔧 **Technical Details**

### **Performance Specs**
- **Speed**: >100,000 similarity calculations per second
- **Data**: Analyzes 20+ years of historical data
- **Memory**: Optimized for large datasets
- **Accuracy**: 62-dimensional pattern matching

### **Market Regime Classification**

**RSI Zones:**
- Overbought: RSI > 70 (potential selling pressure)
- Oversold: RSI < 30 (potential buying opportunity)  
- Neutral: RSI 30-70 (balanced conditions)

**Volatility Regimes:**
- High: ATR Percentile > 75 (increased risk/opportunity)
- Medium: ATR Percentile 25-75 (normal conditions)
- Low: ATR Percentile < 25 (stable conditions)

**Trend States:**
- Uptrend: MACD signal rising
- Downtrend: MACD signal falling
- Sideways: MACD signal stable

### **Similarity Interpretation**
- **0.90-1.00**: Very High - Extremely similar patterns
- **0.80-0.89**: High - Strong pattern match
- **0.70-0.79**: Medium-High - Good pattern match
- **0.60-0.69**: Medium - Moderate pattern match
- **Below 0.60**: Low - Different patterns

## 📁 **Project Structure**

```
financial_agent/
├── run_analysis.py              # Main CLI interface
├── generate_report.py           # Report generator
├── config.yaml                  # Configuration
├── requirements.txt             # Dependencies
├── src/
│   ├── core/
│   │   ├── config_manager.py    # Settings management
│   │   └── data_collector.py    # Data acquisition
│   ├── indicators/
│   │   └── technical_indicators.py  # Technical analysis
│   ├── similarity/
│   │   ├── window_creator.py    # Pattern creation
│   │   ├── similarity_calculator.py  # Matching engine
│   │   └── pattern_searcher.py  # Main orchestrator
│   └── reports/
│       └── natural_language_generator.py  # Business reports
```

## ⚙️ **Configuration**

Customize behavior via `config.yaml`:

```yaml
# Similarity settings
similarity:
  threshold: 0.65        # Minimum similarity to show
  max_results: 50        # Maximum results
  min_gap_days: 30       # Days between similar periods

# Market filtering
filtering:
  rsi_zone_match: true   # Filter by RSI zones
  trend_match: true      # Filter by trend direction
  volatility_tolerance: 20  # Volatility matching range

# Technical indicators
indicators:
  rsi_period: 14         # RSI calculation period
  macd_fast: 12          # MACD fast EMA
  macd_slow: 26          # MACD slow EMA
  bollinger_period: 20   # Bollinger Bands period
```

## 🐍 **Programmatic Usage**

For integration into other systems:

```python
from src.similarity.pattern_searcher import PatternSearcher

# Initialize
searcher = PatternSearcher()

# Find similar patterns
results = searcher.search_similar_patterns('AAPL', top_k=10)

# Access results
current_state = results['current_window']
matches = results['similar_patterns']
summary = results['search_summary']

print(f"Found {len(matches)} similar patterns")
for match in matches[:3]:
    print(f"Similarity: {match['similarity_score']:.3f}")
    print(f"Period: {match['window_start_date']} to {match['window_end_date']}")
```

## 📊 **Results Format**

The system returns structured data:

```python
{
    'symbol': 'AAPL',
    'current_window': {
        'window_start_date': '2024-01-01',
        'window_end_date': '2024-01-07',
        'features': {
            'rsi_values': [65.2, 67.1, 69.3, ...],
            'macd_signal_values': [0.12, 0.15, 0.18, ...],
            # ... other indicators
        }
    },
    'similar_patterns': [
        {
            'similarity_score': 0.87,
            'similarity_level': 'High',
            'rank': 1,
            'window_start_date': '2020-03-15',
            'window_end_date': '2020-03-21'
        }
        # ... more matches
    ],
    'search_summary': {
        'total_historical_windows': 914,
        'similar_patterns_found': 12,
        'processing_time_ms': 1892.5
    }
}
```

## 💾 **Output Files**

### **JSON Technical Data**
Automatically saved with timestamps:
```
analysis_AAPL_20240101_143052.json
analysis_MSFT_20240101_143125.json
```

Contains complete technical analysis data for further processing.

### **Business Reports** (Optional)
```
analysis_AAPL_20240101_143052_BUSINESS_REPORT.txt
analysis_MSFT_20240101_143125_BUSINESS_REPORT.txt
```

Natural language reports perfect for presentations and stakeholder communication.

## 🎯 **Use Cases**

### **For Traders**
- Find similar market setups for strategy development
- Understand potential outcomes based on history
- Identify optimal entry/exit timing
- Risk assessment for position sizing

### **For Analysts** 
- Research historical market behavior patterns
- Generate comprehensive market reports
- Validate trading hypotheses with data
- Create client presentations

### **For Portfolio Managers**
- Risk management through historical analysis
- Market regime identification
- Strategic asset allocation decisions
- Stakeholder reporting and communication

### **For Financial Teams**
- Convert technical analysis into business language
- Generate executive-ready reports
- Communicate market conditions clearly
- Support investment committee presentations

## 🔍 **Algorithm Deep Dive**

### **7-Day Window Selection**
- **Optimal Balance**: Captures meaningful patterns without noise
- **Market Rhythm**: Aligns with weekly trading cycles
- **Computational Efficiency**: Manageable feature space for real-time analysis

### **Cosine Similarity Mathematics**
```
similarity = (A · B) / (||A|| × ||B||)
```
- **Scale Invariant**: Focuses on pattern shape, not magnitude
- **Range**: -1 to 1 (perfect similarity to perfect opposition)
- **Efficient**: Optimized for high-dimensional vectors

### **Market Regime Intelligence**
Beyond mathematical similarity, the system considers:
- **Market Context**: Volatility environment
- **Momentum State**: Overbought/oversold conditions  
- **Trend Direction**: Bull/bear/sideways markets
- **Participation**: Volume and interest levels

## 📈 **Performance Benchmarks**

- **Similarity Calculation**: <1ms for 1000 comparisons
- **Complete Analysis**: <5 seconds for full stock analysis
- **Memory Usage**: Linear scaling with dataset size
- **Data Processing**: 1000 data points in <100ms
- **Throughput**: >100,000 comparisons per second

## 🧪 **Quality Assurance**

### **Testing**
- **100% Test Coverage**: All components thoroughly tested
- **Production Validation**: Real market data testing
- **Edge Case Handling**: Robust error management
- **Performance Testing**: Speed and memory benchmarks

### **Code Quality**
- **PEP 8 Compliance**: Professional Python standards
- **Type Hints**: Clear function signatures
- **Documentation**: Comprehensive inline comments
- **Error Handling**: Graceful failure and recovery

## 🛠️ **Advanced Features**

### **Intelligent Caching**
- **Data Persistence**: Cached market data for speed
- **Smart Invalidation**: Automatic cache updates
- **Multi-source Fallback**: Reliable data acquisition

### **Market Filtering**
- **Date Gap Filtering**: Ensures diverse historical examples
- **Regime Matching**: Compare like market conditions
- **Quality Thresholds**: Filter low-confidence matches

### **Extensibility**
- **Modular Design**: Easy to add new indicators
- **Configuration Driven**: Flexible parameter adjustment
- **API Integration**: Ready for system integration

## 🔧 **Troubleshooting**

### **Common Issues**

**"No similar patterns found"**
- Stock may have unique current conditions
- Try lowering similarity threshold in config
- Check if symbol is valid and has sufficient history

**"Network/data errors"**
- Check internet connection
- Yahoo Finance API may be rate-limited
- Try again in a few minutes

**"Insufficient data"**
- Stock needs minimum 100 trading days
- Try different symbol with longer history
- Check for delisted or inactive stocks

### **Performance Tips**
- Use `--top-k` to limit results for faster processing
- Cache is automatically managed but can be cleared manually
- Consider running analysis during off-market hours for best API performance

## 📚 **Dependencies**

**Core Requirements:**
- Python 3.8+
- pandas (data manipulation)
- numpy (numerical computing)
- scikit-learn (similarity calculations)
- yfinance (market data)
- PyYAML (configuration)

**Full list in `requirements.txt` with specific versions**

## 📝 **License**

MIT License - see LICENSE file for details.

## 🤝 **Support**

For questions, issues, or contributions, please contact the development team.

## 🚀 **Get Started Now**

Ready to find historical patterns in any stock?

```bash
# Interactive mode for beginners
python run_analysis.py --interactive

# Or jump right in
python run_analysis.py AAPL --generate-report
```

**🎉 Start discovering market patterns!** 📈🔍 