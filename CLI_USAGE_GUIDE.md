# 🚀 Financial Agent - CLI Usage Guide

## Quick Start Examples

### 🎯 Basic Analysis
```bash
python run_analysis.py AAPL
```
Analyzes Apple stock with default settings (top 10 similar periods).

### 📊 Custom Number of Results
```bash
python run_analysis.py MSFT --top-k 15
```
Shows top 15 similar periods for Microsoft.

### 🔍 Detailed Analysis
```bash
python run_analysis.py GOOGL --detailed
```
Shows detailed breakdown of the best match for Google.

### ⚙️ Interactive Mode
```bash
python run_analysis.py --interactive
```
Guided prompts for stock symbol and analysis options.

### 🎛️ Full Custom Analysis
```bash
python run_analysis.py TSLA --top-k 20 --detailed
```
Tesla analysis with 20 results and detailed breakdown.

## Available Options

| Option | Description | Example |
|--------|-------------|---------|
| `symbol` | Stock symbol to analyze (required if not using --interactive) | `AAPL`, `MSFT`, `GOOGL` |
| `--top-k N` | Number of similar periods to show (1-50, default: 10) | `--top-k 15` |
| `--detailed` | Show detailed analysis for top match | `--detailed` |
| `--interactive` | Run in interactive mode with prompts | `--interactive` |
| `--help` | Show help message | `--help` |

## What You'll See

### 🎯 Target Analysis
- Current 7-day pattern date range
- Market regime classification (RSI zone, volatility, trend)
- Technical indicator values

### 📈 Search Summary
- Total historical windows analyzed
- Number of windows after filtering
- Similar patterns found above threshold
- Processing time

### 🔍 Similar Periods Table
- Ranked list of similar historical periods
- Similarity scores (0-1 scale)
- Similarity levels (Very High, High, Medium-High, etc.)
- Date ranges for each similar period

### 📊 Detailed Analysis (Optional)
- In-depth breakdown of the top match
- Similarity score and level
- Feature vector information

## Examples by Stock

### Tech Stocks
```bash
python run_analysis.py AAPL --top-k 10
python run_analysis.py MSFT --detailed
python run_analysis.py GOOGL --top-k 15
python run_analysis.py AMZN
```

### Growth Stocks
```bash
python run_analysis.py TSLA --top-k 20
python run_analysis.py NVDA --detailed
python run_analysis.py META --top-k 12
```

### Traditional Stocks
```bash
python run_analysis.py JNJ --top-k 8
python run_analysis.py KO --detailed
python run_analysis.py DIS --top-k 10
```

## Tips for Best Results

### ✅ Good Practices
- Use well-known, actively traded stocks
- Allow processing time (can take 30 seconds - 2 minutes)
- Check saved JSON files for detailed data
- Use --top-k to limit results for faster processing

### ⚠️ Troubleshooting
- **"No similar periods found"**: Try a different stock or lower similarity threshold
- **"Insufficient data"**: Stock might be too new or have data gaps
- **Network errors**: Check internet connection, try again in a few minutes
- **Processing very slow**: Reduce --top-k value or try a different stock

## Output Files

The system automatically saves results to timestamped JSON files:
```
analysis_AAPL_20231210_143052.json
analysis_TSLA_20231210_143125.json
```

These files contain complete analysis data for further processing or review.

## Interactive Mode Guide

When using `--interactive`, you'll see:

1. **Stock Symbol Prompt**: Enter any valid stock symbol
2. **Analysis Options Menu**:
   - Quick Analysis (Top 5 results)
   - Standard Analysis (Top 10 results) 
   - Detailed Analysis (Top 20 results)
   - Custom (specify your preferences)

Example interactive session:
```
🔤 Enter Stock Symbol (e.g., AAPL, MSFT, GOOGL):
Symbol: AAPL

⚙️  Analysis Options:
1. Quick Analysis (Top 5 results)
2. Standard Analysis (Top 10 results)
3. Detailed Analysis (Top 20 results)
4. Custom

Choose option (1-4): 2
```

## Performance Notes

- **Processing Time**: 30 seconds to 2 minutes depending on stock and data size
- **Memory Usage**: Efficient handling of large historical datasets
- **Accuracy**: Uses 62-feature vectors for comprehensive pattern matching
- **Filtering**: Smart market regime filtering for relevant comparisons

Enjoy using the Financial Agent CLI! 🎉 