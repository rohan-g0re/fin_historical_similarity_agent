#!/usr/bin/env python3
"""
Natural Language Report Generator Script
Generate business-friendly reports from existing JSON analysis files.

Usage:
    python generate_report.py analysis_MMM_20250611_143413.json
    python generate_report.py analysis_AAPL_20250611_143413.json --output custom_report.txt
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.reports.natural_language_generator import generate_report_from_json


def main():
    """Main function to handle command line arguments and generate reports"""
    parser = argparse.ArgumentParser(
        description="Generate business-friendly natural language reports from JSON analysis files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_report.py analysis_MMM_20250611_143413.json
  python generate_report.py analysis_AAPL_20250611_143413.json --output apple_report.txt
  python generate_report.py my_analysis.json --output reports/detailed_analysis.txt
        """
    )
    
    parser.add_argument(
        'json_file',
        help='Path to the JSON analysis file'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output file path for the report (if not specified, uses auto-generated name)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.json_file).exists():
        print(f"❌ Error: File '{args.json_file}' does not exist")
        sys.exit(1)
    
    # Determine output file path
    if args.output:
        output_path = args.output
    else:
        # Auto-generate output filename
        input_path = Path(args.json_file)
        output_path = str(input_path.with_suffix('')) + '_BUSINESS_REPORT.txt'
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("📝 NATURAL LANGUAGE REPORT GENERATOR")
    print("=" * 60)
    print(f"📂 Input file: {args.json_file}")
    print(f"📄 Output file: {output_path}")
    print()
    
    try:
        # Generate the report
        print("🔄 Generating business-friendly report...")
        report = generate_report_from_json(args.json_file, output_path)
        
        if report and not report.startswith("Error"):
            print("✅ Report generated successfully!")
            print()
            
            # Show a preview of the report
            print("📋 REPORT PREVIEW:")
            print("─" * 60)
            lines = report.split('\n')
            preview_lines = []
            for line in lines:
                preview_lines.append(line)
                if len(preview_lines) >= 20:  # Show first 20 lines
                    break
            
            print('\n'.join(preview_lines))
            print("─" * 60)
            print(f"📖 Full report available in: {output_path}")
            print()
            print("🎯 The report includes:")
            print("   • Executive Summary with key findings")
            print("   • Current Market Analysis")
            print("   • Historical Pattern Analysis")
            print("   • Risk Assessment")
            print("   • Future Outlook")
            print("   • Detailed Historical Comparisons")
            print("   • Technical Summary")
        else:
            print("❌ Failed to generate report")
            if report:
                print(f"Error details: {report}")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)
    
    print()
    print("=" * 60)
    print("🎉 Report Generation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main() 