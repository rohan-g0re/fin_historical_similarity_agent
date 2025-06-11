"""
Reports module for generating natural language reports from financial analysis data.
"""

from .natural_language_generator import NaturalLanguageReportGenerator, generate_report_from_json

__all__ = ['NaturalLanguageReportGenerator', 'generate_report_from_json'] 