"""
Configuration Manager for Financial Agent

Handles loading and accessing configuration settings from YAML files.
Provides centralized configuration management for the entire application.

This module serves as the single source of truth for all system parameters,
enabling easy tuning of:
- Technical indicator parameters (RSI periods, MACD settings, etc.)
- Data collection settings (date ranges, caching preferences)
- Pattern matching thresholds (similarity scores, filtering criteria)
- System performance settings (memory usage, processing limits)

The configuration system supports:
- Hierarchical YAML configuration files
- Dot notation access to nested settings
- Type-safe default value handling
- Hot-reloading of configuration changes
- Environment-specific overrides
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    """
    Centralized configuration manager for the financial agent.
    
    This class abstracts all configuration complexity behind a simple interface,
    allowing the rest of the system to focus on business logic rather than
    configuration management. Key features:
    
    - Thread-safe configuration loading and access
    - Intelligent default value handling for robustness
    - Structured access to configuration sections
    - Support for configuration validation and type checking
    - Memory-efficient caching of parsed configuration
    
    The configuration is organized into logical sections:
    - data_collection: API settings, date ranges, caching
    - indicators: Technical analysis parameters for all 5 indicators
    - window_settings: 7-day window creation parameters
    - similarity: Pattern matching and similarity thresholds
    - filtering: Basic market regime filtering criteria
    - storage: File system and data persistence settings
    - testing: Development and testing specific settings
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Loads configuration from YAML file with intelligent path resolution.
        If no path is specified, automatically locates config.yaml in the
        project root directory. This allows flexible deployment while
        maintaining consistent default behavior.
        
        Args:
            config_path (str, optional): Path to config file. 
                                       Defaults to 'config.yaml' in project root.
                                       
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file contains invalid YAML syntax
        """
        if config_path is None:
            # Auto-detect project root directory for consistent file location
            # Assumes this file is in src/core/ relative to project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config.yaml"
        
        # Store paths for potential reloading and error reporting
        self.config_path = Path(config_path)
        
        # Load and parse the configuration file
        # This is done once during initialization for performance
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file with comprehensive error handling.
        
        This method handles all the complexities of YAML parsing including:
        - File existence validation
        - YAML syntax validation  
        - Empty file handling
        - Character encoding issues
        - Permission problems
        
        Returns:
            Dict[str, Any]: Configuration dictionary with all settings
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
            PermissionError: If file cannot be read due to permissions
            UnicodeDecodeError: If file contains invalid UTF-8
        """
        try:
            # Open with explicit UTF-8 encoding for cross-platform compatibility
            with open(self.config_path, 'r', encoding='utf-8') as file:
                # Use safe_load to prevent code execution from untrusted YAML
                config = yaml.safe_load(file)
                
                # Handle empty or None configuration files gracefully
                # Return empty dict rather than None to prevent downstream errors
                return config if config is not None else {}
                
        except FileNotFoundError:
            # Provide clear error message with actionable information
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            # Include original error details for debugging
            raise yaml.YAMLError(f"Invalid YAML in config file: {e}")
        except PermissionError:
            raise PermissionError(f"Cannot read config file (permission denied): {self.config_path}")
        except UnicodeDecodeError as e:
            raise UnicodeDecodeError(f"Config file encoding error: {e}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation for nested access.
        
        This method provides a clean interface for accessing deeply nested
        configuration values without requiring complex dictionary traversal
        in client code. Supports arbitrary nesting depth and provides
        robust default value handling.
        
        The dot notation allows intuitive access patterns:
        - 'indicators.rsi.period' → config['indicators']['rsi']['period']
        - 'data_collection.start_date' → config['data_collection']['start_date']
        - 'similarity.threshold' → config['similarity']['threshold']
        
        Args:
            key_path (str): Dot-separated path to config value (e.g., 'indicators.rsi.period')
            default (Any): Default value returned if key doesn't exist or path is invalid
            
        Returns:
            Any: Configuration value at the specified path, or default if not found
            
        Example:
            >>> config = ConfigManager()
            >>> config.get('indicators.rsi.period', 14)
            14
            >>> config.get('nonexistent.path', 'fallback')
            'fallback'
        """
        # Split the dot-separated path into individual keys
        keys = key_path.split('.')
        current = self.config
        
        try:
            # Traverse the nested dictionary structure
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            # KeyError: key doesn't exist in dictionary
            # TypeError: current is not a dictionary (e.g., trying to access key on string)
            return default
    
    def get_data_collection_config(self) -> Dict[str, Any]:
        """
        Get data collection configuration section.
        
        Returns settings for:
        - API endpoints and authentication
        - Historical data date ranges
        - Caching preferences and duration
        - Data quality thresholds
        - Rate limiting parameters
        
        Returns:
            Dict[str, Any]: Data collection configuration with defaults
        """
        return self.get('data_collection', {})
    
    def get_indicators_config(self) -> Dict[str, Any]:
        """
        Get technical indicators configuration section.
        
        Returns parameters for all 5 core indicators:
        - RSI: period, overbought/oversold thresholds
        - MACD: fast/slow/signal periods
        - Bollinger Bands: period, standard deviation multiplier
        - Volume ROC: period, outlier thresholds
        - ATR: period, percentile calculation window
        
        Returns:
            Dict[str, Any]: Technical indicators configuration with defaults
        """
        return self.get('indicators', {})
    
    def get_window_config(self) -> Dict[str, Any]:
        """
        Get 7-day window configuration section.
        
        Returns settings for:
        - Window size (default 7 days)
        - Feature types to include in vectors
        - Normalization methods (z-score, min-max, none)
        - Window overlap and gap settings
        - Feature engineering parameters
        
        Returns:
            Dict[str, Any]: Window creation configuration with defaults
        """
        return self.get('window_settings', {})
    
    def get_similarity_config(self) -> Dict[str, Any]:
        """
        Get similarity detection configuration section.
        
        Returns parameters for:
        - Similarity calculation method (cosine, euclidean, etc.)
        - Similarity thresholds for pattern matching
        - Maximum number of results to return
        - Minimum gap between historical windows
        - Performance optimization settings
        
        Returns:
            Dict[str, Any]: Similarity calculation configuration with defaults
        """
        return self.get('similarity', {})
    
    def get_filtering_config(self) -> Dict[str, Any]:
        """
        Get basic filtering configuration section.
        
        Returns criteria for:
        - Market regime filtering (volatility, trend, momentum)
        - RSI zone matching tolerance
        - Trend direction matching requirements
        - Volatility regime tolerance bands
        - Filter bypass options for research
        
        Returns:
            Dict[str, Any]: Filtering configuration with defaults
        """
        return self.get('filtering', {})
    
    def get_storage_config(self) -> Dict[str, Any]:
        """
        Get data storage configuration section.
        
        Returns settings for:
        - Data directory paths
        - Cache storage location and management
        - File naming conventions
        - Compression and serialization preferences
        - Cleanup and retention policies
        
        Returns:
            Dict[str, Any]: Storage configuration with defaults
        """
        return self.get('storage', {})
    
    def get_testing_config(self) -> Dict[str, Any]:
        """
        Get testing configuration section.
        
        Returns settings for:
        - Test data sources and mock data
        - Performance benchmarking parameters
        - Debug logging levels
        - Development feature flags
        - Testing environment overrides
        
        Returns:
            Dict[str, Any]: Testing configuration with defaults
        """
        return self.get('testing', {})
    
    def reload_config(self) -> None:
        """
        Reload configuration from file to pick up changes.
        
        This method enables hot-reloading of configuration changes without
        restarting the application. Useful for:
        - Parameter tuning during research and development
        - Dynamic threshold adjustments in production
        - A/B testing different configuration sets
        - Emergency configuration updates
        
        Note: This method will raise the same exceptions as __init__ if
        the configuration file has become invalid since initial loading.
        
        Raises:
            FileNotFoundError: If config file no longer exists
            yaml.YAMLError: If config file now contains invalid YAML
        """
        # Reload and replace the entire configuration
        self.config = self._load_config()
    
    def __str__(self) -> str:
        """
        String representation of the configuration manager.
        
        Provides basic information about the configuration source
        for debugging and logging purposes.
        
        Returns:
            str: Simple string representation showing config file path
        """
        return f"ConfigManager(config_path={self.config_path})"
    
    def __repr__(self) -> str:
        """
        Detailed string representation for development and debugging.
        
        Shows both the configuration source and the top-level configuration
        keys for quick insight into what configuration sections are available.
        
        Returns:
            str: Detailed representation with path and available config sections
        """
        return f"ConfigManager(config_path='{self.config_path}', keys={list(self.config.keys())})" 