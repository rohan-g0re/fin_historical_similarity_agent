"""
Configuration Manager for Financial Agent

Handles loading and accessing configuration settings from YAML files.
Provides centralized configuration management for the entire application.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    """
    Centralized configuration manager for the financial agent.
    
    Loads configuration from YAML files and provides easy access to settings
    with proper type handling and default values.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path (str, optional): Path to config file. 
                                       Defaults to 'config.yaml' in project root.
        """
        if config_path is None:
            # Get project root directory (assuming this file is in src/core/)
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                return config if config is not None else {}
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in config file: {e}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path (str): Dot-separated path to config value (e.g., 'data_collection.start_date')
            default (Any): Default value if key doesn't exist
            
        Returns:
            Any: Configuration value or default
            
        Example:
            >>> config.get('indicators.rsi.period', 14)
            14
        """
        keys = key_path.split('.')
        current = self.config
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def get_data_collection_config(self) -> Dict[str, Any]:
        """Get data collection configuration section."""
        return self.get('data_collection', {})
    
    def get_indicators_config(self) -> Dict[str, Any]:
        """Get technical indicators configuration section."""
        return self.get('indicators', {})
    
    def get_window_config(self) -> Dict[str, Any]:
        """Get 7-day window configuration section."""
        return self.get('window_settings', {})
    
    def get_similarity_config(self) -> Dict[str, Any]:
        """Get similarity detection configuration section."""
        return self.get('similarity', {})
    
    def get_filtering_config(self) -> Dict[str, Any]:
        """Get basic filtering configuration section."""
        return self.get('filtering', {})
    
    def get_storage_config(self) -> Dict[str, Any]:
        """Get data storage configuration section."""
        return self.get('storage', {})
    
    def get_testing_config(self) -> Dict[str, Any]:
        """Get testing configuration section."""
        return self.get('testing', {})
    
    def reload_config(self) -> None:
        """Reload configuration from file."""
        self.config = self._load_config()
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return f"ConfigManager(config_path={self.config_path})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"ConfigManager(config_path='{self.config_path}', keys={list(self.config.keys())})" 