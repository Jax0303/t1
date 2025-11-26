"""
Configuration loading and management utilities.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)


class Config:
    """
    Configuration container for HierTable-RAG settings.
    
    Provides type-safe access to configuration values
    with sensible defaults.
    """

    def __init__(self, config_dict: Dict[str, Any]) -> None:
        """
        Initialize configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
        """
        self._config = config_dict
        self._load_sections()

    def _load_sections(self) -> None:
        """Load configuration sections as attributes."""
        self.api_keys = self._config.get("api_keys", {})
        self.models = self._config.get("models", {})
        self.paths = self._config.get("paths", {})
        self.extraction = self._config.get("extraction", {})
        self.processing = self._config.get("processing", {})
        self.retrieval = self._config.get("retrieval", {})
        self.generation = self._config.get("generation", {})
        self.logging = self._config.get("logging", {})
        self.experiments = self._config.get("experiments", {})
        self.performance = self._config.get("performance", {})

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key path.
        
        Args:
            key: Dot-separated key path (e.g., "models.embedding.name")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value if value is not None else default

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self._config[key]

    def __contains__(self, key: str) -> bool:
        """Check if key exists in config."""
        return key in self._config


def load_config(config_path: Optional[Path] = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, looks for config.yaml
                    in current directory and parent directories.
        
    Returns:
        Config object with loaded settings
        
    Raises:
        FileNotFoundError: If config file is not found
        yaml.YAMLError: If config file is invalid YAML
    """
    if config_path is None:
        # Try to find config.yaml in current and parent directories
        current = Path.cwd()
        for parent in [current] + list(current.parents):
            candidate = parent / "config.yaml"
            if candidate.exists():
                config_path = candidate
                break
        
        if config_path is None:
            raise FileNotFoundError(
                "config.yaml not found. Please create it from config.yaml.template"
            )
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    logger.info(f"Loading configuration from {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    
    if config_dict is None:
        config_dict = {}
    
    return Config(config_dict)


