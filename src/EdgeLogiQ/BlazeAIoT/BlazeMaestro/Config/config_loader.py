"""
Configuration Loader Module

Implements Single Responsibility Principle (SRP) by handling only configuration loading and validation.
"""

import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from BlazeAIoT.Core.Utils import BlazeLogger


class ConfigLoader:
    """
    Handles configuration loading and validation.
    
    Responsibilities:
    - Load YAML configuration files
    - Validate configuration structure
    - Provide typed access to configuration sections
    """
    
    def __init__(self, config_path: str):
        """
        Initialize ConfigLoader.
        
        Args:
            config_path: Path to YAML configuration file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        self._logger = BlazeLogger.get_instance()
        self.config_path = Path(config_path)
        self._config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if not isinstance(config, dict):
                raise ValueError("Configuration must be a dictionary")
            
            self._logger.debug(f"Loaded configuration from {self.config_path}")
            return config
        
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def _validate_config(self) -> None:
        """Validate required configuration sections."""
        required_sections = ['environment', 'orchestrator']
        
        for section in required_sections:
            if section not in self._config:
                self._logger.warning(f"Missing configuration section: {section}")
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get complete configuration dictionary."""
        return self._config
    
    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment configuration section."""
        return self._config.get('environment', {})
    
    def get_orchestrator_config(self) -> Dict[str, Any]:
        """Get orchestrator configuration section."""
        return self._config.get('orchestrator', {})
    
    def get_services_config(self) -> Dict[str, Any]:
        """Get services configuration section."""
        return self._config.get('services', {})
    
    def get_clusters_config(self) -> list:
        """Get clusters configuration list."""
        # Try new location first: environment.kubernetes.clusters
        env_config = self.get_environment_config()
        k8s_config = env_config.get('kubernetes', {})
        clusters = k8s_config.get('clusters', [])
        
        # Fallback to legacy location: top-level clusters
        if not clusters:
            clusters = self._config.get('clusters', [])
        
        return clusters
    
    def get_kubernetes_config(self) -> Dict[str, Any]:
        """Get Kubernetes configuration section."""
        # Try new location: environment.kubernetes
        env_config = self.get_environment_config()
        k8s_config = env_config.get('kubernetes', {})
        
        # Fallback to legacy location: top-level kubernetes
        if not k8s_config:
            k8s_config = self._config.get('kubernetes', {})
        
        return k8s_config
    
    def get_metrics_config(self) -> Dict[str, Any]:
        """Get metrics configuration section."""
        return self._config.get('metrics', {})
    
    def get_environment_mode(self) -> str:
        """
        Get environment mode (kubernetes or simulator).
        
        Returns:
            'kubernetes' or 'simulator'
        """
        return self.get_environment_config().get('mode', 'kubernetes')
    
    def is_simulator_mode(self) -> bool:
        """Check if running in simulator mode."""
        return self.get_environment_mode() == 'simulator'
    
    def is_kubernetes_mode(self) -> bool:
        """Check if running in Kubernetes mode."""
        return self.get_environment_mode() == 'kubernetes'
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self._config.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Support dictionary-style access."""
        return self._config[key]
    
    def __contains__(self, key: str) -> bool:
        """Support 'in' operator."""
        return key in self._config
