"""
BlazeLogger Workaround - Avoids circular import for training scripts

This module provides a mock BlazeLogger that prevents the circular import
issue when running training scripts without the full BlazeAIoT stack.

Simply import this before importing the problematic modules:
    from blaze_logger_workaround import install_logger_workaround
    install_logger_workaround()
    
Then proceed with normal imports.
"""

import sys
import logging
from types import ModuleType


class BlazeLoggerMock:
    """Mock BlazeLogger that provides minimal interface."""
    
    _instance = None
    _logger = logging.getLogger("BlazeLogger")
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @staticmethod
    def get_instance():
        """Get singleton instance."""
        return BlazeLoggerMock()
    
    def info(self, msg, *args, **kwargs):
        BlazeLoggerMock._logger.info(msg)
    
    def debug(self, msg, *args, **kwargs):
        BlazeLoggerMock._logger.debug(msg)
    
    def warning(self, msg, *args, **kwargs):
        BlazeLoggerMock._logger.warning(msg)
    
    def error(self, msg, *args, **kwargs):
        BlazeLoggerMock._logger.error(msg)
    
    def critical(self, msg, *args, **kwargs):
        BlazeLoggerMock._logger.critical(msg)


def install_logger_workaround():
    """
    Install mock BlazeLogger to avoid circular import.
    
    Call this BEFORE importing any BlazeMaestro modules.
    """
    # Create a mock BlazeAIoT.Core.Utils module
    mock_utils = ModuleType('BlazeAIoT.Core.Utils')
    mock_utils.BlazeLogger = BlazeLoggerMock
    sys.modules['BlazeAIoT.Core.Utils'] = mock_utils
    
    # Create a mock BlazeAIoT.Core.Utils.blaze_logger module
    mock_logger_module = ModuleType('BlazeAIoT.Core.Utils.blaze_logger')
    mock_logger_module.BlazeLogger = BlazeLoggerMock
    sys.modules['BlazeAIoT.Core.Utils.blaze_logger'] = mock_logger_module
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)-8s | %(name)s: %(message)s'
    )
    
    # Suppress verbose WiFi and scenario logs
    logging.getLogger('WiFiScenarioManager').setLevel(logging.WARNING)
    logging.getLogger('wifi_mobility_controller').setLevel(logging.WARNING)
    logging.getLogger('WiFiOrchestrationAdapter').setLevel(logging.WARNING)


__all__ = ['install_logger_workaround', 'BlazeLoggerMock']
