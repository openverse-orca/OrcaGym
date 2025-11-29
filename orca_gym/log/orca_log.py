"""
OrcaLog - Advanced logging utility for OrcaGym project

This module provides a comprehensive logging solution with the following features:
1. Time logging with millisecond precision (YYYY-MM-DD HH:mm:ss.SSS)
2. Function, file, and line number tracking
3. Multiple log levels: FATAL, ERROR, WARNING, INFO, DEBUG
4. File rotation based on size and count
5. Configurable output levels for console and file
6. Beautiful formatted output with colors
"""

import logging
from logging.handlers import RotatingFileHandler
import sys
import os
import inspect
from datetime import datetime
from typing import Optional, Union

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Custom log level for performance logging (between INFO and WARNING)
PERFORMANCE = 25
logging.addLevelName(PERFORMANCE, 'PERFORMANCE')

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',       # Cyan
        'INFO': '\033[32m',        # Green
        'PERFORMANCE': '\033[34m', # Blue
        'WARNING': '\033[33m',     # Yellow
        'ERROR': '\033[31m',       # Red
        'CRITICAL': '\033[35m',    # Magenta
        'RESET': '\033[0m'         # Reset
    }
    
    def __init__(self, *args, use_colors=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_colors = use_colors
    
    # Width for level name alignment (matches PERFORMANCE length)
    LEVEL_WIDTH = 11
    
    def format(self, record):
        original = super().format(record)
        
        if self.use_colors and hasattr(record, 'levelname'):
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            reset = self.COLORS['RESET']
            # Replace level name with colored version, maintaining alignment
            import re
            pattern = rf'\] {re.escape(record.levelname)}\s+\|'
            # Pad to LEVEL_WIDTH for alignment
            padded_level = record.levelname.ljust(self.LEVEL_WIDTH)
            replacement = f'] {color}{padded_level}{reset} |'
            original = re.sub(pattern, replacement, original)
        
        return original


class OrcaLog:
    """
    Advanced logging class for OrcaGym project.
    
    Features:
    - Precise timestamp logging (with milliseconds)
    - Function/file/line tracking
    - Multiple log levels
    - File rotation (size and count based)
    - Configurable output levels
    - Singleton pattern for global access
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super(OrcaLog, cls).__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        name: str = "OrcaGym",
        log_file: str = "orca_gym.log",
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        console_level: str = "WARNING",
        file_level: str = "DEBUG",
        log_dir: str = f"{PROJECT_ROOT}/logs",
        use_colors: bool = True,
        force_reinit: bool = False
    ):
        """
        Initialize OrcaLog instance.
        
        Args:
            name: Logger name
            log_file: Log file name
            max_bytes: Maximum file size before rotation (bytes)
            backup_count: Maximum number of backup files to keep
            console_level: Console output level (DEBUG, INFO, PERFORMANCE, WARNING,  ERROR, CRITICAL)
            file_level: File output level (DEBUG, INFO, PERFORMANCE, WARNING,  ERROR, CRITICAL)
            log_dir: Directory to store log files
            use_colors: Whether to use colors in console output (file output is always plain text)
            force_reinit: Force reinitialize even if already initialized
        """
        # Only initialize if not already initialized or force_reinit is True
        if not self._initialized or force_reinit:
            self.name = name
            self.log_file = log_file
            self.max_bytes = max_bytes
            self.backup_count = backup_count
            self.console_level = console_level.upper()
            self.file_level = file_level.upper()
            self.log_dir = log_dir
            self.use_colors = use_colors
            
            # Create log directory if it doesn't exist
            os.makedirs(self.log_dir, exist_ok=True)
            
            # Full path to log file
            self.log_path = os.path.join(self.log_dir, self.log_file)
            
            # Initialize logger
            self._setup_logger()
            
            self._initialized = True
    
    def _setup_logger(self):
        """Setup the logger with handlers and formatters."""
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Create formatter with detailed information and better alignment
        # Format: [TIME] LEVEL | FUNCTION | FILE:LINE | MESSAGE
        # LEVEL width is 11 to accommodate "PERFORMANCE"
        log_format = '[%(asctime)s.%(msecs)03d] %(levelname)-11s | %(funcName)-20s | %(filename)s:%(lineno)-4d | %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
        
        # File formatter (plain text for IDE plugin highlighting)
        file_formatter = logging.Formatter(log_format, datefmt=date_format)
        
        # Console formatter with colors
        console_formatter = ColoredFormatter(log_format, datefmt=date_format, use_colors=self.use_colors)
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            self.log_path,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, self.file_level))
        file_handler.setFormatter(file_formatter)
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self.console_level))
        console_handler.setFormatter(console_formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _log_with_caller(self, level: int, message: str):
        """Log a message with the actual caller information."""
        # Get the caller's frame (skip this method and the public method)
        frame = inspect.currentframe()
        try:
            # Go up 2 levels: _log_with_caller -> public method -> actual caller
            caller_frame = frame.f_back.f_back
            caller_info = inspect.getframeinfo(caller_frame)
            
            # Create a log record with caller information
            record = self.logger.makeRecord(
                self.logger.name, level, caller_info.filename, 
                caller_info.lineno, message, (), None
            )
            record.funcName = caller_info.function
            record.filename = caller_info.filename
            
            # Handle the record
            self.logger.handle(record)
        finally:
            del frame
    
    def fatal(self, message: str):
        """Log a FATAL level message."""
        self._log_with_caller(logging.CRITICAL, message)
    
    def error(self, message: str):
        """Log an ERROR level message."""
        self._log_with_caller(logging.ERROR, message)
    
    def warning(self, message: str):
        """Log a WARNING level message."""
        self._log_with_caller(logging.WARNING, message)
    
    def info(self, message: str):
        """Log an INFO level message."""
        self._log_with_caller(logging.INFO, message)
    
    def debug(self, message: str):
        """Log a DEBUG level message."""
        self._log_with_caller(logging.DEBUG, message)
    
    def performance(self, message: str):
        """Log a PERFORMANCE level message."""
        self._log_with_caller(PERFORMANCE, message)
    
    def set_console_level(self, level: str):
        """Set console output level."""
        self.console_level = level.upper()
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, RotatingFileHandler):
                handler.setLevel(getattr(logging, self.console_level))
    
    def set_file_level(self, level: str):
        """Set file output level."""
        self.file_level = level.upper()
        for handler in self.logger.handlers:
            if isinstance(handler, RotatingFileHandler):
                handler.setLevel(getattr(logging, self.file_level))
    
    def get_log_info(self) -> dict:
        """Get current logging configuration information."""
        return {
            'name': self.name,
            'log_file': self.log_path,
            'max_bytes': self.max_bytes,
            'backup_count': self.backup_count,
            'console_level': self.console_level,
            'file_level': self.file_level,
            'log_dir': self.log_dir,
            'use_colors': self.use_colors
        }
    
    def log_system_info(self):
        """Log system information for debugging."""
        import platform
        import psutil
        
        self.info("=== System Information ===")
        self.info(f"Platform: {platform.platform()}")
        self.info(f"Python Version: {platform.python_version()}")
        self.info(f"CPU Count: {psutil.cpu_count()}")
        self.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")
        self.info("==========================")
    
    @classmethod
    def get_instance(cls, **kwargs) -> 'OrcaLog':
        """
        Get the singleton instance of OrcaLog.
        
        Args:
            **kwargs: Optional parameters to initialize the logger if not already initialized
            
        Returns:
            OrcaLog singleton instance
        """
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        else:
            # If instance exists but kwargs are provided, update the configuration
            if kwargs:
                cls._instance.__init__(**kwargs)
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (useful for testing)."""
        if cls._instance is not None:
            # Clear handlers to avoid memory leaks
            if hasattr(cls._instance, 'logger'):
                for handler in cls._instance.logger.handlers[:]:
                    cls._instance.logger.removeHandler(handler)
                    handler.close()
        cls._instance = None
        cls._initialized = False
    
    def is_initialized(self) -> bool:
        """Check if the logger is initialized."""
        return self._initialized


# Convenience function to get the singleton logger
def get_orca_logger(
    name: str = "OrcaGym",
    log_file: str = "orca_gym.log",
    **kwargs
) -> OrcaLog:
    """
    Get the singleton OrcaLog instance.
    
    Args:
        name: Logger name (only used if not already initialized)
        log_file: Log file name (only used if not already initialized)
        **kwargs: Additional arguments for OrcaLog constructor (only used if not already initialized)
    
    Returns:
        OrcaLog singleton instance
    """
    # 将 name 和 log_file 也加入到 kwargs 中，确保所有参数都能传递
    kwargs['name'] = name
    kwargs['log_file'] = log_file
    return OrcaLog.get_instance(**kwargs)

# Example usage
if __name__ == "__main__":
    # Test singleton pattern
    print("=== Testing Singleton Pattern ===")
    
    # Create first logger instance
    # logger1 = OrcaLog(
    #     name="OrcaGymTest",
    #     log_file="test.log",
    #     max_bytes=5*1024*1024,  # 5MB
    #     backup_count=3,
    #     console_level="DEBUG",
    #     file_level="INFO"
    # )
    
    # # Create second logger instance (should be the same as first)
    # logger2 = OrcaLog(name="DifferentName", log_file="different.log")
    
    logger1 = OrcaLog.get_instance()
    logger2 = OrcaLog.get_instance()

    # Test that they are the same instance
    print(f"logger1 is logger2: {logger1 is logger2}")
    print(f"logger1 name: {logger1.name}")
    print(f"logger2 name: {logger2.name}")
    
    # Test get_instance method
    logger3 = OrcaLog.get_instance()
    print(f"logger1 is logger3: {logger1 is logger3}")
    
    # Test convenience function
    logger4 = get_orca_logger()
    print(f"logger1 is logger4: {logger1 is logger4}")
    
    def example_function():
        """Example function to demonstrate logging."""
        val = 6
        logger1.debug(f"This is a debug message {val}")
        logger1.info("This is an info message")
        logger1.performance("This is a performance message")
        logger1.warning("This is a warning message")
        logger1.error("This is an error message")
        logger1.fatal("This is a fatal message")
    
    def another_function():
        """Another function to show different caller info."""
        logger1.info("Message from another function")
        logger1.debug("Debug from another function")
    
    # Test logging
    logger1.info("Starting OrcaLog singleton test")
    example_function()
    another_function()
    logger1.info("OrcaLog singleton test completed")
    
    # Print configuration
    print("\nLogger Configuration:")
    for key, value in logger1.get_log_info().items():
        print(f"  {key}: {value}")
    
    # Test reset functionality
    print("\n=== Testing Reset Functionality ===")
    OrcaLog.reset_instance()
    logger5 = OrcaLog(name="NewLogger", log_file="new.log")
    print(f"New logger name: {logger5.name}")
    print(f"logger1 is logger5: {logger1 is logger5}")
