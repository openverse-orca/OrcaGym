"""
OrcaGym Logging Module

This module provides logging functionality for the OrcaGym project.
"""

try:
    from .orca_log import OrcaLog, PERFORMANCE
    __all__ = ['OrcaLog', 'PERFORMANCE']
except ImportError:
    # Fallback for development
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from orca_log import OrcaLog, PERFORMANCE
    __all__ = ['OrcaLog', 'PERFORMANCE']
