"""
Tools package for the voice assistant.

This package contains all the tools that can be used by the voice assistant.
All tools are automatically registered with the tool registry when this
package is imported.
"""

# Import all tools to ensure they are registered with the tool registry
from . import spotify
from . import lighting
from . import weather_time
from . import google_calendar
from . import airtouch

# Import the tool registry
from .tool_registry import tool_registry, tool

__all__ = [
    'tool_registry',
    'tool',
    'spotify',
    'lighting', 
    'weather_time',
    'google_calendar',
    'airtouch'
] 