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
from . import thinq
from . import webos

# OPTIONAL: Allow intent AI to ask for external information
# from . import search_web

# Import the tool registry
from .tool_registry import tool_registry, tool

__all__ = [
    'tool_registry',
    'tool',
    'spotify',
    'lighting', 
    'weather_time',
    'google_calendar',
    'airtouch',
    'thinq',
    'webos',
    # 'search_web' # OPTIONAL: Allow for intent to call web search tool
] 