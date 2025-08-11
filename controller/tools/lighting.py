"""
Philips Hue lighting control tool using the centralized tool registry.

This module provides lighting control functionality with proper
schema definitions and function calling support.
"""

from phue import Bridge
import os
import json
from typing import Optional

from .tool_registry import tool, tool_registry

b = Bridge('192.168.1.100')

# Load light names and groups from JSON file
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'light_names.json'), 'r') as f:
    light_names = json.load(f)


@tool(
    name="turn_on_lights",
    description="Turn on lights in a specific location",
    aliases=["lights_on", "switch_on_lights", "turn_on"]
)
def turn_on_lights(location: str = "Downlights Office") -> str:
    """
    Turn on lights in the specified location.
    
    Args:
        location: The location/room name for the lights
        
    Returns:
        Status message about the action
    """
    location = location.title()  # First Letters Capitalized
    if location in light_names['lights']:
        b.set_light(location, 'on', True)
        return f"{location} lights on"
    elif location in light_names['groups']:
        b.set_group(location, 'on', True)
        return f"{location} on"
    else:
        return f"No lights or rooms with name {location}"


@tool(
    name="turn_off_lights",
    description="Turn off lights in a specific location",
    aliases=["lights_off", "switch_off_lights", "turn_off"]
)
def turn_off_lights(location: str = "Downlights Office") -> str:
    """
    Turn off lights in the specified location.
    
    Args:
        location: The location/room name for the lights
        
    Returns:
        Status message about the action
    """
    location = location.title()
    if location in light_names['lights']:
        b.set_light(location, 'on', False)
        return f"{location} lights off"
    elif location in light_names['groups']:
        b.set_group(location, 'on', False)
        return f"{location} off"
    else:
        return f"No lights or rooms with name {location}"


@tool(
    name="set_brightness",
    description="Set brightness level for lights in a specific location",
    aliases=["brightness", "dim_lights", "brighten_lights"]
)
def set_brightness(percent: int = 100, location: str = "Downlights Office") -> str:
    """
    Set brightness level for lights in the specified location.
    
    Args:
        percent: Brightness percentage (0-100)
        location: The location/room name for the lights
        
    Returns:
        Status message about the action
    """
    location = location.title()
    if location in light_names['lights']:
        b.set_light(location, 'on', True)
        level = int((int(percent) / 100) * 254)
        b.set_light(location, 'bri', level)
        return f"{location} lights set to {percent} percent."
    elif location in light_names['groups']:
        b.set_group(location, 'on', True)
        level = int((int(percent) / 100) * 254)
        b.set_group(location, 'bri', level)
        return f"{location} set to {percent} percent."
    else:
        return f"No lights or rooms with name {location}"


if __name__ == "__main__":
    print("Philips Hue Lighting Controller")
    
    # Print available tools
    print("\nAvailable tools:")
    for schema in tool_registry.get_all_schemas():
        print(f"  {schema.name}: {schema.description}")
        for param in schema.parameters:
            print(f"    - {param.name} ({param.type.value}): {param.description}")
    
    # Test function calling
    print("\nTesting function calling:")
    result = tool_registry.execute_tool("turn_on_lights", kwargs={"location": "kitchen"})
    print(f"Result: {result}")



