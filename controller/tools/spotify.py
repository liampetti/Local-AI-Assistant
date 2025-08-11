"""
Spotify music control tool using the centralized tool registry.

This module provides Spotify music control functionality with proper
schema definitions and function calling support.
"""

import spotipy
from spotipy.oauth2 import SpotifyOAuth
import json
import os
import re
from typing import Optional

from .tool_registry import tool, tool_registry

# Load credentials from JSON file
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spotify_creds.json'), 'r') as f:
    creds = json.load(f)

SCOPE = 'user-read-playback-state user-modify-playback-state user-read-currently-playing'

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=creds['client_id'],
    client_secret=creds['client_secret'],
    redirect_uri=creds['redirect_uri'],
    scope='user-read-playback-state user-modify-playback-state user-read-currently-playing'
))


def get_active_device():
    """Get the active Spotify device ID."""
    devices = sp.devices()
    for device in devices['devices']:
        if device['name'] == 'RedBox':
            return device['id']
    return None


@tool(
    name="play_song",
    description="Play a song by artist and title, or search for a song by query",
    aliases=["play", "play_music", "start_music"]
)
def play_song(artist_query: Optional[str] = None, song: Optional[str] = None) -> str:
    """
    Play a song on Spotify.
    
    Args:
        artist_query: Artist name or search query
        song: Song title (if two arguments are provided, one is artist and one is song title)
    
    Returns:
        Status message about the played song
    """
    # First check if artist_query can be split into artist and song
    if artist_query and re.search(r'\s+by\s+', artist_query):
        parts = artist_query.split(' by ')
        if len(parts) == 2:
            artist_query, song = parts[0].strip(), parts[1].strip()

    if artist_query:
        # Use query for search
        results = sp.search(q=artist_query, type='track', limit=1)
    elif artist_query and song:
        # Use artist and song for search
        results = sp.search(q=f"artist:{artist_query} track:{song}", type='track', limit=1)
    else:
        return "Please provide either artist and song, or a search query"
    
    tracks = results.get('tracks', {}).get('items', [])
    uris = []
    
    for track in tracks:
        if 'uri' in track.keys():
            uris.append(track['uri'])
    
    if len(uris) == 0:
        sp.start_playback(device_id=get_active_device())
        return "No tracks found, starting playback"
    
    sp.start_playback(device_id=get_active_device(), uris=uris)
    track = tracks[0]
    return f"Playing {track['name']} by {track['artists'][0]['name']}"


@tool(
    name="pause",
    description="Pause the currently playing music",
    aliases=["stop"]
)
def pause() -> str:
    """Pause the currently playing music on Spotify."""
    sp.pause_playback()
    return "Playback paused."


@tool(
    name="resume",
    description="Resume the currently paused music",
    aliases=["play", "unpause"]
)
def resume() -> str:
    """Resume the currently paused music on Spotify."""
    sp.start_playback(device_id=get_active_device())
    return "Playback resumed."


@tool(
    name="skip",
    description="Skip to the next track",
    aliases=["next", "next_track"]
)
def skip() -> str:
    """Skip to the next track in the playlist."""
    sp.next_track()
    return "Skipped to next track."

if __name__ == "__main__":
    print("Spotify Music Controller")
    
    # Print available tools
    print("\nAvailable tools:")
    for schema in tool_registry.get_all_schemas():
        print(f"  {schema.name}: {schema.description}")
        for param in schema.parameters:
            print(f"    - {param.name} ({param.type.value}): {param.description}")
    
    # Test function calling
    print("\nTesting function calling:")
    result = tool_registry.execute_tool("play_song", kwargs={"query": "Bohemian Rhapsody"})
    print(f"Result: {result}")