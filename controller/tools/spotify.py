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
import difflib

from .tool_registry import tool, tool_registry

# Load credentials from JSON file
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spotify_creds.json'), 'r') as f:
    creds = json.load(f)

SCOPE = 'user-read-playback-state user-modify-playback-state user-read-currently-playing'

SIMILARITY_THRESHOLD = 0.6 # How similar a user query is to a playlist, track or album

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
        if device['name'] == creds['device_id']:
            return device['id']
    return None


@tool(
    name="play_song",
    description="Play a song by artist and title, or search for a song by query",
    aliases=["play", "play_music", "start_music"]
)
def play_song(artist_query: Optional[str] = None, song: Optional[str] = None) -> str:
    """
    Play a song or playlist on Spotify, prioritizing user's playlist names, then songs in playlists,
    top artists, saved albums, and finally general search.

    Args:
        artist_query: Artist name, playlist name, or search query
        song: Song title (if two arguments are provided, one is artist and one is song title)

    Returns:
        Status message about the played song or playlist.
    """

    # Ensure arguments are separated
    if artist_query and re.search(r'\s+by\s+', artist_query):
        parts = artist_query.split(' by ')
        if len(parts) == 2:
            artist_query, song = parts[1].strip(), parts[0].strip()

    playlists = sp.current_user_playlists(limit=50)['items']

    # 1. Check if query closely matches a playlist name
    playlist_names = [pl['name'] for pl in playlists]
    # Find close matches to artist_query
    matches = difflib.get_close_matches(artist_query, playlist_names, n=1, cutoff=0.6)
    if matches:
        for playlist in playlists:
            if playlist['name'] == matches[0]:
                pause()
                sp.start_playback(device_id=get_active_device(), context_uri=playlist['uri'])
                return f"Playing your playlist \"{playlist['name']}\""

    # # 2. Search user's playlists for the track
    # for playlist in playlists:
    #     results = sp.playlist_tracks(playlist['id'])
    #     for item in results['items']:
    #         track = item['track']
    #         if (song and song.lower() in track['name'].lower()) or \
    #            (artist_query and artist_query.lower() in track['artists'][0]['name'].lower()):
    #             sp.start_playback(device_id=get_active_device(), uris=[track['uri']])
    #             return f"Playing {track['name']} by {track['artists'][0]['name']} from your playlist \"{playlist['name']}\""

    # 3. Search user's top artists for tracks
    # top_artists = sp.current_user_top_artists(limit=20, time_range='medium_term')['items']
    # for artist in top_artists:
    #     results = sp.search(q=f"artist:{artist['name']}" + (f" track:{song}" if song else ""), type='track', limit=1)
    #     tracks = results.get('tracks', {}).get('items', [])
    #     if tracks:
    #         sp.start_playback(device_id=get_active_device(), uris=[tracks[0]['uri']])
    #         return f"Playing {tracks[0]['name']} by {artist['name']} "

    # # 4. Search user's saved albums for tracks
    # saved_albums = sp.current_user_saved_albums(limit=50)['items']
    # for album_item in saved_albums:
    #     album = album_item['album']
    #     for track in album['tracks']['items']:
    #         if song and song.lower() in track['name'].lower():
    #             sp.start_playback(device_id=get_active_device(), uris=[track['uri']])
    #             return f"Playing {track['name']} from your saved album \"{album['name']}\""

    # 5. Fallback to general search
    if artist_query and song:
        results = sp.search(q=f"artist:{artist_query} track:{song}", type='track', limit=1)
    elif artist_query:
        results = sp.search(q=artist_query, type='track', limit=1)
    else:
        return "Please provide either artist and song, playlist name, or a search query"

    try:
        tracks = results.get('tracks', {}).get('items', [])
        uris = [track['uri'] for track in tracks if 'uri' in track]

        if len(uris) == 0:
            pause()
            sp.start_playback(device_id=get_active_device())
            return "No tracks found, starting playback"
        else:
            pause()
            sp.start_playback(device_id=get_active_device(), uris=uris)
            track = tracks[0]
            return f"Playing {track['name']} by {track['artists'][0]['name']}"
    except Exception as e:
        return "Unable to play your request"


def is_playing() -> bool:
    """Check if currently playing music on Spotify."""
    playback = sp.current_playback()
    if playback and playback['is_playing']:
        return True
    return False

@tool(
    name="pause",
    description="Pause the currently playing music",
    aliases=["stop"]
)
def pause() -> str:
    """Pause the currently playing music on Spotify."""
    playback = sp.current_playback()
    if playback and playback['is_playing']:
        sp.pause_playback()
    return "Playback paused."


@tool(
    name="resume",
    description="Resume the currently paused music",
    aliases=["play", "unpause"]
)
def resume() -> str:
    """Resume the currently paused music on Spotify."""
    playback = sp.current_playback()
    if playback and not playback['is_playing']:
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
    print(sp.current_playback())
    
    # Print available tools
    print("\nAvailable tools:")
    for schema in tool_registry.get_all_schemas():
        print(f"  {schema.name}: {schema.description}")
        for param in schema.parameters:
            print(f"    - {param.name} ({param.type.value}): {param.description}")
    
    # Test function calling
    print("\nTesting function calling:")
    result = tool_registry.execute_tool("play_song", kwargs={"artist_query": "cell phone by old mervs"})
    print(f"Result: {result}")