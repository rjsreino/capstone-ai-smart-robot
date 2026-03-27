#!/usr/bin/env python3
"""
Spotify Authentication Setup for ROVY
Run this once to authenticate with Spotify Web API
"""
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import config

print("=" * 60)
print("  SPOTIFY SETUP FOR ROVY")
print("=" * 60)
print()
print("This will authenticate your Spotify account with ROVY")
print("so the robot can control playback programmatically.")
print()

# Create Spotify OAuth manager
sp_oauth = SpotifyOAuth(
    client_id=config.SPOTIFY_CLIENT_ID,
    client_secret=config.SPOTIFY_CLIENT_SECRET,
    redirect_uri=config.SPOTIFY_REDIRECT_URI,
    scope="user-read-playback-state,user-modify-playback-state,user-read-currently-playing",
    cache_path="/tmp/.spotify_cache",
    open_browser=True
)

# Get auth URL
auth_url = sp_oauth.get_authorize_url()

print("Steps:")
print("1. A browser will open (or copy this URL):")
print(f"   {auth_url}")
print()
print("2. Log in to Spotify")
print("3. Click 'Agree' to grant permissions")
print("4. You'll be redirected to localhost:8888/callback")
print("5. Copy the FULL URL from your browser and paste it here")
print()

# Get the redirect URL from user
redirect_response = input("Paste the full redirect URL here: ")

# Extract the code
code = sp_oauth.parse_response_code(redirect_response)
token_info = sp_oauth.get_access_token(code)

print()
print("=" * 60)
print("✅ SUCCESS! Spotify authenticated")
print("=" * 60)
print()

# Test it
sp = spotipy.Spotify(auth=token_info['access_token'])
user = sp.current_user()

print(f"Logged in as: {user['display_name']}")
print()

# List devices
devices = sp.devices()
print("Available Spotify devices:")
for device in devices.get('devices', []):
    active = " (ACTIVE)" if device['is_active'] else ""
    print(f"  - {device['name']}{active}")

print()
print("Setup complete! You can now say 'play music' to ROVY!")
print()
print("Note: Make sure 'ROVY' device is visible in your Spotify app")
print("and connect to it at least once before voice control.")

