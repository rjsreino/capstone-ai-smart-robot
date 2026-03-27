#!/usr/bin/env python3
"""
One-time Spotify authentication setup
Run this to get a refresh token that allows ROVY to control your Spotify
"""
import os
import sys

# Set credentials
os.environ["SPOTIFY_CLIENT_ID"] = "93138e86ecf24daea4b07df74c7cb8e9"
os.environ["SPOTIFY_CLIENT_SECRET"] = "f8f131ad542a4cf2a021aae8bdbc5763"
os.environ["SPOTIFY_REDIRECT_URI"] = "http://localhost:8888/callback"

try:
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
except ImportError:
    print("ERROR: spotipy not installed")
    print("Run: pip install spotipy")
    sys.exit(1)

print("=" * 70)
print("  SPOTIFY AUTHENTICATION FOR ROVY")
print("=" * 70)
print()
print("This will open a browser to log in to Spotify.")
print("After logging in, you'll be redirected to localhost:8888")
print("Just copy the FULL URL and paste it here.")
print()
input("Press Enter to continue...")

# Create OAuth manager
sp_oauth = SpotifyOAuth(
    client_id=os.environ["SPOTIFY_CLIENT_ID"],
    client_secret=os.environ["SPOTIFY_CLIENT_SECRET"],
    redirect_uri=os.environ["SPOTIFY_REDIRECT_URI"],
    scope="user-read-playback-state,user-modify-playback-state,user-library-read,user-read-currently-playing",
    cache_path=".spotify_token_cache",
    open_browser=True
)

# Get token (will open browser)
token_info = sp_oauth.get_cached_token()
if not token_info:
    auth_url = sp_oauth.get_authorize_url()
    print("\n1. Browser should open automatically")
    print("   If not, open this URL:")
    print(f"   {auth_url}\n")
    print("2. Log in with your Spotify account")
    print("3. Click 'Agree'")
    print("4. Copy the FULL redirect URL (http://localhost:8888/callback?code=...)")
    print()
    
    redirect_url = input("Paste the full URL here: ").strip()
    code = sp_oauth.parse_response_code(redirect_url)
    token_info = sp_oauth.get_access_token(code, as_dict=True)

print("\n" + "=" * 70)
print("✅ AUTHENTICATION SUCCESSFUL!")
print("=" * 70)

# Test it
sp = spotipy.Spotify(auth=token_info['access_token'])
user = sp.current_user()
print(f"\n Logged in as: {user['display_name']}")

# Show devices
devices = sp.devices()
print(f"\n📱 Spotify devices found:")
for dev in devices.get('devices', []):
    active = " ✅ ACTIVE" if dev['is_active'] else ""
    print(f"   • {dev['name']} ({dev['type']}){active}")

print("\n" + "=" * 70)
print("🎉 SETUP COMPLETE!")
print("=" * 70)
print("\n✅ Token saved to .spotify_token_cache")
print("✅ ROVY can now control your Spotify")
print("\nNow:")
print("1. Make sure cloud server has SPOTIFY_ENABLED=true")
print("2. Start the cloud server")
print("3. Say 'play music' to ROVY!")
print()

