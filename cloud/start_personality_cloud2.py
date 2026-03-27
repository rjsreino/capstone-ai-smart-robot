#!/usr/bin/env python3
"""
Start script for Cloud 2 - Autonomous Personality Service
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from personality_cloud2 import main

if __name__ == "__main__":
    asyncio.run(main())

