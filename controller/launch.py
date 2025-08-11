#!/usr/bin/env python3
"""
Launcher script for the voice assistant.

This script provides a simple way to start the voice assistant with
proper error handling and logging.
"""

import sys
import os
import logging
import signal
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("voice_assistant.log")
        ]
    )

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logging.info("Received shutdown signal, stopping voice assistant...")
    sys.exit(0)

def check_dependencies():
    """Check if required dependencies are available."""
    try:
        import sounddevice
        import numpy
        import wyoming
        import requests
        logging.info("All core dependencies are available")
        return True
    except ImportError as e:
        logging.error(f"Missing dependency: {e}")
        logging.error("Please install dependencies with: pip install -r requirements.txt")
        return False

def check_services():
    """Check if required services are running."""
    import requests
    import socket
    
    services = {
        "Ollama": "http://localhost:11434",
        "Wyoming Wake Word": "tcp://localhost:10400",
        "Wyoming Whisper": "tcp://localhost:10300",
        "Wyoming Piper": "tcp://localhost:10200"
    }
    
    all_available = True
    
    for service_name, service_url in services.items():
        try:
            if service_url.startswith("http"):
                response = requests.get(service_url, timeout=2)
                if response.status_code == 200:
                    logging.info(f"✅ {service_name} is available")
                else:
                    logging.warning(f"⚠️  {service_name} returned status {response.status_code}")
                    all_available = False
            else:
                # For TCP services, just check if port is open
                host, port = service_url.replace("tcp://", "").split(":")
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex((host, int(port)))
                sock.close()
                
                if result == 0:
                    logging.info(f"✅ {service_name} is available")
                else:
                    logging.warning(f"⚠️  {service_name} is not available")
                    all_available = False
        except Exception as e:
            logging.warning(f"⚠️  {service_name} check failed: {e}")
            all_available = False
    
    return all_available

def main():
    """Main entry point for the launcher."""
    print("🎤 Voice Assistant Launcher")
    print("=" * 40)
    
    # Set up logging
    setup_logging()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check services (optional)
    print("\nChecking external services...")
    services_available = check_services()
    
    if not services_available:
        print("\n⚠️  Some external services are not available.")
        print("The assistant may not work properly without these services.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    print("\n🚀 Starting voice assistant...")
    
    try:
        from app import main as app_main
        app_main()
    except KeyboardInterrupt:
        logging.info("Voice assistant stopped by user")
    except Exception as e:
        logging.exception(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 