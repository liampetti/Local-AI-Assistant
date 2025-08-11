#!/bin/bash

# Setup script for local voice assistant configuration
# This runs everything on the same computer

echo "🎤 Setting up Voice Assistant (Local Configuration)"
echo "=================================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating data directories..."
mkdir -p openwake_data whisper_data ollama_intent_data ollama_chat_data piper_data

# Copy entrypoint scripts if they don't exist
if [ ! -f "intent_entrypoint.sh" ]; then
    echo "⚠️  intent_entrypoint.sh not found. Please create this file."
fi

if [ ! -f "chat_entrypoint.sh" ]; then
    echo "⚠️  chat_entrypoint.sh not found. Please create this file."
fi

# Start the services
echo "🚀 Starting voice assistant services (local configuration)..."
docker compose -f compose.yml up

echo ""
echo "✅ Local setup complete!"
echo ""
echo "Services running:"
echo "  - Wake word detection: localhost:10400"
echo "  - Speech recognition: localhost:10300"
echo "  - Intent AI model: localhost:11434"
echo "  - Chat AI model: localhost:11435"
echo "  - Text-to-speech: localhost:10200"
echo ""
echo "To view logs:"
echo "  docker-compose -f docker-compose.local.yml logs -f"
echo ""
echo "To stop services:"
echo "  docker-compose -f docker-compose.local.yml down"
echo ""
echo "To start the voice assistant:"
echo "  cd controller && python launch.py" 