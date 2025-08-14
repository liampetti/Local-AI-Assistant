# Local AI Voice Home Assistant

A voice assistant for fully local home automation and AI interactions. Places locally-run AI in the middle of the home automation system. Wyoming Protocol is used for voice processing pipelines and Model Context Protocol (MCP) is implemented for AI tool and service connections.

Built with modules from Rhasspy (https://github.com/rhasspy)


**Note: This is a very alpha 'concept' version of how a fully local AI home assistant could work, do not expect plug-and-play. A mid-high range PC with GPU will be required for usable response times depending on LLM models chosen.**

## Overview

This voice assistant provides hands-free control of your smart home devices, answers questions, and engages in natural conversations. It uses local AI models for privacy and includes home automation and life admin capabilities.

### Key Features

- **Voice Recognition**: Real-time speech-to-text using Whisper
- **Wake Word Detection**: Custom wake word detection using OpenWakeWord
- **Quick Intent Capture**: Optional keyword matching to quickly capture intents before activating AI
- **AI Chat**: Natural language conversations using local LLMs
- **Home Automation**: Control lights, thermostats, music, and more
- **Life Admin**: Provide household information and check calender, set timers etc.
- **Text-to-Speech**: Natural voice responses using Piper TTS
- **Privacy-First**: Designed to work with small LLM's running on home pc or laptop, no data sent to cloud.
- **AI Centric**: AI is main focus of the home assistant, not an addon.
- **Split AI for task focus**: Smaller, specifically prompted AI model for tool and intent identification. Larger AI model for conversation.

## Version

**Current Version**: 0.1.0

## Architecture

The application is built to run on a single pc. The "chat" model could be offloaded and API endpoints changed easily if wanting to distribute load.

## Setup Instructions

   #### Docker Setup
   
   **Local Setup** (everything on one computer):
   ```bash
   ./setup-local.sh
   ```
   
### Configuration & Setup

1. **Audio Configuration**: Edit `config.py` to match your audio hardware
2. **Service URLs**: Update service URIs in `config.py` if using different ports or offloading LLM's and speech modules
3. **Smart Home**: Configure information maps and device credentials in `tools/` modules, see json.example files. Remove any tools not needed.
4. **Create Vector DB**: Update family, location and intent info jsons in `utils/`. Run `createdb.py` to build a vector db for retrieval augmented generation.
5. **Download Voice**: Choose a voice from (https://huggingface.co/rhasspy/piper-voices) and load it into `piper_data/voice/`. Set correct voice in docker compose file.

## Usage Instructions

### Starting the Assistant

```bash
cd controller
python launch.py
```

The launcher will:
1. Check dependencies and services
2. Set up logging and error handling
3. Start the voice assistant with proper configuration

## Voice Commands

### Home Automation

**Philips Hue and AirTouch Tools Included**

- "Turn on the lights"
- "Set brightness to 50% in the living room"
- "Turn off the kitchen lights"
- "Set temperature to 22 degrees"

### Information Queries

**Australian BOM and Google Calendar Tools Included**

- "What's the weather today?"
- "What time is it?"
- "What's on my calendar today?"

### Music Control

**Spotify Tool Included**

- "Play some music"
- "Play Bohemian Rhapsody by Queen"
- "Pause music"
- "Skip to next song"

### General Conversation
- "Tell me a joke"
- "What's the capital of France?"
- "How are you today?"

### Interruption

Say the wake word again at any time to interrupt the assistant's response.

## Development

### Adding New Features

1. **New Intent**: Add to `tools/` directory and register in `utils/intents.py`
2. **New Audio Processing**: Extend `audio/` modules
3. **New AI Feature**: Extend `ai/` modules
4. **New TTS Feature**: Extend `tts/` modules

### TODO:

1. **ESP32 Device Compatibility**: Setup controller to accept audio stream from ESP32 devices (e.g. ReSpeaker), offloading wakeword detection and audio processing to the ESP32.
2. **Optimise Controller for Mini PC & Server Split**: Split functions into basic abilities that can all run on a mini pc or Rhaspberry Pi, with heavy compute being done by central server.
3. **Fine-Tuning of Intent LLM**: Fine-tune a very small Intent LLM to only output the required JSON commands for common home automation and life admin tasks. Super fast and possible to run on small edge devices.
4. **RAG and Tool Use for Chat LLM**: Improve information gathering options of the Chat LLM and allow it to also utilise tools such as web search etc.

### Performance Tuning

- Adjust buffer sizes in `config.py` for your hardware
- Modify sample rates for better performance
- Tune silence detection parameters
- Tune echo cancellation parameters

### Contributors

- Claude Sonnet 3.5 and 4.0
- GPT 4.1
- Perplexity

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.