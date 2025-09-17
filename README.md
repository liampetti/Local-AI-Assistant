# Local AI Voice Home Assistant (Experimental)

An experimental, fully local, privacy-first voice assistant for home automation and natural conversations. This project is a live testbed for assembling a Wyoming-powered voice pipeline, orchestrated with Docker Compose, while experimenting with a central knowledge graph and ongoing local LLM evaluations.

Note: This repository evolves rapidly as components and models are swapped, measured, and iterated for real-world home use cases.

## Core Components

* Wyoming-first voice pipeline: Audio services (wakeword, STT, TTS) communicate over the Wyoming Protocol to keep the pipeline decoupled, testable, and easy to swap during experiments.

* Modular via Docker Compose: Each service is intended to run as an independent container. This supports mix-and-match model trials and GPU performance testing of different modules.

* LLM as planner over a knowledge graph: A structured knowledge graph is used as the central knowledge store for the home, while a planning LLM decides when and how to traverse entities/relations to answer queries or route tools.

* Conversational LLM with tool access: The conversational model can pull from the knowledge graph and augment with external web search for richer, contextual dialogues with residents when permitted, keeping the core interaction private-first by default.

## Current Architecture

* Wake word: OpenWakeWord for fast, customizable activation.

* STT: Whisper for real-time transcription in the pipeline.

* TTS: Piper with locally downloaded voices for low-latency responses.

* LLM roles: Split models. One focused on tool/intent identification, web search summaries and KG planning, another for free-form conversation and reasoning.

* Tools and integrations: Philips Hue and AirTouch for home control, Australian BOM weather, Google Calendar, and Spotify with device targeting via device_id in credentials.

* Orchestration: Docker Compose coordinates services; the controller currently runs as a Python app with plans to containerise for a fully compose-driven stack.

* Distributed audio option: Reference script for using AudioRelay and Android phones as remote audio endpoints while centralizing inference on a mid-range GPU home pc server.

## Setup

1. Configure audio hardware and service URIs in config.py; update per-machine ports if offloading STT/TTS/LLMs.

2. Provide device credentials and info maps under tools/, following the included json.example patterns; set Spotify device_id to target playback endpoints.

3. Download Piper voice files into piper_data/voice/ and reference the chosen voice in the compose configuration.

4. Run using `launch.sh` script, followed by `python controller/app.py` after installing requirements (to be containerised).

## What’s Being Tested

1. Local model trials: Continuous benchmarking of small-to-mid local LLMs for latency, instruction-following, tool reliability, and robustness in noisy, multi-turn household conversations.

2. KG planning quality: Evaluating how well the LLM planner selects graph traversals vs. direct tool calls and when to supplement with web search for richer dialogue, while keeping on-device knowledge authoritative for the home.

3. Voice pipeline tuning: Buffer sizes, sample rates, silence detection, and echo cancellation for reliable wake, fast barge-in, and clear full-duplex behavior across diverse audio hardware.

## Roadmap

1. Containerize the controller: Move the Python controller (app.py) into its own container so the entire system runs under docker-compose up for simpler deployment, updates, and A/B test orchestration.

2. Add voice identification: Introduce a speaker ID module compatible with the Wyoming pipeline to personalize responses and apply per-speaker rules and permissions.

3. Keep evaluating local models and knowledge retrieval systems: Iterate on quantizations, runtimes, and scheduling to improve throughput and responsiveness within realistic VRAM/CPU constraints.

## References / Similar Projects
* [Rhasspy](https://github.com/rhasspy)
* [Home Assistant](https://github.com/home-assistant)
* [OpenVoiceOS](https://github.com/OpenVoiceOS)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. Contributions and experiment branches are welcome—especially around KG schemas, Wyoming-compatible speaker ID, controller containerization, and model runtime optimizations.