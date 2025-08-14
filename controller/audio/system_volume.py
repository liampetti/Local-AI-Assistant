"""
System volume manager module for the voice assistant.

Handles the main system volume using PulseAudio (pulsectl).
"""

import pulsectl
import time

class VolumeManager:
    def __init__(self):
        self.pulse = pulsectl.Pulse('system-volume-manager')

    def get_master_volume(self):
        # Find the default PulseAudio sink (output device)
        default_sink_name = self.pulse.server_info().default_sink_name
        # Get the actual Sink object
        sink = next(s for s in self.pulse.sink_list() if s.name == default_sink_name)
        # Return the average volume as a percentage
        return sink.volume.value_flat * 100

    def set_master_volume(self, volume):
        # Clamp value to [0, 100] for safety
        value = max(0, min(100, float(volume)))
        default_sink_name = self.pulse.server_info().default_sink_name
        sink = next(s for s in self.pulse.sink_list() if s.name == default_sink_name)
        # pulsectl expects volume as float in [0.0, 1.0]
        self.pulse.volume_set_all_chans(sink, value / 100)

    def close(self):
        self.pulse.close()

if __name__ == "__main__":
    manager = VolumeManager()
    print("Current volume:", manager.get_master_volume())
    time.sleep(2)
    manager.set_master_volume(0)    # Set to 0%
    print("Volume after change:", manager.get_master_volume())
    time.sleep(3)
    manager.set_master_volume(50)   # Set to 50%
    print("Volume after change:", manager.get_master_volume())
    manager.close()
