pactl load-module module-null-sink \
    sink_name=audiorelay-speakers \
    sink_properties=device.description=AudioRelay-Speakers

pactl load-module module-null-sink \
    sink_name=audiorelay-virtual-mic-sink \
    sink_properties=device.description=Virtual-Mic-Sink

pactl load-module module-remap-source \
    master=audiorelay-virtual-mic-sink.monitor \
    source_name=audiorelay-virtual-mic-sink \
    source_properties=device.description=Virtual-Mic
