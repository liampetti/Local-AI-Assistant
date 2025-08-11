import re

# Cleans wakewords from transcript
# Hey Mycroft
# Alexa
def clean_transcript(text):
    pattern = r'''
        \b
        (                                   # Whole wake word group
            (?:
                (?:h[aeiy]{0,2}|a)?         # optional fuzzy prefix: he, hay, hey, ha, a
                [\s\-]*                     # optional space or dash
            )
            m[iy]c?ro?[fp]{1,2}t+           # fuzzy 'mycroft'
            |
            a+l+[e3]?x+[a@]+                # fuzzy 'alexa'
        )
        [\s,.:;!?\"'\-]*                    # optional trailing junk
    '''
    return re.sub(pattern, '', text, flags=re.IGNORECASE | re.VERBOSE).strip()