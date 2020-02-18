from pippi import dsp
from pippi import wavetables

sound1 = dsp.read('sound1.wav')
sound2 = dsp.read('sound2.flac')

# Mix two sounds
sound = sound1 & sound2

# Apply a skewed hann Wavetable as an envelope to a sound
enveloped = sound * wavetables.window(dsp.HANN, dsp.MS*20) # .skewed(0.6)

# Or the same, via a shortcut method on the `SoundBuffer`
enveloped = sound.env(dsp.HANN)

# Synthesize a 10 second graincloud from the sound, 
# with grain length modulating between 20ms and 2s 
# over a hann shaped curve.
cloudy = enveloped.cloud(10, grainlength=wavetables.window(dsp.HANN, dsp.MS*20))

