from pippi.soundbuffer import SoundBuffer
from pippi import dsp, grains

def test_unmodulated_graincloud():
    sound = SoundBuffer(filename='sound1.wav')
    # cloud = grains.Cloud(sound)
    cloud = grains.GrainCloud(sound)

    length = 3
    framelength = int(length * sound.samplerate)

    out = cloud.play(length)
    # self.assertEqual(len(out), framelength)

    out.write('graincloud_unmodulated.wav')


def test_pulsed_graincloud():
    sound = SoundBuffer(filename='sound2.wav')
    # out = sound.cloud(10, grainlength=0.06, grid=0.12)
    out = sound.cloud(10, grainlength=0.06, spread=0.12)
    out.write('graincloud_pulsed.wav')

def test_graincloud_with_length_lfo():
    sound = SoundBuffer(filename='sound1.wav')
    grainlength = dsp.wt(dsp.HANN, 0.01, 0.1)
    length = 3
    framelength = int(length * sound.samplerate)

    out = sound.cloud(length, grainlength=grainlength)

    # self.assertEqual(len(out), framelength)

    out.write('graincloud_with_length_lfo.wav')


test_unmodulated_graincloud()

test_pulsed_graincloud()

test_graincloud_with_length_lfo()
