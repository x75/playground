from os import path
import random
import shutil
import tempfile
from unittest import TestCase

from pippi.soundbuffer import SoundBuffer
from pippi import dsp, grains

#save = False
save = True

class TestGrainCloud(TestCase):
    def setUp(self):
        self.soundfiles = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.soundfiles)

    def test_unmodulated_graincloud(self):
        sound = SoundBuffer(filename=self.filename)
        cloud = grains.GrainCloud(sound)

        length = random.triangular(1, 4)
        framelength = int(length * sound.samplerate)

        out = cloud.play(length)
        self.assertEqual(len(out), framelength)

        if save:
            out.write('test_unmodulated_graincloud.flac')

    def test_minspeed_graincloud(self):
        sound = SoundBuffer(filename=self.filename)
        cloud = grains.GrainCloud(sound, speed=0)

        length = random.triangular(1, 4)
        framelength = int(length * sound.samplerate)

        out = cloud.play(length)
        self.assertEqual(len(out), framelength)

        if self.save:
            out.write('test_minspeed_graincloud.flac')

    def test_maxspeed_graincloud(self):
        sound = SoundBuffer(filename=self.filename)
        cloud = grains.GrainCloud(sound, speed=99)

        length = random.triangular(1, 4)
        framelength = int(length * sound.samplerate)

        out = cloud.play(length)
        self.assertEqual(len(out), framelength)

        if self.save:
            out.write('test_maxspeed_graincloud.flac')

    def test_graincloud_with_length_lfo(self):
        sound = SoundBuffer(filename=self.filename)
        cloud = grains.GrainCloud(sound, grainlength_lfo=dsp.RND)

        length = random.triangular(1, 4)
        framelength = int(length * sound.samplerate)

        out = cloud.play(length)
        self.assertEqual(len(out), framelength)

        if self.save:
            out.write('test_graincloud_with_length_lfo.flac')

    def test_graincloud_with_speed_lfo(self):
        sound = SoundBuffer(filename=self.filename)
        minspeed = random.triangular(0.5, 1)
        maxspeed = minspeed + random.triangular(0.5, 1)
        cloud = grains.GrainCloud(sound, 
                            speed_lfo=dsp.RND, 
                            minspeed=minspeed, 
                            maxspeed=maxspeed
                        )

        length = random.triangular(1, 4)
        framelength = int(length * sound.samplerate)

        out = cloud.play(length)
        self.assertEqual(len(out), framelength)

        if self.save:
            out.write('test_graincloud_with_speed_lfo.flac')

    def test_graincloud_with_density_lfo(self):
        sound = SoundBuffer(filename=self.filename)
        cloud = grains.GrainCloud(sound, 
                            density_lfo=dsp.RND, 
                            density=random.triangular(0.5, 5),
                        )

        length = random.triangular(1, 4)
        framelength = int(length * sound.samplerate)

        out = cloud.play(length)
        self.assertEqual(len(out), framelength)

        if self.save:
            out.write('test_graincloud_with_density_lfo.flac')

    def test_graincloud_with_read_lfo(self):
        sound = SoundBuffer(filename=self.filename)
        cloud = grains.GrainCloud(sound, 
                            read_lfo=dsp.RND, 
                            read_lfo_speed=random.triangular(0.5, 10)
                        )

        length = random.triangular(1, 4)
        framelength = int(length * sound.samplerate)

        out = cloud.play(length)
        self.assertEqual(len(out), framelength)

        if self.save:
            out.write('test_graincloud_with_read_lfo.flac')


