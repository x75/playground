from os import path
import os
import random
import shutil
import tempfile
import time
from unittest import TestCase

import numpy as np

from pippi.soundbuffer import SoundBuffer
from pippi import dsp, fx, grains, wavetables, interpolation, oscs, tune

class GrainCloudFoo(object):
    def __init__(self):
        # self.filenames_prefix = '/home/lib/audio/work/tsx_recur_3/data/trk006-3-sco-0000-voice-1/'
        # self.filenames = [
        #     'trk006-3-sco-0000-voice-1_0000.000000.wav',
        #     'trk006-3-sco-0000-voice-1_0000.362585.wav',
        #     'trk006-3-sco-0000-voice-1_0001.430476.wav',
        #     'trk006-3-sco-0000-voice-1_0001.708730.wav',
        #     'trk006-3-sco-0000-voice-1_0003.218503.wav',
        #     'trk006-3-sco-0000-voice-1_0003.670816.wav',
        #     'trk006-3-sco-0000-voice-1_0006.319932.wav',
        #     'trk006-3-sco-0000-voice-1_0008.755420.wav',
        #     'trk006-3-sco-0000-voice-1_0015.204898.wav',
        #     'trk006-3-sco-0000-voice-1_0018.055374.wav',
        #     'trk006-3-sco-0000-voice-1_0018.869728.wav',
        #     'trk006-3-sco-0000-voice-1_0019.433764.wav',
        #     'trk006-3-sco-0000-voice-1_0019.595125.wav',
        #     'trk006-3-sco-0000-voice-1_0020.934036.wav',
        #     'trk006-3-sco-0000-voice-1_0021.758027.wav',
        #     'trk006-3-sco-0000-voice-1_0024.585760.wav',
        #     'trk006-3-sco-0000-voice-1_0026.430317.wav',
        #     # 'trk006-3-sco-0000-voice-1_0030.100499.wav',
        #     'trk006-3-sco-0000-voice-1_0030.130023.wav',
        #     'trk006-3-sco-0000-voice-1_0030.460227.wav',
        #     'trk006-3-sco-0000-voice-1_0030.785669.wav',
        #     'trk006-3-sco-0000-voice-1_0033.043492.wav',
        #     'trk006-3-sco-0000-voice-1_0033.471315.wav',
        #     'trk006-3-sco-0000-voice-1_0034.878617.wav',
        #     'trk006-3-sco-0000-voice-1_0036.747438.wav',
        #     'trk006-3-sco-0000-voice-1_0040.232290.wav',
        #     'trk006-3-sco-0000-voice-1_0042.518798.wav',
        #     'trk006-3-sco-0000-voice-1_0044.700159.wav',
        #     'trk006-3-sco-0000-voice-1_0047.015918.wav',
        #     'trk006-3-sco-0000-voice-1_0048.933243.wav',
        #     'trk006-3-sco-0000-voice-1_0050.795306.wav',
        #     'trk006-3-sco-0000-voice-1_0052.730385.wav',
        #     'trk006-3-sco-0000-voice-1_0054.534921.wav',
        #     'trk006-3-sco-0000-voice-1_0057.953651.wav',
        #     'trk006-3-sco-0000-voice-1_0058.263492.wav',
        #     'trk006-3-sco-0000-voice-1_0060.136122.wav',
        #     'trk006-3-sco-0000-voice-1_0061.773537.wav',
        #     'trk006-3-sco-0000-voice-1_0063.659592.wav',
        # ]
        
        # self.filenames_prefix = '/home/lib/audio/work/tsx_recur_3/data/trk008-6-sco-20200211-WA0001/'
        # self.filenames = [
        #     'fxgo-000.wav',
        #     'swarmy_graincloud-000.wav',
        #     'test_graincloud_with_read_lfo-000.wav',
        #     'trk008-6-sco-20200211-WA0001_0000.000000.wav',
        #     'trk008-6-sco-20200211-WA0001_0000.213900.wav',
        #     'trk008-6-sco-20200211-WA0001_0001.512268.wav',
        #     'trk008-6-sco-20200211-WA0001_0003.045170.wav',
        #     'trk008-6-sco-20200211-WA0001_0004.843107.wav',
        #     # 'trk008-6-sco-20200211-WA0001_0006.615011.wav',
        #     'trk008-6-sco-20200211-WA0001_0006.627324.wav',
        #     'trk008-6-sco-20200211-WA0001_0008.240408.wav',
        #     'trk008-6-sco-20200211-WA0001_0009.413560.wav',
        #     'trk008-6-sco-20200211-WA0001_0010.698458.wav',
        #     'trk008-6-sco-20200211-WA0001_0011.831723.wav',
        #     'trk008-6-sco-20200211-WA0001_0012.618413.wav',
        #     'trk008-6-sco-20200211-WA0001_0014.775057.wav',
        #     'trk008-6-sco-20200211-WA0001_0016.413061.wav',
        #     'trk008-6-sco-20200211-WA0001_0018.374853.wav',
        #     'trk008-6-sco-20200211-WA0001_0018.966122.wav',
        #     'trk008-6-sco-20200211-WA0001_0019.901837.wav',
        #     'trk008-6-sco-20200211-WA0001_0020.214467.wav',
        #     'trk008-6-sco-20200211-WA0001_0020.639955.wav',
        #     'trk008-6-sco-20200211-WA0001_0021.958776.wav',
        #     # 'trk008-6-sco-20200211-WA0001_0023.482200.wav',
        #     'trk008-6-sco-20200211-WA0001_0023.526281.wav',
        #     'trk008-6-sco-20200211-WA0001_0025.556145.wav',
        #     'trk008-6-sco-20200211-WA0001_0025.921179.wav',
        #     'trk008-6-sco-20200211-WA0001_0026.333152.wav',
        #     'trk008-6-sco-20200211-WA0001_0027.540317.wav',
        #     'trk008-6-sco-20200211-WA0001_0029.113401.wav',
        #     'trk008-6-sco-20200211-WA0001_0030.356599.wav',
        #     'trk008-6-sco-20200211-WA0001_0030.727279.wav',
        #     'trk008-6-sco-20200211-WA0001_0031.952789.wav',
        #     'trk008-6-sco-20200211-WA0001_0032.939705.wav',
        #     'trk008-6-sco-20200211-WA0001_0033.417052.wav',
        #     'trk008-6-sco-20200211-WA0001_0034.135057.wav',
        #     'trk008-6-sco-20200211-WA0001_0034.361701.wav',
        #     'trk008-6-sco-20200211-WA0001_0034.740023.wav',
        #     'trk008-6-sco-20200211-WA0001_0035.437755.wav',
        #     'trk008-6-sco-20200211-WA0001_0035.696077.wav',
        #     # 'trk008-6-sco-20200211-WA0001_0036.397052.wav',
        #     'trk008-6-sco-20200211-WA0001_0036.442630.wav',
        #     'trk008-6-sco-20200211-WA0001_0037.293605.wav',
        #     'trk008-6-sco-20200211-WA0001_0038.723197.wav',
        #     'trk008-6-sco-20200211-WA0001_0039.285034.wav',
        #     'trk008-6-sco-20200211-WA0001_0040.527460.wav',
        #     'trk008-6-sco-20200211-WA0001_0041.135533.wav',
        #     'trk008-6-sco-20200211-WA0001_0042.157007.wav',
        #     'trk008-6-sco-20200211-WA0001_0042.908322.wav',
        #     'trk008-6-sco-20200211-WA0001_0043.849025.wav',
        #     'trk008-6-sco-20200211-WA0001_0044.167256.wav',
        #     'trk008-6-sco-20200211-WA0001_0044.764649.wav',
        #     'trk008-6-sco-20200211-WA0001_0046.011678.wav',
        #     'trk008-6-sco-20200211-WA0001_0046.286190.wav',
        #     'trk008-6-sco-20200211-WA0001_0046.569410.wav',
        #     'trk008-6-sco-20200211-WA0001_0047.097188.wav',
        #     'trk008-6-sco-20200211-WA0001_0047.569093.wav',
        #     'trk008-6-sco-20200211-WA0001_0048.821950.wav',
        #     'trk008-6-sco-20200211-WA0001_0049.187460.wav',
        #     'trk008-6-sco-20200211-WA0001_0050.656463.wav',
        #     'trk008-6-sco-20200211-WA0001_0051.288980.wav',
        #     'trk008-6-sco-20200211-WA0001_0051.765420.wav',
        #     'trk008-6-sco-20200211-WA0001_0052.199909.wav',
        #     'trk008-6-sco-20200211-WA0001_0053.037143.wav',
        #     'trk008-6-sco-20200211-WA0001_0053.982902.wav',
        #     'trk008-6-sco-20200211-WA0001_0054.445964.wav',
        #     'trk008-6-sco-20200211-WA0001_0054.649796.wav',
        #     'trk008-6-sco-20200211-WA0001_0054.821043.wav',
        #     'trk008-6-sco-20200211-WA0001_0055.602154.wav',
        #     'trk008-6-sco-20200211-WA0001_0056.882472.wav',
        #     'trk008-6-sco-20200211-WA0001_0058.521701.wav',
        #     'trk008-6-sco-20200211-WA0001_0062.203424.wav',
        #     'trk008-6-sco-20200211-WA0001_0063.736485.wav',
        #     'trk008-6-sco-20200211-WA0001_0066.493810.wav',
        #     'trk008-6-sco-20200211-WA0001_0068.087256.wav',
        #     'trk008-6-sco-20200211-WA0001_0069.169909.wav',
        #     'trk008-6-sco-20200211-WA0001_0069.663651.wav',
        #     'trk008-6-sco-20200211-WA0001_0070.219229.wav',
        #     'trk008-6-sco-20200211-WA0001_0070.764376.wav',
        #     'trk008-6-sco-20200211-WA0001_0073.784218.wav',
        #     'trk008-6-sco-20200211-WA0001_0074.735760.wav',
        #     # 'trk008-6-sco-20200211-WA0001_0075.839796.wav',
        #     'trk008-6-sco-20200211-WA0001_0075.864399.wav',
        #     'trk008-6-sco-20200211-WA0001_0076.221837.wav',
        #     'trk008-6-sco-20200211-WA0001_0076.423946.wav',
        #     'trk008-6-sco-20200211-WA0001_0077.614966.wav',
        #     'trk008-6-sco-20200211-WA0001_0079.461723.wav',
        #     'trk008-6-sco-20200211-WA0001_0080.593129.wav',
        #     'trk008-6-sco-20200211-WA0001_0080.998821.wav',
        #     'trk008-6-sco-20200211-WA0001_0086.861361.wav',
        #     'trk008-6-sco-20200211-WA0001_0088.777823.wav',
        #     # 'trk008-6-sco-20200211-WA0001_0091.299025.wav',
        #     'trk008-6-sco-20200211-WA0001_0091.316190.wav',
        #     'trk008-6-sco-20200211-WA0001_0092.917120.wav',
        #     'trk008-6-sco-20200211-WA0001_0093.260227.wav',
        #     'trk008-6-sco-20200211-WA0001_0093.409705.wav',
        #     'trk008-6-sco-20200211-WA0001_0094.543651.wav',
        #     'trk008-6-sco-20200211-WA0001_0095.327460.wav',
        #     'trk008-6-sco-20200211-WA0001_0095.714830.wav',
        #     'trk008-6-sco-20200211-WA0001_0096.174535.wav',
        #     'trk008-6-sco-20200211-WA0001_0096.329955.wav',
        #     'trk008-6-sco-20200211-WA0001_0097.467120.wav',
        #     'trk008-6-sco-20200211-WA0001_0098.030091.wav',
        #     'trk008-6-sco-20200211-WA0001_0102.422358.wav',
        #     'trk008-6-sco-20200211-WA0001_0102.755397.wav',
        #     'trk008-6-sco-20200211-WA0001_0103.022744.wav',
        #     'trk008-6-sco-20200211-WA0001_0104.421497.wav',
        #     'trk008-6-sco-20200211-WA0001_0105.623084.wav',
        #     'trk008-6-sco-20200211-WA0001_0106.926893.wav',
        #     'trk008-6-sco-20200211-WA0001_0108.235760.wav',
        #     'trk008-6-sco-20200211-WA0001_0109.524059.wav',
        #     'trk008-6-sco-20200211-WA0001_0110.627279.wav',
        #     'trk008-6-sco-20200211-WA0001_0111.760726.wav',
        #     'trk008-6-sco-20200211-WA0001_0112.788027.wav',
        #     'trk008-6-sco-20200211-WA0001_0114.777959.wav',
        #     'trk008-6-sco-20200211-WA0001_0116.676689.wav',
        #     'trk008-6-sco-20200211-WA0001_0117.293900.wav',
        #     'trk008-6-sco-20200211-WA0001_0117.698231.wav',
        #     'trk008-6-sco-20200211-WA0001_0117.843628.wav',
        #     'trk008-6-sco-20200211-WA0001_0117.988231.wav',
        #     'trk008-6-sco-20200211-WA0001_0118.801882.wav',
        #     'trk008-6-sco-20200211-WA0001_0120.039524.wav',
        #     'trk008-6-sco-20200211-WA0001_0121.071134.wav',
        #     'trk008-6-sco-20200211-WA0001_0121.164535.wav',
        #     'trk008-6-sco-20200211-WA0001_0122.046213.wav',
        #     'trk008-6-sco-20200211-WA0001_0124.397052.wav',
        #     'trk008-6-sco-20200211-WA0001_0125.488073.wav',
        #     'trk008-6-sco-20200211-WA0001_0126.771519.wav',
        #     'trk008-6-sco-20200211-WA0001_0131.890771.wav',
        #     'trk008-6-sco-20200211-WA0001_0132.012517.wav',
        #     'trk008-6-sco-20200211-WA0001_0133.220726.wav',
        #     'trk008-6-sco-20200211-WA0001_0134.228617.wav',
        #     'trk008-6-sco-20200211-WA0001_0135.428118.wav',
        #     'trk008-6-sco-20200211-WA0001_0136.448435.wav',
        #     'trk008-6-sco-20200211-WA0001_0137.058844.wav',
        #     'trk008-6-sco-20200211-WA0001_0138.400544.wav',
        #     # 'trk008-6-sco-20200211-WA0001_0139.284059.wav',
        #     'trk008-6-sco-20200211-WA0001_0139.342222.wav',
        #     # 'trk008-6-sco-20200211-WA0001_0139.410340.wav',
        # ]
        # self.output_prefix = '/home/lib/audio/work/tsx_recur_3/data/trk008-6-sco-20200211-WA0001/proc/'

        # self.filenames_prefix = '/home/lib/audio/work/tsx_recur_3/data/trk026-1-sco-20200211-WA0003/'
        # self.filenames = [
        #     'trk026-1-sco-20200211-WA0003_0000.000000.wav',
        #     'trk026-1-sco-20200211-WA0003_0004.213084.wav',
        #     'trk026-1-sco-20200211-WA0003_0004.296893.wav',
        #     'trk026-1-sco-20200211-WA0003_0005.970952.wav',
        #     'trk026-1-sco-20200211-WA0003_0007.144444.wav',
        #     'trk026-1-sco-20200211-WA0003_0009.772494.wav',
        #     'trk026-1-sco-20200211-WA0003_0010.635964.wav',
        #     'trk026-1-sco-20200211-WA0003_0012.598980.wav',
        #     'trk026-1-sco-20200211-WA0003_0013.742404.wav',
        #     # 'trk026-1-sco-20200211-WA0003_0013.974853.wav',
        #     'trk026-1-sco-20200211-WA0003_0014.005374.wav',
        #     # 'trk026-1-sco-20200211-WA0003_0016.714830.wav',
        #     'trk026-1-sco-20200211-WA0003_0016.737370.wav',
        #     'trk026-1-sco-20200211-WA0003_0017.325850.wav',
        #     'trk026-1-sco-20200211-WA0003_0019.031338.wav',
        #     'trk026-1-sco-20200211-WA0003_0020.543061.wav',
        #     'trk026-1-sco-20200211-WA0003_0021.731633.wav',
        #     # 'trk026-1-sco-20200211-WA0003_0024.616168.wav',
        #     # 'trk026-1-sco-20200211-WA0003_0024.764422.wav',
        #     'trk026-1-sco-20200211-WA0003_0024.777166.wav',
        #     # 'trk026-1-sco-20200211-WA0003_0025.682608.wav',
        #     'trk026-1-sco-20200211-WA0003_0025.700703.wav',
        #     'trk026-1-sco-20200211-WA0003_0025.985351.wav',
        #     'trk026-1-sco-20200211-WA0003_0026.112744.wav',
        #     'trk026-1-sco-20200211-WA0003_0026.230771.wav',
        #     'trk026-1-sco-20200211-WA0003_0026.797415.wav',
        #     'trk026-1-sco-20200211-WA0003_0027.679660.wav',
        #     # 'trk026-1-sco-20200211-WA0003_0027.849615.wav',
        #     'trk026-1-sco-20200211-WA0003_0027.912290.wav',
        #     'trk026-1-sco-20200211-WA0003_0028.499206.wav',
        #     'trk026-1-sco-20200211-WA0003_0029.945692.wav',
        #     'trk026-1-sco-20200211-WA0003_0030.831633.wav',
        #     'trk026-1-sco-20200211-WA0003_0036.926871.wav',
        #     'trk026-1-sco-20200211-WA0003_0039.006054.wav',
        #     'trk026-1-sco-20200211-WA0003_0039.266712.wav',
        #     # 'trk026-1-sco-20200211-WA0003_0040.868639.wav',
        #     'trk026-1-sco-20200211-WA0003_0041.168322.wav',
        #     'trk026-1-sco-20200211-WA0003_0042.559206.wav',
        #     'trk026-1-sco-20200211-WA0003_0044.280159.wav',
        #     'trk026-1-sco-20200211-WA0003_0045.870181.wav',
        #     'trk026-1-sco-20200211-WA0003_0045.927891.wav',
        #     'trk026-1-sco-20200211-WA0003_0046.392653.wav',
        #     'trk026-1-sco-20200211-WA0003_0047.321451.wav',
        #     'trk026-1-sco-20200211-WA0003_0049.031020.wav',
        #     'trk026-1-sco-20200211-WA0003_0050.336757.wav',
        #     'trk026-1-sco-20200211-WA0003_0052.185102.wav',
        #     'trk026-1-sco-20200211-WA0003_0052.924422.wav',
        #     'trk026-1-sco-20200211-WA0003_0054.830045.wav',
        #     'trk026-1-sco-20200211-WA0003_0057.082132.wav',
        #     'trk026-1-sco-20200211-WA0003_0059.561043.wav',
        #     'trk026-1-sco-20200211-WA0003_0060.856576.wav',
        #     'trk026-1-sco-20200211-WA0003_0063.061066.wav',
        #     'trk026-1-sco-20200211-WA0003_0063.982744.wav',
        #     'trk026-1-sco-20200211-WA0003_0065.587687.wav',
        #     'trk026-1-sco-20200211-WA0003_0066.074376.wav',
        #     'trk026-1-sco-20200211-WA0003_0066.516100.wav',
        #     'trk026-1-sco-20200211-WA0003_0066.765397.wav',
        #     'trk026-1-sco-20200211-WA0003_0067.305941.wav',
        #     'trk026-1-sco-20200211-WA0003_0068.004966.wav',
        #     'trk026-1-sco-20200211-WA0003_0069.084331.wav',
        #     'trk026-1-sco-20200211-WA0003_0069.331882.wav',
        #     'trk026-1-sco-20200211-WA0003_0069.560000.wav',
        #     'trk026-1-sco-20200211-WA0003_0069.649456.wav',
        #     'trk026-1-sco-20200211-WA0003_0070.136440.wav',
        #     'trk026-1-sco-20200211-WA0003_0070.863764.wav',
        #     'trk026-1-sco-20200211-WA0003_0071.450499.wav',
        #     'trk026-1-sco-20200211-WA0003_0071.817370.wav',
        #     'trk026-1-sco-20200211-WA0003_0072.696145.wav',
        #     'trk026-1-sco-20200211-WA0003_0074.885896.wav',
        #     'trk026-1-sco-20200211-WA0003_0075.727256.wav',
        #     'trk026-1-sco-20200211-WA0003_0076.920000.wav',
        #     'trk026-1-sco-20200211-WA0003_0077.449297.wav',
        #     'trk026-1-sco-20200211-WA0003_0077.517528.wav',
        #     'trk026-1-sco-20200211-WA0003_0078.224399.wav',
        #     # 'trk026-1-sco-20200211-WA0003_0078.604671.wav',
        #     'trk026-1-sco-20200211-WA0003_0078.645351.wav',
        #     'trk026-1-sco-20200211-WA0003_0079.449683.wav',
        #     'trk026-1-sco-20200211-WA0003_0080.725828.wav',
        #     'trk026-1-sco-20200211-WA0003_0082.346236.wav',
        #     # 'trk026-1-sco-20200211-WA0003_0084.086032.wav',
        #     'trk026-1-sco-20200211-WA0003_0084.133265.wav',
        #     'trk026-1-sco-20200211-WA0003_0085.669138.wav',
        #     'trk026-1-sco-20200211-WA0003_0087.670952.wav',
        #     # 'trk026-1-sco-20200211-WA0003_0089.338571.wav',
        #     'trk026-1-sco-20200211-WA0003_0089.723741.wav',
        #     'trk026-1-sco-20200211-WA0003_0090.269456.wav',
        #     'trk026-1-sco-20200211-WA0003_0090.810930.wav',
        #     'trk026-1-sco-20200211-WA0003_0091.452517.wav',
        #     # 'trk026-1-sco-20200211-WA0003_0091.756508.wav',
        #     'trk026-1-sco-20200211-WA0003_0091.790204.wav',
        #     # 'trk026-1-sco-20200211-WA0003_0092.262404.wav',
        #     'trk026-1-sco-20200211-WA0003_0092.292154.wav',
        #     'trk026-1-sco-20200211-WA0003_0093.729002.wav',
        #     # 'trk026-1-sco-20200211-WA0003_0094.776984.wav',
        #     'trk026-1-sco-20200211-WA0003_0094.798118.wav',
        #     'trk026-1-sco-20200211-WA0003_0095.743810.wav',
        #     'trk026-1-sco-20200211-WA0003_0095.887868.wav',
        #     'trk026-1-sco-20200211-WA0003_0096.747846.wav',
        #     'trk026-1-sco-20200211-WA0003_0096.953311.wav',
        #     'trk026-1-sco-20200211-WA0003_0097.899524.wav',
        #     'trk026-1-sco-20200211-WA0003_0098.756757.wav',
        #     'trk026-1-sco-20200211-WA0003_0099.784603.wav',
        #     'trk026-1-sco-20200211-WA0003_0099.958254.wav',
        #     'trk026-1-sco-20200211-WA0003_0101.029048.wav',
        #     'trk026-1-sco-20200211-WA0003_0101.171429.wav',
        #     # 'trk026-1-sco-20200211-WA0003_0102.106145.wav',
        #     'trk026-1-sco-20200211-WA0003_0102.150295.wav',
        #     # 'trk026-1-sco-20200211-WA0003_0103.673696.wav',
        #     'trk026-1-sco-20200211-WA0003_0103.691451.wav',
        #     'trk026-1-sco-20200211-WA0003_0104.090998.wav',
        #     'trk026-1-sco-20200211-WA0003_0106.070023.wav',
        #     'trk026-1-sco-20200211-WA0003_0107.571066.wav',
        #     'trk026-1-sco-20200211-WA0003_0108.781905.wav',
        #     'trk026-1-sco-20200211-WA0003_0109.915011.wav',
        #     'trk026-1-sco-20200211-WA0003_0110.051519.wav',
        #     'trk026-1-sco-20200211-WA0003_0111.643220.wav',
        #     'trk026-1-sco-20200211-WA0003_0112.229615.wav',
        #     'trk026-1-sco-20200211-WA0003_0112.619864.wav',
        #     'trk026-1-sco-20200211-WA0003_0112.830431.wav',
        #     'trk026-1-sco-20200211-WA0003_0113.257778.wav',
        #     'trk026-1-sco-20200211-WA0003_0114.292109.wav',
        #     'trk026-1-sco-20200211-WA0003_0114.540000.wav',
        #     'trk026-1-sco-20200211-WA0003_0115.502902.wav',
        #     'trk026-1-sco-20200211-WA0003_0115.770522.wav',
        #     'trk026-1-sco-20200211-WA0003_0118.600317.wav',
        #     'trk026-1-sco-20200211-WA0003_0120.391293.wav',
        #     'trk026-1-sco-20200211-WA0003_0121.579773.wav',
        #     'trk026-1-sco-20200211-WA0003_0122.034853.wav',
        #     'trk026-1-sco-20200211-WA0003_0123.544195.wav',
        #     # 'trk026-1-sco-20200211-WA0003_0125.050975.wav',
        #     'trk026-1-sco-20200211-WA0003_0125.070227.wav',
        #     # 'trk026-1-sco-20200211-WA0003_0125.486712.wav',
        #     'trk026-1-sco-20200211-WA0003_0125.499252.wav',
        #     'trk026-1-sco-20200211-WA0003_0126.196304.wav',
        #     'trk026-1-sco-20200211-WA0003_0126.562608.wav',
        #     'trk026-1-sco-20200211-WA0003_0129.116825.wav',
        #     'trk026-1-sco-20200211-WA0003_0130.414943.wav',
        #     'trk026-1-sco-20200211-WA0003_0130.852336.wav',
        #     'trk026-1-sco-20200211-WA0003_0131.675918.wav',
        #     'trk026-1-sco-20200211-WA0003_0131.767732.wav',
        #     'trk026-1-sco-20200211-WA0003_0132.546621.wav',
        #     'trk026-1-sco-20200211-WA0003_0132.875011.wav',
        #     'trk026-1-sco-20200211-WA0003_0139.600771.wav',
        #     'trk026-1-sco-20200211-WA0003_0149.558844.wav',
        #     'trk026-1-sco-20200211-WA0003_0149.875034.wav',
        #     'trk026-1-sco-20200211-WA0003_0150.870136.wav',
        #     'trk026-1-sco-20200211-WA0003_0151.779342.wav',
        #     'trk026-1-sco-20200211-WA0003_0152.065306.wav',
        #     'trk026-1-sco-20200211-WA0003_0152.220159.wav',
        #     # 'trk026-1-sco-20200211-WA0003_0152.285374.wav',
        # ]

        self.filenames_prefix = '/home/lib/audio/work/tsx_recur_3/data/trk-AUD20200213/AUD-20200213-WA0001/'
        self.filenames = [
            # 'AUD-20200213-WA0000-isolated_0000.000000.wav',
            # 'AUD-20200213-WA0000-isolated_0000.183810.wav',
            # 'AUD-20200213-WA0000-isolated_0000.849433.wav',
            # 'AUD-20200213-WA0000-isolated_0003.311701.wav',
            # 'AUD-20200213-WA0000-isolated_0005.388776.wav',
            # 'AUD-20200213-WA0000-isolated_0005.713424.wav',
            # 'AUD-20200213-WA0000-isolated_0007.756803.wav',
            # 'AUD-20200213-WA0000-isolated_0008.177234.wav',
            # 'AUD-20200213-WA0000-isolated_0019.337868.wav',
            # 'AUD-20200213-WA0000-isolated_0019.959501.wav',
            # 'AUD-20200213-WA0000-isolated_0021.732766.wav',
            # 'AUD-20200213-WA0000-isolated_0022.357302.wav',
            # 'AUD-20200213-WA0000-isolated_0024.417029.wav',
            # 'AUD-20200213-WA0000-isolated_0024.563401.wav',
            # 'AUD-20200213-WA0000-isolated_0026.885714.wav',
            # 'AUD-20200213-WA0000-isolated_0026.965488.wav',
            # 'AUD-20200213-WA0000-isolated_0029.381293.wav',
            # 'AUD-20200213-WA0000-isolated_0031.832132.wav',
            # 'AUD-20200213-WA0000-isolated_0033.780431.wav',
            # 'AUD-20200213-WA0000-isolated_0033.909546.wav',
            # 'AUD-20200213-WA0000-isolated_0034.530068.wav',
            # # 'AUD-20200213-WA0000-isolated_0035.924535.wav',
            # 'AUD-20200213-WA0000-isolated_0036.041474.wav',
            # 'AUD-20200213-WA0000-isolated_0038.651950.wav',
            # 'AUD-20200213-WA0000-isolated_0038.958798.wav',
            # # 'AUD-20200213-WA0000-isolated_0041.386281.wav',
            # 'AUD-20200213-WA0000-isolated_0041.405420.wav',
            # 'AUD-20200213-WA0000-isolated_0043.378730.wav',
            # 'AUD-20200213-WA0000-isolated_0043.475578.wav',
            # 'AUD-20200213-WA0000-isolated_0045.674921.wav',
            # # 'AUD-20200213-WA0001-isolated_0000.000000.wav',
            # 'AUD-20200213-WA0001-isolated_0000.072245.wav',
            # 'AUD-20200213-WA0001-isolated_0000.199660.wav',
            # 'AUD-20200213-WA0001-isolated_0000.826984.wav',
            # 'AUD-20200213-WA0001-isolated_0003.112971.wav',
            # 'AUD-20200213-WA0001-isolated_0003.234649.wav',
            # 'AUD-20200213-WA0001-isolated_0003.774467.wav',
            # 'AUD-20200213-WA0001-isolated_0005.381156.wav',
            # 'AUD-20200213-WA0001-isolated_0007.688005.wav',
            # 'AUD-20200213-WA0001-isolated_0008.437302.wav',
            # 'AUD-20200213-WA0001-isolated_0008.779116.wav',
            # 'AUD-20200213-WA0001-isolated_0009.047891.wav',
            # 'AUD-20200213-WA0001-isolated_0010.001519.wav',
            # 'AUD-20200213-WA0001-isolated_0010.409977.wav',
            # 'AUD-20200213-WA0001-isolated_0011.176349.wav',
            # 'AUD-20200213-WA0001-isolated_0011.850431.wav',
            # 'AUD-20200213-WA0001-isolated_0013.926735.wav',
            # 'AUD-20200213-WA0001-isolated_0014.903084.wav',
            # # 'AUD-20200213-WA0001-isolated_0014.968118.wav',
            # 'AUD-20200213-WA0001-isolated_0014.989728.wav',
            # 'AUD-20200213-WA0001-isolated_0015.299932.wav',
            # 'AUD-20200213-WA0001-isolated_0016.036689.wav',
            # 'AUD-20200213-WA0001-isolated_0016.333810.wav',
            # 'AUD-20200213-WA0001-isolated_0016.680658.wav',
            # 'AUD-20200213-WA0001-isolated_0017.243333.wav',
            # 'AUD-20200213-WA0001-isolated_0018.443719.wav',
            # 'AUD-20200213-WA0001-isolated_0018.707392.wav',
            # 'AUD-20200213-WA0001-isolated_0019.288141.wav',
            # 'AUD-20200213-WA0001-isolated_0019.699229.wav',
            # 'AUD-20200213-WA0001-isolated_0019.971361.wav',
            # 'AUD-20200213-WA0001-isolated_0020.891905.wav',
            # 'AUD-20200213-WA0001-isolated_0021.724036.wav',
            # 'AUD-20200213-WA0001-isolated_0022.113605.wav',
            # 'AUD-20200213-WA0001-isolated_0023.274558.wav',
            # 'AUD-20200213-WA0001-isolated_0024.064376.wav',
            # 'AUD-20200213-WA0001-isolated_0024.184785.wav',
            # 'AUD-20200213-WA0001-isolated_0024.815964.wav',
            # 'AUD-20200213-WA0001-isolated_0025.758980.wav',
            # 'AUD-20200213-WA0001-isolated_0026.611633.wav',
            # 'AUD-20200213-WA0001-isolated_0027.211723.wav',
            # 'AUD-20200213-WA0001-isolated_0028.174172.wav',
            # 'AUD-20200213-WA0001-isolated_0029.001224.wav',
            # 'AUD-20200213-WA0001-isolated_0029.625102.wav',
            # # 'AUD-20200213-WA0001-isolated_0030.068277.wav',
            # 'AUD-20200213-WA0001-isolated_0030.083447.wav',
            # 'AUD-20200213-WA0001-isolated_0030.256916.wav',
            # 'AUD-20200213-WA0001-isolated_0030.570567.wav',
            # 'AUD-20200213-WA0001-isolated_0031.461383.wav',
            # 'AUD-20200213-WA0001-isolated_0031.838889.wav',
            # 'AUD-20200213-WA0001-isolated_0032.542630.wav',
            # 'AUD-20200213-WA0001-isolated_0033.014195.wav',
            # 'AUD-20200213-WA0001-isolated_0033.850363.wav',
            # 'AUD-20200213-WA0001-isolated_0034.062041.wav',
            # 'AUD-20200213-WA0001-isolated_0034.629206.wav',
            # 'AUD-20200213-WA0001-isolated_0034.781882.wav',
            # 'AUD-20200213-WA0001-isolated_0036.308141.wav',
            # 'AUD-20200213-WA0001-isolated_0036.498934.wav',
            # 'AUD-20200213-WA0001-isolated_0036.667596.wav',
            # 'AUD-20200213-WA0001-isolated_0038.039637.wav',
            'AUD-20200213-WA0002_0000.000000.wav',
            # 'AUD-20200213-WA0002_0000.251859.wav',
            'AUD-20200213-WA0002_0000.267075.wav',
            'AUD-20200213-WA0002_0002.615329.wav',
            # 'AUD-20200213-WA0002_0004.797438.wav',
            'AUD-20200213-WA0002_0004.814649.wav',
            'AUD-20200213-WA0002_0006.855465.wav',
            # 'AUD-20200213-WA0003_0000.000000.wav',
            'AUD-20200213-WA0003_0000.000023.wav',
            'AUD-20200213-WA0003_0001.001610.wav',
            'AUD-20200213-WA0003_0001.429637.wav',
            'AUD-20200213-WA0003_0001.574399.wav',
            'AUD-20200213-WA0003_0002.307120.wav',
            'AUD-20200213-WA0003_0003.661134.wav',
            'AUD-20200213-WA0003_0003.984694.wav',
            'AUD-20200213-WA0003_0004.839070.wav',
            'AUD-20200213-WA0003_0005.452132.wav',
            'AUD-20200213-WA0003_0005.487664.wav',
            'AUD-20200213-WA0003_0006.199320.wav',
            'AUD-20200213-WA0003_0006.389048.wav',
            'AUD-20200213-WA0003_0008.483583.wav',
            'AUD-20200213-WA0003_0008.832245.wav',
        ]
        
        # self.filenames_prefix = '/home/lib/audio/work/tsx_recur_2/sue/Robert R Graham 2/'
        # self.filenames = [
        #     'Robert R Graham 2_0000.000000.wav',
        #     'Robert R Graham 2_0002.211250.wav',
        #     'Robert R Graham 2_0002.691333.wav',
        #     'Robert R Graham 2_0002.772354.wav',
        #     'Robert R Graham 2_0003.330521.wav',
        #     'Robert R Graham 2_0004.130000.wav',
        #     'Robert R Graham 2_0004.705896.wav',
        #     'Robert R Graham 2_0004.995396.wav',
        #     'Robert R Graham 2_0005.698083.wav',
        #     'Robert R Graham 2_0006.280375.wav',
        #     'Robert R Graham 2_0007.426896.wav',
        #     'Robert R Graham 2_0007.973104.wav',
        # ]

        self.filenames_prefix = '/home/lib/audio/work/tsx_recur_2/sue/Robert R Graham/'
        self.filenames = [
            'Robert R Graham_0000.000000.wav',
            'Robert R Graham_0001.777625.wav',
            'Robert R Graham_0006.003458.wav',
            'Robert R Graham_0006.102500.wav',
            'Robert R Graham_0006.542750.wav',
            'Robert R Graham_0007.777437.wav',
            'Robert R Graham_0009.601458.wav',
            'Robert R Graham_0009.699479.wav',
            'Robert R Graham_0012.492229.wav',
            'Robert R Graham_0013.422896.wav',
            'Robert R Graham_0016.854979.wav',
            'Robert R Graham_0017.047583.wav',
            'Robert R Graham_0018.425292.wav',
            'Robert R Graham_0019.521833.wav',
            'Robert R Graham_0020.513437.wav',
            'Robert R Graham_0020.953937.wav',
            'Robert R Graham_0021.644104.wav',
            'Robert R Graham_0023.324021.wav',
            'Robert R Graham_0023.559062.wav',
            'Robert R Graham_0024.483417.wav',
        ]
        self.output_prefix = self.filenames_prefix + '/proc/'

        self.filename_i = 0
        self.set_filename()
        self.save = True
        self.save_format = '.wav'

    def set_filename(self):
        self.filename = self.filenames_prefix + self.filenames[self.filename_i]
        
    def setUp(self):
        self.soundfiles = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.soundfiles)

    def test_unmodulated_graincloud(self):
        sound = SoundBuffer(filename=self.filename)
        cloud = grains.GrainCloud(sound)

        length = random.triangular(0.25, 4)
        framelength = int(length * sound.samplerate)

        out = cloud.play(length)
        # self.assertEqual(len(out), framelength)

        if self.save:
            out.write('%s/test_unmodulated_graincloud-%03d.wav' % (self.output_prefix, self.filename_i))

    def test_minspeed_graincloud(self):
        sound = SoundBuffer(filename=self.filename)
        cloud = grains.GrainCloud(sound, speed=0.002)

        length = random.triangular(0.25, 4)
        framelength = int(length * sound.samplerate)

        out = cloud.play(length)
        # self.assertEqual(len(out), framelength)

        if self.save:
            out.write('%s/test_minspeed_graincloud-%03d.wav' % (self.output_prefix, self.filename_i))

    def test_maxspeed_graincloud(self):
        sound = SoundBuffer(filename=self.filename)
        cloud = grains.GrainCloud(sound, speed=99)

        length = random.triangular(0.25, 4)
        framelength = int(length * sound.samplerate)

        out = cloud.play(length)
        # self.assertEqual(len(out), framelength)

        if self.save:
            out.write('%s/test_maxspeed_graincloud-%03d.wav' % (self.output_prefix, self.filename_i))

    def test_graincloud_with_length_lfo(self):
        sound = SoundBuffer(filename=self.filename)
        cloud = grains.GrainCloud(sound, grainlength_lfo=dsp.RND)

        length = random.triangular(0.25, 4)
        framelength = int(length * sound.samplerate)

        out = cloud.play(length)
        # self.assertEqual(len(out), framelength)

        if self.save:
            out.write('%s/test_graincloud_with_length_lfo-%03d.wav' % (self.output_prefix, self.filename_i))

    def test_graincloud_with_speed_lfo(self):
        sound = SoundBuffer(filename=self.filename)
        minspeed = random.triangular(0.5, 1)
        maxspeed = minspeed + random.triangular(0.5, 1)
        cloud = grains.GrainCloud(sound, 
                            speed_lfo=dsp.RND, 
                            minspeed=minspeed, 
                            maxspeed=maxspeed
                        )

        length = random.triangular(0.25, 4)
        framelength = int(length * sound.samplerate)

        out = cloud.play(length)
        # self.assertEqual(len(out), framelength)

        if self.save:
            out.write('%s/test_graincloud_with_speed_lfo-%03d.wav' % (self.output_prefix, self.filename_i))

    def test_graincloud_with_density_lfo(self):
        sound = SoundBuffer(filename=self.filename)
        cloud = grains.GrainCloud(sound, 
                            density_lfo=dsp.RND, 
                            density=random.triangular(0.5, 5),
                        )

        length = random.triangular(0.25, 4)
        framelength = int(length * sound.samplerate)

        out = cloud.play(length)
        # self.assertEqual(len(out), framelength)

        if self.save:
            out.write('%s/test_graincloud_with_density_lfo-%03d.wav' % (self.output_prefix, self.filename_i))

    def test_graincloud_with_read_lfo(self):
        sound = SoundBuffer(filename=self.filename)
        cloud = grains.GrainCloud(sound, 
                            read_lfo=dsp.RND, 
                            read_lfo_speed=random.triangular(0.5, 10)
                        )

        length = random.triangular(0.25, 4)
        framelength = int(length * sound.samplerate)

        out = cloud.play(length)
        # self.assertEqual(len(out), framelength)

        if self.save:
            out.write('%s/test_graincloud_with_read_lfo-%03d.wav' % (self.output_prefix, self.filename_i))

    def fxgo(self):
        # snd = dsp.read('%s/sounds/harpc2.wav' % PATH)
        snd = dsp.read(self.filename)
        snd = fx.go(snd,
                    # factor=200 + np.random.normal(0, 10),
                    # minclip=0.125,
                    # maxclip=0.5,
                    # density=np.random.uniform(1e-2, 0.99), # 0.65,
                    # minlength=np.random.uniform(1e-3, 1e-2), # 0.01,
                    # maxlength=np.random.uniform(4e-2, 1e-1), # 0.04
                    factor=100 + np.random.normal(0, 10),
                    minclip=0.125 * 2,
                    maxclip=0.5 * 2,
                    density=np.random.uniform(1e-2, 0.65),
                    minlength=np.random.uniform(1e-2, 1e-1), # 0.01,
                    maxlength=np.random.uniform(2e-1, 5e-1), # 0.04
        )

        if self.save:
            # snd.write('%s/fxgo.wav' % PATH)
            snd.write('%s/fxgo-%03d.wav' % (self.output_prefix, self.filename_i))

    def swarmy_graincloud(self):
        # snd = dsp.read('%s/sounds/linus.wav' % PATH)
        snd = dsp.read(self.filename)

        def makecloud(density):
            print('cloud density', density)
            return grains.GrainCloud(snd * 0.125,
                                     win=dsp.HANN,
                                     read_lfo=dsp.PHASOR,
                                     speed_lfo_wt=interpolation.linear([ random.random() for _ in range(random.randint(10, 1000)) ], 4096), 
                                     density_lfo_wt=interpolation.linear([ random.random() for _ in range(random.randint(10, 1000)) ], 4096),
                                     grainlength_lfo_wt=interpolation.linear([ random.random() for _ in range(random.randint(10, 500)) ], 4096),
                                     minspeed=0.25,
                                     maxspeed=random.triangular(0.25, 10),
                                     density=density,
                                     minlength=1,
                                     maxlength=random.triangular(60, 100),
                                     spread=random.random(),
                                     jitter=random.triangular(0, 0.1),
            ).play(5)

        numclouds = 10
        densities = [ (random.triangular(0.1, 2),) for _ in range(numclouds) ]

        #clouds = dsp.pool(makecloud, numclouds, densities)
        out = dsp.buffer(length=5)
        #for cloud in clouds:
        for i in range(numclouds):
            cloud = makecloud(*densities[i])
            out.dub(cloud, 0)

        # out.write('%s/swarmy_graincloud.wav' % PATH)
        if self.save:
            # snd.write('%s/fxgo.wav' % PATH)
            out.write('%s/swarmy_graincloud-%03d.wav' % (self.output_prefix, self.filename_i))

    def pulsar_synth(self):

        # PATH = os.path.dirname(os.path.realpath(__file__))
        # print(__file__)
        start_time = time.time()

        out = dsp.buffer(length=4)
        freqs = tune.fromdegrees([1,2,3,4,5,6,7,8,9], octave=-8, root='d')
        print(freqs)
        
        snd = dsp.read(self.filename)
        
        print('Creating 5,000 1ms - 100ms long pulsar notes in a 40 second buffer...')
        for _ in range(100):
            pos = random.triangular(0, 3.5)
            # length = random.triangular(0.001, 0.1)
            length = random.triangular(0.1, 0.5)

            # Pulsar wavetable constructed from a random set of linearly interpolated points & a randomly selected window
            # Frequency modulated between 1% and 300% with a randomly generated wavetable LFO between 0.01hz and 30hz
            # with a random, fixed-per-note pulsewidth
            # wavetable = [0] + [ random.triangular(-1, 1) for _ in range(random.randint(3, 100)) ] + [0]
            
            # wavetable = [0] + [ snd[_][0] for _ in range(random.randint(3, 100)) ] + [0]
            # wtlen = random.randint(5, 1000)
            wtlen = random.randint(2000, min(len(snd), 20000))
            wtoffset = random.randint(0, len(snd) - wtlen - 1)
            wavetable = [0] + [ snd[wtoffset + _][0] for _ in range(wtlen) ] + [0]
            mod = [ random.triangular(0, 1) for _ in range(random.randint(3, 20)) ]
            osc = oscs.Osc(wavetable, window=dsp.RND, mod=mod)
            pulsewidth = random.random()
            
            freq = random.choice(freqs) * 2**random.randint(0, 10)
            # freq = random.choice(freqs) * 2**random.randint(0, 4)
            mod_freq = random.triangular(0.01, 30)
            # mod_range = random.triangular(0, random.choice([0.03, 0.02, 0.01, 3]))
            mod_range = random.triangular(0, random.choice([0.1, 0.2, 0.3, 1.0]))
            amp = random.triangular(0.05, 0.5)

            if mod_range > 1:
                amp *= 0.5

            note = osc.play(length, freq, amp, pulsewidth, mod_freq=mod_freq, mod_range=mod_range)
            note = note.env(dsp.RND)
            note = note.pan(random.random())

            out.dub(note, pos)

        # out.write('%s/pulsar_synth.wav' % '.')
        out.write('%s/pulsar_synth-%03d.wav' % (self.output_prefix, self.filename_i))
        elapsed_time = time.time() - start_time
        print('Render time: %s seconds' % round(elapsed_time, 2))
        print('Output length: %s seconds' % out.dur)
        
if __name__ == '__main__':

    gc = GrainCloudFoo()
    # gc.test_graincloud_with_extreme_speed_lfo()
    for i in range(len(gc.filenames)):
        gc.filename_i = i
        gc.set_filename()
        print('working {0}'.format(gc.filename))
        
        # gc.test_unmodulated_graincloud()
        # gc.test_minspeed_graincloud()
        # gc.test_maxspeed_graincloud()
        # gc.test_graincloud_with_length_lfo()
        # gc.test_graincloud_with_speed_lfo()
        # gc.test_graincloud_with_density_lfo()
        # gc.test_graincloud_with_read_lfo()
        gc.fxgo()        
        gc.swarmy_graincloud()
        try:
            gc.pulsar_synth()
        except Exception as e:
            print('failed with {0}'.format(e))
