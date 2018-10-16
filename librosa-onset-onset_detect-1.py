# Get onset times from a signal
import argparse
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import madmom

DEBUG=True

def myprint(*args, **kwargs):
    if not DEBUG: return
    print(*args, **kwargs)

def plotit(**kwargs):
    ############################################################
    # plotting
    y = kwargs['y']
    o_env = kwargs['o_env']
    times = kwargs['times']
    onset_frames = kwargs['onset_frames']
    tempo = kwargs['tempo']
    beats = kwargs['beats']
    dtempo = kwargs['dtempo']
    tempo2 = kwargs['tempo2']
    beats2 = kwargs['beats2']
    dtempo2 = kwargs['dtempo2']
    chroma = kwargs['chroma']
    bounds = kwargs['bounds']
    bound_times = kwargs['bound_times']
    mm_beat_times = kwargs['mm_beat_times']
    file = kwargs['file']

    D = np.abs(librosa.stft(y))
    fig = plt.figure()
    fig.suptitle('%s' % (file))
    fig_numrow = 5

    ax1 = plt.subplot(fig_numrow, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                             x_axis='time', y_axis='log')
    plt.title('Power spectrogram')
    
    plt.subplot(fig_numrow, 1, 2, sharex=ax1)
    plt.plot(times, o_env, label='Onset strength')
    plt.vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9,
               linestyle='--', label='Onsets')
    
    plt.subplot(fig_numrow, 1, 3, sharex=ax1)
    plt.plot(times, librosa.util.normalize(o_env),
             label='Onset strength', alpha=0.33)
    plt.vlines(times[beats], 0.5, 1, alpha=0.5, color='b',
               linestyle='--', label='Beats')
    plt.vlines(times[beats2], 0, 0.5, alpha=0.5, color='k',
               linestyle='-.', label='Beats2')
    plt.vlines(mm_beat_times, 0.25, 0.75, alpha=0.75, color='r',
               linestyle='-', linewidth=2, label='mm beats')
    
    plt.subplot(fig_numrow, 1, 4, sharex=ax1)
    plt.plot(times, dtempo, alpha=0.5, color='b',
               linestyle='none', marker='o', label='Tempo')
    plt.plot(times, dtempo2, alpha=0.5, color='k',
               linestyle='none', marker='o', label='Tempo2')

    
    plt.subplot(fig_numrow, 1, 5, sharex=ax1)
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
    plt.vlines(bound_times, 0, chroma.shape[0], color='linen', linestyle='--',
            linewidth=2, alpha=0.9, label='Segment boundaries')    
    
    plt.axis('tight')
    plt.legend(frameon=True, framealpha=0.75)

def data_load_librosa(args):
    if args.file is None:
        filename = librosa.util.example_audio_file()
    else:
        # filename = '/home/x75/Downloads/BEAT1R-mono.wav'
        filename = args.file
    myprint('Loading audio file %s' % (filename))
    
    y, sr = librosa.load(filename, offset=0, duration=args.duration)
    myprint('Loaded audio file with %s samples at rate %d' % (y.shape, sr))
    return y, sr

def compute_onsets_librosa(y, sr):
    myprint('Computing onset strength envelope')
    o_env = librosa.onset.onset_strength(y, sr=sr)
    times = librosa.frames_to_time(np.arange(len(o_env)), sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)
    myprint('Computing onset strength envelope o_env = %s' % (o_env.shape, ))
    return o_env, times, onset_frames
    

def compute_beats_librosa(onset_env, onset_frames, start_bpm, sr):
    myprint('Computing beat_track with start_bpm = {0}'.format(start_bpm))
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, start_bpm=float(start_bpm))
    dtempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=None, start_bpm=float(start_bpm))
    return tempo, dtempo, beats
    
def main_segmentation_iter_clust(args):
    

def main(args):
    # load data from file
    y, sr = data_load_librosa(args)
    
    # myprint('Computing onsets')
    # onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    # librosa.frames_to_time(onset_frames, sr=sr)
    
    # # Or use a pre-computed onset envelope
    # # array([ 0.07 ,  0.395,  0.511,  0.627,  0.766,  0.975,
    # # 1.207,  1.324,  1.44 ,  1.788,  1.881])

    # compute onsets
    onset_env, onset_times_ref, onset_frames = compute_onsets_librosa(y, sr)

    # FIXME: is there a beat? danceability?
    # compute_beat(onset_env, onset_frames)

    beats = {}
    for start_bpm in [30, 60, 90]:
        t_, dt_, b_ = compute_beats_librosa(onset_env, onset_frames, start_bpm, sr)
        beats[start_bpm] = {}
        beats[start_bpm]['tempo'] = t_
        beats[start_bpm]['dtempo'] = dt_
        beats[start_bpm]['beats'] = b_
    
    # madmom beat tracking
    myprint('Computing beat_track mm')
    mm_proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    mm_act = madmom.features.beats.RNNBeatProcessor()(filename)

    mm_beat_times = mm_proc(mm_act)
    myprint('mm_beat_times', mm_beat_times)
    
    # part segmentation
    # FIXME: how many parts?
    myprint('Computing parts segmentation')
    numparts = 100
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    bounds = librosa.segment.agglomerative(chroma, numparts)
    bound_times = librosa.frames_to_time(bounds, sr=sr)
    bound_samples = librosa.frames_to_samples(bounds, hop_length=512, n_fft=2048)
    myprint('bound_samples = %s / %s' % (bound_samples.shape, bound_samples))
    
    for i in range(numparts):
        i_start = bound_samples[i-1]
        if i < 1:
            i_start = 0
            
        tmp_ = y[i_start:bound_samples[i]]
        outfilename = filename[:-4] + "-%d.wav" % (i)
        myprint('writing seg %d to outfile %s' % (i, outfilename))
        # librosa.output.write_wav(outfilename, tmp_, sr)
    
    # bound_times
    # array([  0.   ,   1.672,   2.322,   2.624,   3.251,   3.506,
    #      4.18 ,   5.387,   6.014,   6.293,   6.943,   7.198,
    #      7.848,   9.033,   9.706,   9.961,  10.635,  10.89 ,
    #     11.54 ,  12.539])


    myprint('Plotting')
    plotit(y=y, o_env=o_env, times=times, onset_frames=onset_frames, file=args.file,
           tempo=tempo, beats=beats, dtempo=dtempo, 
           tempo2=tempo2, beats2=beats2, dtempo2=dtempo2,
           chroma=chroma, bounds=bounds, bound_times=bound_times,
           mm_beat_times=mm_beat_times,
    )

    # # more segmentation with recurrence_matrix
    # winsize = int(2**14)
    # hopsize = int(winsize/4)
    # mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=winsize, hop_length=hopsize)
    # R = librosa.segment.recurrence_matrix(mfcc)
    # R_aff = librosa.segment.recurrence_matrix(mfcc, mode='affinity')
    # myprint('R = %s, R_aff = %s' % (R.shape, R_aff.shape))

    # fig2 = plt.figure(figsize=(8, 4))
    # plt.subplot(1, 2, 1)
    # librosa.display.specshow(R, x_axis='time', y_axis='time')
    # plt.title('Binary recurrence (symmetric)')
    # plt.subplot(1, 2, 2)
    # librosa.display.specshow(R_aff, x_axis='time', y_axis='time', cmap='magma_r')
    # plt.title('Affinity recurrence')
    # plt.tight_layout()
    
    plt.show()
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--duration', help='Input duration (secs) to select from input file [10.0]',
                        default=10.0, type=float)
    parser.add_argument('-f', '--file', help='Sound file to process', default=None, type=str)

    args = parser.parse_args()

    main(args)
    
