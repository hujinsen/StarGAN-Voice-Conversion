import argparse
import os
import numpy as np

from model import StarGANVC
from preprocess import *
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from utility import *

#get all speaker
all_speaker = get_speakers(trainset='./data/fourspeakers')
label_enc = LabelEncoder()
label_enc.fit(all_speaker)


def conversion(model_dir, test_dir, output_dir, source, target):
    if not os.path.exists(model_dir) or not os.path.exists(test_dir):
        raise Exception('model dir or test dir not exist!')
    model = StarGANVC(num_features=FEATURE_DIM, mode='test')

    model.load(filepath=os.path.join(model_dir, MODEL_NAME))
    #f'./data/fourspeakers_test/{source}/*.wav'
    p = os.path.join(test_dir, f'{source}/*.wav')
    tempfiles = glob.glob(p)

    normlizer = Normalizer()

    for one_file in tempfiles:
        _, speaker, name = os.path.normpath(one_file).rsplit(os.sep, maxsplit=2)
        # print(speaker, name)
        wav_, fs = librosa.load(one_file, sr=SAMPLE_RATE, mono=True, dtype=np.float64)
        wav, pad_length = pad_wav_to_get_fixed_frames(wav_, frames=FRAMES)

        f0, timeaxis = pyworld.harvest(wav, fs, f0_floor=71.0, f0_ceil=500.0)

        #CheapTrick harmonic spectral envelope estimation algorithm.
        sp = pyworld.cheaptrick(wav, f0, timeaxis, fs, fft_size=FFTSIZE)

        #D4C aperiodicity estimation algorithm.
        ap = pyworld.d4c(wav, f0, timeaxis, fs, fft_size=FFTSIZE)
        #feature reduction
        coded_sp = pyworld.code_spectral_envelope(sp, fs, FEATURE_DIM)

        coded_sps_mean = np.mean(coded_sp, axis=0, dtype=np.float64, keepdims=True)
        coded_sps_std = np.std(coded_sp, axis=0, dtype=np.float64, keepdims=True)
        #normalize
        # coded_sp = (coded_sp - coded_sps_mean) / coded_sps_std
        # print(coded_sp.shape, f0.shape, ap.shape)

        #one audio file to multiple slices(that's one_test_sample),every slice is an input
        one_test_sample = []
        csp_transpose = coded_sp.T  #36x512 36x128...
        for i in range(0, csp_transpose.shape[1] - FRAMES + 1, FRAMES):
            t = csp_transpose[:, i:i + FRAMES]
            #normalize t
            t = normlizer.forward_process(t, speaker)
            t = np.reshape(t, [t.shape[0], t.shape[1], 1])
            one_test_sample.append(t)
        # print(f'{len(one_test_sample)} slices appended!')

        #generate target label (one-hot vector)
        one_test_sample_label = np.zeros([len(one_test_sample), len(all_speaker)])
        temp_index = label_enc.transform([target])[0]
        one_test_sample_label[:, temp_index] = 1

        generated_results = model.test(one_test_sample, one_test_sample_label)

        reshpaped_res = []
        for one in generated_results:
            t = np.reshape(one, [one.shape[0], one.shape[1]])

            t = normlizer.backward_process(t, target)
            reshpaped_res.append(t)
        #collect the generated slices, and concate the array to be a whole representation of the whole audio
        c = []
        for one_slice in reshpaped_res:
            one_slice = np.ascontiguousarray(one_slice.T, dtype=np.float64)
            # one_slice = one_slice * coded_sps_std + coded_sps_mean

            # print(f'one_slice : {one_slice.shape}')
            decoded_sp = pyworld.decode_spectral_envelope(one_slice, SAMPLE_RATE, fft_size=FFTSIZE)
            # print(f'decoded_sp shape: {decoded_sp.shape}')
            c.append(decoded_sp)

        concated = np.concatenate((c), axis=0)
        # print(f'concated shape: {concated.shape}')
        #f0 convert
        f0 = normlizer.pitch_conversion(f0, speaker, target)

        synwav = pyworld.synthesize(f0, concated, ap, fs)
        # print(f'origin wav:{len(wav_)} paded wav:{len(wav)} synthesize wav:{len(synwav)}')

        #remove synthesized wav paded length
        synwav = synwav[:-pad_length]

        #save synthesized wav to file
        wavname = f'{speaker}-{target}+{name}'
        wavpath = f'{output_dir}/wavs'
        if not os.path.exists(wavpath):
            os.makedirs(wavpath, exist_ok=True)
        librosa.output.write_wav(f'{wavpath}/{wavname}', synwav, sr=fs)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert voices using pre-trained CycleGAN model.')

    model_dir = './out/90_2018-10-17-22-58/model/'
    test_dir = './data/fourspeakers_test/'
    source_speaker = 'SF1'
    target_speaker = 'TM1'
    output_dir = './converted_voices'

    parser.add_argument('--model_dir', type=str, help='Directory for the pre-trained model.', default=model_dir)
    parser.add_argument('--test_dir', type=str, help='Directory for the voices for conversion.', default=test_dir)
    parser.add_argument('--output_dir', type=str, help='Directory for the converted voices.', default=output_dir)
    parser.add_argument('--source_speaker', type=str, help='source_speaker', default=source_speaker)
    parser.add_argument('--target_speaker', type=str, help='target_speaker', default=target_speaker)

    argv = parser.parse_args()

    model_dir = argv.model_dir
    test_dir = argv.test_dir
    output_dir = argv.output_dir
    source_speaker = argv.source_speaker
    target_speaker = argv.target_speaker

    conversion(model_dir = model_dir,\
     test_dir = test_dir, output_dir = output_dir, source=source_speaker, target=target_speaker)
