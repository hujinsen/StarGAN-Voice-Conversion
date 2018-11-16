import os
import numpy as np
import argparse
import time
import librosa
import glob
from preprocess import *
from model import *
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from utility import *


def get_files_labels(pattern: str):
    files = glob.glob(pattern)
    names = []
    for f in files:
        t = f.rsplit('/', maxsplit=1)[1]  #'./data/processed/SF2-100008_11.npy'
        name = t.rsplit('.', maxsplit=1)[0]
        names.append(name)

    return files, names


def train(processed_dir: str, test_wav_dir: str):
    timestr = time.strftime("%Y-%m-%d-%H-%M", time.localtime())  #like '2018-10-10-14-47'

    all_speaker = get_speakers()
    label_enc = LabelEncoder()
    label_enc.fit(all_speaker)

    lambda_cycle = 10
    lambda_identity = 5
    lambda_classifier = 3

    generator_learning_rate = 0.0002
    generator_learning_rate_decay = generator_learning_rate / 20000
    discriminator_learning_rate = 0.0002
    discriminator_learning_rate_decay = discriminator_learning_rate / 20000
    domain_classifier_learning_rate = 0.0001
    domain_classifier_learning_rate_decay = domain_classifier_learning_rate / 20000
    #====================load data================#
    print('Loading Data...')

    files, names = get_files_labels(os.path.join(processed_dir, '*.npy'))
    assert len(files) > 0

    normlizer = Normalizer()

    exclude_dict = {}  #key that not appear in the value list.(eg. SF1:[TM1**.wav,TM2**.wav,SF2**.wav ... ])
    for s in all_speaker:
        p = os.path.join(processed_dir, '*.npy')  #'./data/processed/*.npy'
        temp = [fn for fn in glob.glob(p) if fn.find(s) == -1]
        exclude_dict[s] = temp

    print('Loading Data Done.')

    #====================create model=============#
    BATCHSIZE = 8
    model = StarGANVC(num_features=FEATURE_DIM, frames=FRAMES)
    #====================start train==============#
    EPOCH = 101

    num_samples = len(files)
    for epoch in range(EPOCH):
        start_time_epoch = time.time()

        files_shuffled, names_shuffled = shuffle(files, names)

        for i in range(num_samples // BATCHSIZE):
            num_iterations = num_samples // BATCHSIZE * epoch + i

            if num_iterations > 2500:
                
                domain_classifier_learning_rate = max(0, domain_classifier_learning_rate - domain_classifier_learning_rate_decay)
                generator_learning_rate = max(0, generator_learning_rate - generator_learning_rate_decay)
                discriminator_learning_rate = max(0, discriminator_learning_rate - discriminator_learning_rate_decay)

            if discriminator_learning_rate == 0 or generator_learning_rate == 0:
                print('Early stop training.')
                break
            # if num_iterations > 2500:
            #     lambda_identity = 1
            #     domain_classifier_learning_rate = 0
            #     generator_learning_rate = max(0, generator_learning_rate - generator_learning_rate_decay)
            #     discriminator_learning_rate = discriminator_learning_rate + discriminator_learning_rate_decay

            # if generator_learning_rate <= 0.0001:
            #     generator_learning_rate = 0.0001
            # if discriminator_learning_rate >= 0.0002:
            #     discriminator_learning_rate = 0.0002

            start = i * BATCHSIZE
            end = (i + 1) * BATCHSIZE

            if end > num_samples:
                end = num_samples

            X, X_t, y, y_t = [], [], [], []

            #get target file paths
            batchnames = names_shuffled[start:end]
            pre_targets = []
            for name in batchnames:
                name = name.split(sep='-')[0]  #SF1
                t = np.random.choice(exclude_dict[name], 1)[0]
                pre_targets.append(t)

            #one batch train data
            for one_filename, one_name, one_target in zip(files_shuffled[start:end], names_shuffled[start:end], pre_targets):

                #target name
                t = one_target.rsplit('/', maxsplit=1)[1]  #'./data/processed/SF2-100008_11.npy'
                target_speaker_name = t.rsplit('.', maxsplit=1)[0].split('-')[0]

                #source name
                speaker_name = one_name.split('-')[0]  #SF1

                #shape [36,512]
                one_file = np.load(one_filename)
                one_file = normlizer.forward_process(one_file, speaker_name)

                #shape [36,512,1]
                one_file = np.reshape(one_file, [one_file.shape[0], one_file.shape[1], 1])
                X.append(one_file)

                #source label
                temp_index = label_enc.transform([speaker_name])[0]
                temp_arr_s = np.zeros([
                    len(all_speaker),
                ])
                temp_arr_s[temp_index] = 1
                y.append(temp_arr_s)

                #load target files and labels
                one_file_t = np.load(one_target)
                one_file_t = normlizer.forward_process(one_file_t, target_speaker_name)

                #[36,512,1]
                one_file_t = np.reshape(one_file_t, [one_file_t.shape[0], one_file_t.shape[1], 1])
                X_t.append(one_file_t)

                #target label
                temp_index_t = label_enc.transform([target_speaker_name])[0]
                temp_arr_t = np.zeros([
                    len(all_speaker),
                ])
                temp_arr_t[temp_index_t] = 1
                y_t.append(temp_arr_t)


            generator_loss, discriminator_loss, domain_classifier_loss = model.train(\
            input_source=X, input_target=X_t, source_label=y, \
            target_label=y_t, generator_learning_rate=generator_learning_rate,\
             discriminator_learning_rate=discriminator_learning_rate,\
            classifier_learning_rate=domain_classifier_learning_rate, \
            lambda_identity=lambda_identity, lambda_cycle=lambda_cycle,\
            lambda_classifier=lambda_classifier
            )

            print('Iteration: {:07d}, Generator Learning Rate: {:.7f}, Discriminator Learning Rate: {:.7f},Generator Loss : {:.3f}, Discriminator Loss : {:.3f}, domain_classifier_loss: {:.3f}'\
            .format(num_iterations, generator_learning_rate, discriminator_learning_rate, generator_loss, \
            discriminator_loss, domain_classifier_loss))

        #=======================test model==========================

        if epoch % 10 == 0 and epoch != 0:
            print('============test model============')
            #out put path
            file_path = os.path.join('./out', f'{epoch}_{timestr}')
            if not os.path.exists(file_path):
                os.makedirs(file_path)

            tempfiles = []
            for one_speaker in all_speaker:
                p = os.path.join(test_wav_dir, f'{one_speaker}/*.wav')
                wavs = glob.glob(p)
                tempfiles.append(wavs[0])
                tempfiles.append(wavs[1])  #'./data/fourspeakers_test/200006.wav'

            for one_file in tempfiles:
                _, speaker, name = one_file.rsplit('/', maxsplit=2)
                wav_, fs = librosa.load(one_file, sr=SAMPLE_RATE, mono=True, dtype=np.float64)
                wav, pad_length = pad_wav_to_get_fixed_frames(wav_, frames=FRAMES)

                f0, timeaxis = pyworld.harvest(wav, fs, f0_floor=71.0, f0_ceil=500.0)
                sp = pyworld.cheaptrick(wav, f0, timeaxis, fs, fft_size=FFTSIZE)
                ap = pyworld.d4c(wav, f0, timeaxis, fs, fft_size=FFTSIZE)
                coded_sp = pyworld.code_spectral_envelope(sp, fs, FEATURE_DIM)

                #one audio file to multiple slices(that's one_test_sample),every slice is an input
                one_test_sample = []
                csp_transpose = coded_sp.T  #36x512 36x128...
                for i in range(0, csp_transpose.shape[1] - FRAMES + 1, FRAMES):
                    t = csp_transpose[:, i:i + FRAMES]
                    t = normlizer.forward_process(t, speaker)
                    t = np.reshape(t, [t.shape[0], t.shape[1], 1])
                    one_test_sample.append(t)

                #target label 1->2, 2->3, 3->0, 0->1
                one_test_sample_label = np.zeros([len(one_test_sample), len(all_speaker)])
                temp_index = label_enc.transform([speaker])[0]
                temp_index = (temp_index + 2) % len(all_speaker)

                for i in range(len(one_test_sample)):
                    one_test_sample_label[i][temp_index] = 1

                #get conversion target name ,like SF1
                target_name = label_enc.inverse_transform([temp_index])[0]

                generated_results = model.test(one_test_sample, one_test_sample_label)

                reshpaped_res = []
                for one in generated_results:
                    t = np.reshape(one, [one.shape[0], one.shape[1]])
                    t = normlizer.backward_process(t, target_name)

                    reshpaped_res.append(t)
                #collect the generated slices, and concate the array to be a whole representation of the whole audio
                c = []
                for one_slice in reshpaped_res:
                    one_slice = np.ascontiguousarray(one_slice.T, dtype=np.float64)
                    decoded_sp = pyworld.decode_spectral_envelope(one_slice, SAMPLE_RATE, fft_size=FFTSIZE)
                    c.append(decoded_sp)

                concated = np.concatenate((c), axis=0)

                #f0 convert
                f0 = normlizer.pitch_conversion(f0, speaker, target_name)

                synwav = pyworld.synthesize(f0, concated, ap, fs)

                #remove synthesized wav paded length
                synwav = synwav[:-pad_length]

                #save synthesized wav to file
                wavname = f'{speaker}-{target_name}+{name}'
                wavpath = os.path.join(file_path, 'wavs')
                if not os.path.exists(wavpath):
                    os.makedirs(wavpath, exist_ok=True)
                librosa.output.write_wav(f'{wavpath}/{wavname}', synwav, sr=fs)
                print('============save converted audio============')

            print('============test finished!============')

        #====================save model=======================

        if epoch % 10 == 0 and epoch != 0:
            print('============save model============')
            model_path = os.path.join(file_path, 'model')

            if not os.path.exists(model_path):
                os.makedirs(model_path, exist_ok=True)

            print(f'save model: {model_path}')
            model.save(directory=model_path, filename=MODEL_NAME)

        end_time_epoch = time.time()
        time_elapsed_epoch = end_time_epoch - start_time_epoch

        print('Time Elapsed for This Epoch: %02d:%02d:%02d' % (time_elapsed_epoch // 3600, (time_elapsed_epoch % 3600 // 60),
                                                               (time_elapsed_epoch % 60 // 1)))


if __name__ == '__main__':

    processed_dir = './data/processed'
    test_wav_dir = './data/fourspeakers_test'

    parser = argparse.ArgumentParser(description='Train StarGAN Voice conversion model.')

    parser.add_argument('--processed_dir', type=str, help='train dataset directory that contains processed npy and npz files', default=processed_dir)
    parser.add_argument('--test_wav_dir', type=str, help='test directory that contains raw audios', default=test_wav_dir)

    argv = parser.parse_args()

    processed_dir = argv.processed_dir
    test_wav_dir = argv.test_wav_dir

    start_time = time.time()

    train(processed_dir, test_wav_dir)

    end_time = time.time()
    time_elapsed = end_time - start_time

    print('Training Time: %02d:%02d:%02d' % \
    (time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))
