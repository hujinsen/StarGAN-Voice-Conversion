import numpy as np
import pyworld as pw
# import soundfile as sf
import tensorflow as tf
import os, shutil
import glob


def get_speakers(trainset: str = './data/fourspeakers'):
    '''return current selected speakers for training
        eg. ['SF2', 'TM1', 'SF1', 'TM2']
    '''
    p = os.path.join(trainset, "*")
    all_sub_folder = glob.glob(p)

    all_speaker = [os.path.normpath(s).rsplit(os.sep, maxsplit=1)[1] for s in all_sub_folder]

    return all_speaker


class Normalizer(object):
    '''Normalizer: convience method for fetch normalize instance'''

    def __init__(self, statfolderpath: str = './etc'):

        self.all_speaker = get_speakers()
        self.folderpath = statfolderpath

        self.norm_dict = self.normalizer_dict()

    def forward_process(self, x, speakername):
        mean = self.norm_dict[speakername]['coded_sps_mean']
        std = self.norm_dict[speakername]['coded_sps_std']
        mean = np.reshape(mean, [-1, 1])
        std = np.reshape(std, [-1, 1])
        x = (x - mean) / std

        return x

    def backward_process(self, x, speakername):
        mean = self.norm_dict[speakername]['coded_sps_mean']
        std = self.norm_dict[speakername]['coded_sps_std']
        mean = np.reshape(mean, [-1, 1])
        std = np.reshape(std, [-1, 1])
        x = x * std + mean

        return x

    def normalizer_dict(self):
        '''return all speakers normailzer parameter'''

        d = {}
        for one_speaker in self.all_speaker:

            p = os.path.join(self.folderpath, '*.npz')
            try:
                stat_filepath = [fn for fn in glob.glob(p) if one_speaker in fn][0]
            except:
                raise Exception('====no match files!====')
            print(f'found stat file: {stat_filepath}')
            t = np.load(stat_filepath)
            d_temp = t.f.arr_0.item()
            # print(d_temp.keys())

            d[one_speaker] = d_temp

        return d

    def pitch_conversion(self, f0, source_speaker, target_speaker):
        '''Logarithm Gaussian normalization for Pitch Conversions'''

        mean_log_src = self.norm_dict[source_speaker]['log_f0s_mean']
        std_log_src = self.norm_dict[source_speaker]['log_f0s_std']

        mean_log_target = self.norm_dict[target_speaker]['log_f0s_mean']
        std_log_target = self.norm_dict[target_speaker]['log_f0s_std']

        f0_converted = np.exp((np.ma.log(f0) - mean_log_src) / std_log_src * std_log_target + mean_log_target)

        return f0_converted


class GenerateStatics(object):

    def __init__(self, folder: str = './data/processed'):
        self.folder = folder

        self.all_speaker = get_speakers()

        #key is speaker(SF1, SF2...) and value is corresponding file list
        self.include_dict = {}
        for s in self.all_speaker:
            if not self.include_dict.__contains__(s):
                self.include_dict[s] = []

            for one_file in os.listdir(folder):
                if one_file.startswith(s) and one_file.endswith('npy'):
                    self.include_dict[s].append(one_file)
        # print(self.include_dict)

        self.include_dict_npz = {}
        for s in self.all_speaker:
            if not self.include_dict_npz.__contains__(s):
                self.include_dict_npz[s] = []

            for one_file in os.listdir(folder):
                if one_file.startswith(s) and one_file.endswith('npz'):
                    self.include_dict_npz[s].append(one_file)
        # print(self.include_dict_npz)

    @staticmethod
    def coded_sp_statistics(coded_sps):
        # sp shape (T, D)
        coded_sps_concatenated = np.concatenate(coded_sps, axis=1)
        coded_sps_mean = np.mean(coded_sps_concatenated, axis=1, keepdims=False)
        coded_sps_std = np.std(coded_sps_concatenated, axis=1, keepdims=False)
        return coded_sps_mean, coded_sps_std

    @staticmethod
    def logf0_statistics(f0s):
        log_f0s_concatenated = np.ma.log(np.concatenate(f0s))
        log_f0s_mean = log_f0s_concatenated.mean()
        log_f0s_std = log_f0s_concatenated.std()

        return log_f0s_mean, log_f0s_std

    def generate_stats(self, statfolder: str = './etc'):
        '''generate all user's statitics used for calutate normalized
           input like sp, f0
           step 1: generate coded_sp mean std
           step 2: generate f0 mean std
         '''
        etc_path = os.path.join(os.path.realpath('.'), statfolder)
        if not os.path.exists(etc_path):
            os.makedirs(etc_path, exist_ok=True)

        for one_speaker in self.include_dict.keys():
            coded_sps = []

            arr = self.include_dict[one_speaker]
            if len(arr) == 0:
                continue
            for one_file in arr:
                t = np.load(os.path.join(self.folder, one_file))
                # print(t.shape)
                coded_sps.append(t)

            coded_sps_mean, coded_sps_std = self.coded_sp_statistics(coded_sps)
            # print(f'sp_mean: {coded_sps_mean.shape} \
            # sp_std: {coded_sps_std.shape}')

            f0s = []
            arr01 = self.include_dict_npz[one_speaker]
            if len(arr01) == 0:
                continue
            for one_file in arr01:
                t = np.load(os.path.join(self.folder, one_file))
                d = t.f.arr_0.item()
                f0_ = np.reshape(d['f0'], [-1, 1])
                # print(f'f0 shape: {f0_.shape}')
                f0s.append(f0_)
            log_f0s_mean, log_f0s_std = self.logf0_statistics(f0s)
            print(log_f0s_mean, log_f0s_std)

            tempdict = {'log_f0s_mean': log_f0s_mean, 'log_f0s_std': log_f0s_std, 'coded_sps_mean': coded_sps_mean, 'coded_sps_std': coded_sps_std}

            filename = os.path.join(etc_path, f'{one_speaker}-stats.npz')
            print(f'save: {filename}')
            np.savez(filename, tempdict)


if __name__ == "__main__":
    pass