import librosa
import numpy as np
import os
import pyworld
import pyworld as pw
import glob
from utility import *
import argparse

FEATURE_DIM = 36
SAMPLE_RATE = 16000
FRAMES = 512
FFTSIZE = 1024
SPEAKERS_NUM = 4

EPSILON = 1e-10
MODEL_NAME = 'starganvc_model'

def load_wavs(dataset: str, sr):
    '''
    data dict contains all audios file path
    resdict contains all wav files   
    '''
    data = {}
    with os.scandir(dataset) as it:
        for entry in it:
            if entry.is_dir():
                data[entry.name] = []
                # print(entry.name, entry.path)
                with os.scandir(entry.path) as it_f:
                    for onefile in it_f:
                        if onefile.is_file():
                            # print(onefile.path)
                            data[entry.name].append(onefile.path)
    print(f'loaded keys: {data.keys()}')
    #data like {TM1:[xx,xx,xxx,xxx]}
    resdict = {}

    cnt = 0
    for key, value in data.items():
        resdict[key] = {}

        for one_file in value:
            
            filename = one_file.split('/')[-1].split('.')[0] #like 100061
            newkey = f'{filename}'
            wav, _ = librosa.load(one_file, sr=sr, mono=True, dtype=np.float64)
            
            resdict[key][newkey] = wav
            # resdict[key].append(temp_dict) #like TM1:{100062:[xxxxx], .... }
            print('.', end='')
            cnt += 1

    print(f'\nTotal {cnt} aduio files!')
    return resdict

def wav_to_mcep_file(dataset: str, sr=16000, ispad:bool=False,processed_filepath: str = './data/processed'):
    '''convert wavs to mcep feature using image repr'''
    #if no processed_filepath, create it ,or delete all npz files
    if not os.path.exists(processed_filepath):
        os.makedirs(processed_filepath)
    else:
        filelist = glob.glob(os.path.join(processed_filepath, "*.npy"))
        for f in filelist:
            os.remove(f)

    allwavs_cnt = len(glob.glob(f'{dataset}/*/*.wav'))
    # allwavs_cnt = allwavs_cnt//4*3 * 12+200 #about this number not precise
    print(f'Total {allwavs_cnt} audio files!')

    d = load_wavs(dataset, sr)
    cnt = 1 # 

    for one_speaker in d.keys():
        for audio_name, audio_wav in d[one_speaker].items():
            # cal source audio feature
            audio_mcep_dict = cal_mcep(audio_wav, fs=sr,ispad=ispad, frame_period=0.005, dim=FEATURE_DIM)
            newname = f'{one_speaker}-{audio_name}'

            #save the dict as npz
            file_path_z = f'{processed_filepath}/{newname}'
            print(f'save file: {file_path_z}')
            np.savez(file_path_z, audio_mcep_dict)

            #save every  36*FRAMES blocks
            print(f'audio mcep shape {audio_mcep_dict["coded_sp"].shape}')
            
            #TODO step may be FRAMES//2
            for start_idx in range(0, audio_mcep_dict["coded_sp"].shape[1] - FRAMES + 1, FRAMES):
                one_audio_seg = audio_mcep_dict["coded_sp"][:, start_idx : start_idx+FRAMES]

                if one_audio_seg.shape[1] == FRAMES:

                    temp_name = f'{newname}_{start_idx}'
                    filePath = f'{processed_filepath}/{temp_name}'

                    print(f'[{cnt}:{allwavs_cnt}]svaing file: {filePath}.npy')
                    np.save(filePath, one_audio_seg)
            cnt += 1

def cal_mcep(wav_ori, fs=SAMPLE_RATE, ispad=False, frame_period=0.005, dim=FEATURE_DIM, fft_size=FFTSIZE):
    '''cal mcep given wav singnal
        the frame_period used only for pad_wav_to_get_fixed_frames
    '''
    if ispad:
        wav,pad_length = pad_wav_to_get_fixed_frames(wav_ori,frames=FRAMES, frame_period=frame_period, sr=fs)
    else:
        wav = wav_ori
    #Harvest F0 extraction algorithm.
    f0, timeaxis = pyworld.harvest(wav, fs, f0_ceil = 500.0)

    #CheapTrick harmonic spectral envelope estimation algorithm.
    sp = pyworld.cheaptrick(wav, f0, timeaxis, fs,fft_size=fft_size)
    
    #D4C aperiodicity estimation algorithm.
    ap = pyworld.d4c(wav, f0, timeaxis, fs, fft_size=fft_size)
    #feature reduction nxdim
    coded_sp = pyworld.code_spectral_envelope(sp, fs, dim)
    #log 
    coded_sp = coded_sp.T # dim x n

    res = {
        'f0':f0, #n
        'ap':ap, #n*fftsize//2+1
        'sp':sp, #n*fftsize//2+1
        'coded_sp':coded_sp, #dim * n
    }
    return res


def pad_wav_to_get_fixed_frames(x: np.ndarray, frames:int=128, frame_period:float=0.005, sr:int=16000):
    #one frame's points
    frame_length = frame_period * sr
    #frames points
    frames_points = frames * frame_length
    
    wav_len = len(x)
    
    # pad amount
    pieces = wav_len // frames_points

    need_pad = 0
    if wav_len % frames_points != 0:
        #can't devide need pad
        need_pad = int((pieces+1) * frames_points-wav_len)

    afterpad_len = wav_len + need_pad
    # print(f'need pad: {need_pad}, after pad: {afterpad_len}')
    #padding process
    tempx = x.tolist()
    
    if need_pad <= len(x):
        tempx.extend(x[:need_pad])
    else:
        temp1, temp2= need_pad//len(x), need_pad/len(x)
        tempx = tempx * (temp1+1)
        samll_pad_len =int(np.ceil((temp2-temp1) * len(x)))
        tempx.extend(x[:samll_pad_len]) 

        diff = 0
        if afterpad_len != len(tempx):
            diff = afterpad_len - len(tempx)
        if diff > 0:
            tempx.extend(tempx[:diff])
        elif diff < 0:
            tempx = tempx[:diff]

    # print(f'padding length: {len(x)}-->length: {len(tempx)}')
    #remove last point for calculate convience:the frame length are 128*(some integer).
    tempx = tempx[:-1] 
    
    return np.asarray(tempx, dtype=np.float), need_pad


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Convert the wav waveform to mel-cepstral coefficients(MCCs)\
    and calculate the speech statistical characteristics')
    
    input_dir = './data/fourspeakers'
    output_dir = './data/processed'
    ispad = False
   

    parser.add_argument('--input_dir', type = str, help = 'the direcotry contains data need to be processed', default = input_dir)
    parser.add_argument('--output_dir', type = str, help = 'the directory stores the processed data', default = output_dir)
    parser.add_argument('--ispad', type = bool, help = 'whether to pad the wavs  to get fixed length MCEP', default = ispad)
    

    argv = parser.parse_args()
    input_dir = argv.input_dir
    output_dir = argv.output_dir
    ispad = argv.ispad
        
    wav_to_mcep_file(input_dir, SAMPLE_RATE, ispad=ispad, processed_filepath=output_dir)

    #input_dir is train dataset. we need to calculate and save the speech\
    # statistical characteristics for each speaker.
    generator = GenerateStatics(output_dir)
    generator.generate_stats()
    