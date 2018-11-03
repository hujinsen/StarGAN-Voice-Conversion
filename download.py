import os
import random
from urllib.request import urlretrieve
import zipfile
import argparse
import shlex, subprocess
import zipfile


def unzip(zip_filepath, dest_dir='./data'):
    with zipfile.ZipFile(zip_filepath) as zf:
        zf.extractall(dest_dir)
    print("Extraction complete!")

def download_vcc2016():
    datalink="https://datashare.is.ed.ac.uk/bitstream/handle/10283/2211/"
    data_files = ['vcc2016_training.zip', 'evaluation_all.zip']

    if os.path.exists(data_files[0]) or os.path.exists(data_files[1]):
        print("File already exists!")
        return

    trainset = f'{datalink}/{data_files[0]}'
    evalset = f'{datalink}/{data_files[1]}'

    train_comm = f'wget {trainset}'
    eval_comm = f'wget {evalset}'

    train_comm = shlex.split(train_comm)
    eval_comm = shlex.split(eval_comm)

    print('Start download dataset...')
    
    subprocess.run(train_comm)
    subprocess.run(eval_comm)

    unzip(data_files[0])
    unzip(data_files[1])
    
    print('Finish download dataset...')

def create_dirs(trainset: str='./data/fourspeakers', testset: str='./data/fourspeakers_test'):
    '''create train test dirs'''
    if not os.path.exists(trainset):
        print(f'create train set dir {trainset}')
        os.makedirs(trainset, exist_ok=True)

    if not os.path.exists(testset):
        print(f'create test set dir {testset}')
        os.makedirs(testset, exist_ok=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Download  voice conversion datasets.')

    datasets_default = 'vcc2016'
    train_dir = './data/fourspeakers'
    test_dir = './data/fourspeakers_test'
    parser.add_argument('--datasets', type = str, help = 'Datasets available: vcc2016', default = datasets_default)
    
    parser.add_argument('--train_dir', type = str, help = 'trainset directory', default = train_dir)
    parser.add_argument('--test_dir', type = str, help = 'testset directory', default = test_dir)

    argv = parser.parse_args()

    datasets = argv.datasets
    create_dirs(train_dir, test_dir)

    if datasets == 'vcc2016' or datasets == 'VCC2016':
        download_vcc2016()
    else:
        print('Dataset not available.')

   
