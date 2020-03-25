import os
import random
import zipfile
import argparse
import zipfile
import urllib.request as req
import ssl
from threading import Thread
from queue import SimpleQueue as Queue


def unzip(zip_filepath, dest_dir='./data'):
    with zipfile.ZipFile(zip_filepath) as zf:
        zf.extractall(dest_dir)
    print("Extraction complete!")


def download_vcc2016():
    datalink = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/2211/"
    data_files = ['vcc2016_training.zip', 'evaluation_all.zip']

    if os.path.exists(data_files[0]) or os.path.exists(data_files[1]):
        print("File already exists!")
        return

    trainset = f'{datalink}{data_files[0]}'
    evalset = f'{datalink}{data_files[1]}'

    print('Start download dataset...')

    th = Thread(target=download_file, args=[trainset])
    th.start()
    download_file(evalset)
    th.join()

    unzip(data_files[0])
    unzip(data_files[1])

    print('Finish download dataset...')

def download_file(url: str, out_path: str = None, buffer_size: int = 10*(1024**2)):
    data = Queue()
    def _download():
        b = data.get()
        with open(out_path or url.split('/')[-1], 'wb') as o:
            while b:
                o.write(b)
                b = data.get()
    
    s = ssl.SSLContext()
    f = req.urlopen(url, context=s)
    th = Thread(target=_download)
    th.start()
    b = f.read(buffer_size)
    while b:
        data.put(b)
        b = f.read(buffer_size)
    data.put([])
    th.join()

def create_dirs(trainset: str = './data/fourspeakers', testset: str = './data/fourspeakers_test'):
    '''create train test dirs'''
    if not os.path.exists(trainset):
        print(f'create train set dir {trainset}')
        os.makedirs(trainset, exist_ok=True)

    if not os.path.exists(testset):
        print(f'create test set dir {testset}')
        os.makedirs(testset, exist_ok=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Download  voice conversion datasets.')

    datasets_default = 'vcc2016'
    train_dir = './data/fourspeakers'
    test_dir = './data/fourspeakers_test'
    parser.add_argument('--datasets', type=str, help='Datasets available: vcc2016', default=datasets_default)

    parser.add_argument('--train_dir', type=str, help='trainset directory', default=train_dir)
    parser.add_argument('--test_dir', type=str, help='testset directory', default=test_dir)

    argv = parser.parse_args()

    datasets = argv.datasets
    create_dirs(train_dir, test_dir)

    if datasets == 'vcc2016' or datasets == 'VCC2016':
        download_vcc2016()
    else:
        print('Dataset not available.')
