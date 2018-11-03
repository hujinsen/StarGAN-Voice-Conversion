## StarGAN Voice Conversion

----



This is a tensorflow implementation of the paper: [StarGAN-VC: Non-parallel many-to-many voice conversion with star generative adversarial networks](https://arxiv.org/abs/1806.02169).

In the experiment, we choose four speakers from vcc 2016 dataset.  We  move the corresponding folder(eg. SF1,SF2,TM1,TM2) to ./data/fourspeakers. Then we run preprocess.py to generate npy files and statistical characteristics for each speaker. Before training, we choose some test examples from four speakers and put them in ./data/fourspeaker_test. Now we can train our model.

## Dependencies

- Python 3.6 (or higher)
- tensorflow 1.7
- librosa
- pyworld
- tensorboard
- scikit-learn

## Usage

#### Downloading the dataset

The following line will download the vcc 2016 dataset to the current directory and create train_dir and test_dir.

```
python download.py --datasets vcc2016 --train_dir ./data/fourspeakers --test_dir ./data/fourspeakers_test
```

After downloaded, we need manually select some speakers from unziped train set to train_dir, and unziped test set to test_dir.

Now The data directory looks like this:

```
data
├── fourspeakers
│   ├── SF1
│   ├── SF2
│   ├── TM1
│   └── TM2
├── fourspeakers_test
│   ├── SF1
│   ├── SF2
│   ├── TM1
│   └── TM2
```



#### Preprocess dataset

We use 36 Mel-cepstral coefficients(MCEPs) and frame length is 512. The processed raw wav files are stored as npy and npz files. We also calculate the statistical characteristics for each speaker.

```python
python preprocess.py --input_dir ./data/fourspeakers --output_dir ./data/processed --ispad True
```

#### train

We read npy files from ./data/processed and raw wav files from ./data/fourspeaker_test. Note  that test set doesn’t need preprocess.

```python
python train.py --processed_dir ./data/processed --test_wav_dir ./data/fourspeaker_test
```

#### convert

Restore model from model_dir, convert SF1’s speech to TM1’s speech, store the result in output_dir.

```
python convert.py --model_dir ./your_model_dir --test_dir ./data/fourspeaker_test --output_dir ./converted --source_speaker SF1 --target_speaker TM1
```



## Summary

The network structure shown as follows:

![Snip20181102_2](./imgs/Snip20181102_2.png)





## Reference

[CycleGAN-VC code](https://github.com/leimao/Voice_Converter_CycleGAN)

[pytorch StarGAN-VC code](https://github.com/liusongxiang/StarGAN-Voice-Conversion)

[StarGAN code](https://github.com/taki0112/StarGAN-Tensorflow)

[StarGAN-VC paper](https://arxiv.org/abs/1806.02169)

[StarGAN paper](https://arxiv.org/abs/1806.02169)

[CycleGAN paper](https://arxiv.org/abs/1703.10593v4)

