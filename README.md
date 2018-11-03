## StarGAN Voice Conversion

----



This is a tensorflow implementation of the paper: [StarGAN-VC: Non-parallel many-to-many voice conversion with star generative adversarial networks](https://arxiv.org/abs/1806.02169).

In the experiment, we choose **four speakers** from vcc 2016 dataset.  We  move the corresponding folder(eg. SF1,SF2,TM1,TM2 from vcc2016 training set. ) to ./data/fourspeakers. Then we run preprocess.py to generate npy files and statistical characteristics for each speaker. And then we choose some test examples( from vcc2016 evaluation set)  and put them in ./data/fourspeaker_test. Now we can train our model.

## Dependencies

- Python 3.6 (or higher)
- tensorflow 1.7
- librosa
- pyworld
- tensorboard
- scikit-learn

## Usage

#### Downloading the dataset

The following line will download the vcc 2016 dataset to the current directory and create ./data/fourspeakers and ./data/fourspeakers_test.

```
python download.py --datasets vcc2016 --train_dir ./data/fourspeakers --test_dir ./data/fourspeakers_test

For simplicity use:
python download.py 
```

When download finished, we **manually select some speakers** from ./data/vcc2016_training and ./data/.

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

Extract features from audios. We extract 36 Mel-cepstral coefficients(MCEPs) and frame length is 512. The features are stored as npy and npz files. We also calculate the statistical characteristics for each speaker.

```
python preprocess.py --input_dir ./data/fourspeakers --output_dir ./data/processed --ispad True

For simplicity use:
python preprocess.py
```

This process may take a few minutes !

Note  that test set doesn’t need preprocess.

#### Train

We read npy files from ./data/processed to train and raw wav files from ./data/fourspeakers_test to randomly generate some converted samples during training.

```
python train.py --processed_dir ./data/processed --test_wav_dir ./data/fourspeakers_test

For simplicity use:
python train.py
```

#### Convert

Restore model from model_dir, convert source_speaker’s speech to target_speaker’s speech. The results are strored in ./converted_voices

```
python convert.py --model_dir ./your_model_dir  --source_speaker SF1 --target_speaker TM1
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

---

If you feel this repo is good, please  **star**  ! 

Your encouragement is my biggest motivation!