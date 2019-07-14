
## A semi-supervised sequence-to-sequence ASR model
A PyTorch implementation of the model proposed in [A Semi-Supervised Approach to Automatic Speech Recognition Training For the Icelandic Language](thesis.pdf)

## Architecture
Blablabla

* `src/lm.py`: The character level language model
* `src/asr.py`: The baseline ASR, similar to _Listen, Attend & Spell_.
* `src/text_autoencoder.py`: The `forward()` of the whole autoencoder as well as the architecture of the encoder.
* `src/speech_autoencoder.py`: The `forward()` of the whole autoencoder as well as the architecture of the utterance level encoder and the decoder.
* `src/discriminator.py`: The simple FF discriminator used for adversarial training.


## Setup
Install dependencies using `pip install -r requirements.txt`
The most notable dependencies here are
```
torch
tensorboardX
librosa
```
No further setup is required. The code will take care of checking for CUDA availability and there are no requirements in terms of the location of training data.

## Preprocessing
The training data consists of mel-scaled spectrograms and the corresponding text. Since the directory layout of datasets tend to differ, additional code might be needed depending on the dataset used. `src/preprocess.py` offers support for the _Málrómur_ dataset and generic datasets with a directory structure like
```
dataset_directory/
    wav_directory/
        utt_1.wav
        utt_2.wav
        ...
    txt_directory/
        utt_1.txt
        utt_2.txt
        ...
```
To preprocess Málrómur, run `python3 src/preprocess.py malromur -o=<out_directory> -index='<malromur_index_path> -wav_dir=<malromur_wav_dir>`. A generic dataset can be preprocessed similarilly with `src/preprocess.py generic`, see `src/preprocess.py` for detail. This will generate and store the preprocessed data under
```
./data/
    <out_directory>
        fbanks/
            utt_1.npy
            utt_2.npy
            ...
        index.tsv
```
where each line in `index.tsv` contains
`normalized_text, path_to_fbank, s_len, unpadded_num_frames, text_fname, wav_fname` for each utterance in the dataset. Each spectrogram is zero-padded at the end up to the maximum length of the whole dataset.

Depending on the use case and the dataset, 3 other functions in `src/preprocess.py` could be useful. No useful API is available for these three functions but they can be easily imported into any python script. These three functions are
* `sort_index()`: This sorts the index file of the preprocessed data by one of the column IDs mentioned above. To use some of the padding functions in PyTorch, data needs to be sorted by the length of the axis of the data that is to be padded. So this function can be used to sort the index by either the temporal length of the signal or the length of the corresponding text.
* `make_split()`: This splits the preprocessed data index file into two new files, training and validation indexes, given a training/validation split. The samples in each are randomly split.
* `subset_by_t()`: If say only 2 hours should be used for an experiment, this function can be used to randomly select utterances from the preprocessed data index that amount to 2 hours.

Options and information about input arguments can be displayed with `python3 src/preprocess.py -h` and e.g. `python3 src/preprocess.py malromur -h`
## Training
All training is contained in `src/trainer.py` and can be initiated from `src/train.py`. 7 types of training/testing are avilable via the `src/train.py t=<type>` positional argument

Type | Description | Identifier
------------ | -------------- | --------------
`t=ASRTrainer` | Trains the basline ASR | asr
`t=ASRTester` | Tests the baseline ASR | asr
`t=LMTrainer` | Trains the character level RNN LM | char_lm
`t=TAETrainer` | Trains the text autoencoder * | tae
`t=SAETrainer` | Trains the speech autoencoder * | sae
`t=AdvTrainer` | Peroforms adversarial training * | adv
`t=Seed` | Performs a combination of `TAETRainer`, `SAETrainer` and `AdvTrainer` to produce a seed model to further train the baseline * | n/a

(*): Will affect the parameters of the baseline ASR.

The identifier is used to seperate results generated (see usage [here](##results). All training runs require certain parameters, see `python3 src/train.py -h` for information but most notably a configuration `.yaml` file. An example of a configuration file is found [here](/conf/default.yaml) and detailed information here [here](./conf/README.md).

## Results
The results produced by training depend on the type of training. Each training type will
* Store tensorboard logging under `<logdir>/<name>/<identifier>`
* Save the most recent model at  `<ckpdir>/<name>/<identifier>.cpt`
* Save the best model at `<ckpdir>/<name>/<identifier>_best.cpt`

### Tensorboard
This project uses `TensorboardX` to visualize training progress, displaying hypothesis and alignment plots and more. If installed, all generated results can be loaded via `tensorboard --logdir='./<logdir>'` and then visiting `localhost:6006`. To limit loading, a specific experiment can also be loaded via e.g. `tensorboard --logdir='./<logdir>/<name>'`.


