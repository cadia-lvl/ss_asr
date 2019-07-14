# Parameters
A default parameter configuration is available in `default.yaml`. The parameters are categorized after model components where e.g. `asr.opt` contains parameters relvant for the ASR optimizer, `asr.mdl` contains ASR model parameters and `asr` contains other parameters in general.

A couple of other hyperparameters that are related to preprocessing and more technial details of training are found in `./src/preprocess.py`.
## Common parameters
These parameters should be configured for each model component indvidually.

Parameter | Description
------------ | --------------
opt.type | The optimizer to use for training, see PyTorch documentation for more information
opt.learning_rate | The initial learning rate to use for the optimizer. Depending on the optimizer used, this might not be applicable but is sufficient for e.g. ADAM or ADADELTA.
train_index * | Path to training index produced by `./src/preprocess.py`.
valid_index * | Path to validation index
test_index  * | Path to test index

(*) Currently, the baseline ASR is the only model that supports testing. It is important that these indexes are correctly sorted since we use `torch.pack_padded_sequences` which speeds up training. The sequences to train on must be sorted in decreasing order of original length on the padded axis. So, for e.g. the baseline ASR the index must be sorted by `unpadded_num_frames` and `.src.preprocess.sort_index() can be used exactly for this.

In most cases the index should be sorted by `unpadded_num_frames` but if training is performed on padded text tokens, then sort the index by `s_len` using the same function.

Some additional parameters can be set for each model component seperately and have default values if not set.

Parameter | Description | Default value
------------ | -------------- | --------------
valid_step | Number of steps between validation measurements | 500
save_step | Number of steps between saving the most current version of the model | 1000
logging_step | Number of steps between verbose logging. Type of training determines what type of logging is output | 250
train_batch_size | Training batch size | 32
valid_batch_size | Validation batch size | 32
test_batch_size | Test batch size | 1
n_epohcs | Number of training epochs | 5



## ASR baseline specific parameters
Parameter | Description
------------ | --------------
mdl.encoder_state_size | The state size of each pBLSTM unit in the ASR encoder
mdl.mlp_out_size | Output dimensions of the Φ and Ψ  attention projection networks
mdl.decoder_state_size | The state sie of each LSTM unit in the ASR decoder
mdl.tf_rate | Teacher forcing rate of the decoder
mdl.feature_dim | The feature dimension of the input. This should match with `.src.preprocess.N_DIMS`
decode_beam_size | Number of hypothesis to consider at each level of beam search
decode_lm_weight | The weight that determines the influence of the language model during decoding. See thesis for more information
wer_step | The number of steps between each measure of WER on the training set.

## Speech Autoencoder specific parameters
The speech autoencoder contains a CNN encoder. Here we only consider CNNs with 3 layers of convolutional -> batch norm -> RELU -> max pool.

Parameter | Description
------------ | --------------
mdl.kernel_sizes | The 2D kernel sizes of each convolutional layer.
mdl.num_filters | The number of filters in each convolutional layer
mdl.pool_kernel_sizes | The 2D pooling kernels in each pooling layer. We aim to pool over the entire feature in the last pooling layer so the last pool kernel size has to be chosen with that in mind. In the default configuration we choose [2000, 40] since it approximately the size of the largest Malromur utterance.

## Text Autoencoder specific parameters
Parameter | Description
------------ | --------------
mdl.state_size | The state size of each BLSTM unit in the text encoder
mdl.emb_dim | The character embedding dimensionality
mdl.num_layers | Number of BLSTM layers in the encoder

## Adversarial training specific parameters
For adversarial training we configure the optimizers of both the discriminator, __D_optimizer__, and the generator, __G_optimizer__.

Parameter | Description
------------ | --------------
mdl.hidden_dim | The hidden dimension of the simple discriminator.

## Language model specific parameters
Parameter    | Description
------------ | --------------
mdl.hidden_size | The state size of the RNN used in the language model. This is also used as the output dimension of the character embeddings used in the language model.
mdl.tf_rate | The teacher forcing rate used during decoding.
chunk_size | The size of each training sample.

## Seed training
Parameter    | Description
------------ | --------------
its | The number of iterations of the main training loop to perform

## Preprocessing parameters
This is a short description of the parameters that are found in `./src/preprocess.py`.

Parameter    | Description
------------ | --------------
CHARS | The latin alphabet and digits
ICE_CHARS | Special Icelandic characters
SPECIAL_CHARS | Some tokens that are likely to affect pronounciation.
SOS_TKN | Appended to the start of each sentence
EOS_TKN | Appended to the end of each sentence
UNK_TKN | Tokens in the unprocessed text that are not covered by CHARS/ICE_CHARS/SPECIAL_CHARS are replaced with this token
N_JOBS | no. jobs to run in parallel when writing the index
N_DIMS | no. frequency coefficients in the spectrograms
WIN_SIZE | size of window in STFT
STRIDE | stride of the window
TEXT_XTSN |extension of token files (if applicable)
