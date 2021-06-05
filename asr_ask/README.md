# ASR PROJECT

This project aims at removing grammatical errors from the text transcribed from audio.

The 1st step towards accomplishing the task is to train the model for speech-to-text detection. To do so, audio is synthesized from the grammatically incorrect sentences from the [Lang-8 English](https://sites.google.com/site/naistlang8corpora/) dataset.

**SCRIPT DESCRIPTION**

1. _[DataGenerator.py](https://github.com/askaydevs/ITN_Phore/blob/asr/asr_ask/scripts/DataGenerator.py)_

The datagenerator class contains _text_process function_ which extracts the grammatically incorrect and correct sentences and saves them in separate files whose paths are declared as train.train and target.train respectively in the main function. 
The _audio_generator function_ reads the file that contains the grammatically incorrect sentences created from the previous function and converts it into an audio file. The audio files are saved in audio_train path declared in the main function.

The audio is synthesized by using two pre-trained models, where one model converts text to spectrogram and the other converts spectrogram into .wav files. _Tactotron-2_ model is used for spectrogram generation and _higigan_ model is used for audio file generation.

Once the audio files are created, they are used to train the ASR model against the incorrect sentences. In addition to the ASR model, a Grammatical Error Correction (GEC) model and an ITN model are used to remove the grammatical errors from the sentences transcribed from the audio and convert the text from spoken form to written form, respectively.
