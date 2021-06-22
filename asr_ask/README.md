# ASR PROJECT

This project aims at removing grammatical errors from the text transcribed from audio.

Dataset used for ASR training - [Lang-8 English](https://sites.google.com/site/naistlang8corpora/)

   &nbsp;&nbsp;Input ---> raw text from lang-8 english dataset <br/>
   
   &nbsp;&nbsp;Output ---> json file with data stored in the format shown below <br/>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{"audio_filepath": "path/to/audio.wav", "duration": duration, "text": "sentence used to generate audio"} -- (1)

**SCRIPT DESCRIPTION**

1. _[Data_Set_Generator.py](https://github.com/askaydevs/ITN_Phore/blob/asr/asr_ask/scripts/DataGenerator.py)_

    * _init_ method---> initializes paths to lang-8 dataset, grammatically incorrect sentences, correct sentences, audio files, json file respectively
 
    * _text_process_ method---> obtains grammatically incorrect and correct sentences

    * _audio_generator_ method ---> generates audio.wav files using grammatically incorrect sentences
      * _Tactotron-2_ pre-trained model --> generates spectrogram from text input
      * _higigan_ pre-trained model --> generates .wav files from spectrogram
      * stores data in suitable format(1) for ASR model training
