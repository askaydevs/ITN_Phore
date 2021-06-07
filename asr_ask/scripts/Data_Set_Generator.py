#importing packages
import codecs
import soundfile as sf
from nemo.collections.tts.models.base import SpectrogramGenerator, Vocoder
import os
import json
import librosa

class datagenerator():
    
    #initialization of file locations and pre-trained model for audio generation
    def __init__(self, root, train, target, audio, manifest):
        self.dataset = root
        self.train = train
        self.target = target
        self.audio = audio
        self.manifest_path = manifest
        self.count = 1

        #text-to-speech can be done by converting text to spectrogram and then to audio.
        #tacoton-2 model is used to generate spectrogram and hifigan is used for audio file generation from spectrogram
        self.spectro = SpectrogramGenerator.from_pretrained('tts_en_tacotron2')
        self.vocoder = Vocoder.from_pretrained('tts_hifigan').eval().cuda()

    #function obtains grammatically incorrect and correct sentences
    def text_process(self):
        with codecs.open(self.dataset, 'r', encoding='utf8') as f, codecs.open(self.train, 'w', encoding='utf8') as ftt, codecs.open(self.target, 'w', encoding='utf8') as ft:
            for line in f:
                line = line.rstrip()
                cols = line.split('\t')
                if len(cols) > 4:
                    orig = cols[4]
                    corr = cols[4]
                if len(cols) > 5:
                    corr = cols[5]

                #orig is the original grammatically incorrect sentence and corr is the grammtically correct sentence
                ftt.write(orig + "\n")
                ft.write(corr + "\n")
    
    #function generates audio files from grammatically incorrect sentences
    def audio_process(self):
        with codecs.open(self.train, 'r', encoding='utf8') as fin, codecs.open(self.manifest_path, 'w') as fout:
            for line in fin:

                parsed = self.spectro.parse(line)
                spectrogram = self.spectro.generate_spectrogram(tokens=parsed)
                audio = self.vocoder.convert_spectrogram_to_audio(spec=spectrogram)[0]
                
                audio = audio.to('cpu').detach().numpy()
                audio_path = os.path.join(self.audio, str("audio_spec2wav_sample_"+ str("{:04d}".format(self.count))+ ".wav"))

                sf.write(audio_path, audio, 22050)

                duration = librosa.core.get_duration(filename=audio_path)

                metadata = {
                    "audio_filepath": audio_path,
                    "duration": duration,
                    "text": line
                }
                json.dump(metadata, fout)
                fout.write('\n')

                self.count+= 1


if __name__ == '__main__':

    #path to lang-8 dataset, grammatically incorrect sentences, correct sentences and audio files respectively
    #Similarly, paths to test data can be created
    dataset_root = '/path/train/transcript/testing_lang-8.train'
    train_train = '/path/train/transcript/train.train'
    target_train = 'path/train/transcript/target.train'
    audio_train = 'path/train/audio'
    manifest = '/path/train/transcript/train_dataset.json'

    #declaring object of class datagenerator
    obj = datagenerator(dataset_root, train_train, target_train, audio_train, manifest)
    #function used to obtain grammatically incorrect and correct sentences from raw data
    obj.text_process()
    #function used to convert incorrect sentences to audio format for training ASR model
    obj.audio_process()
