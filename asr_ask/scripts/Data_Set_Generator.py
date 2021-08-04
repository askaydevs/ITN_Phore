#importing packages
import codecs
import soundfile as sf
from nemo.collections.tts.models.base import SpectrogramGenerator, Vocoder
import os
from glob import glob
import nemo.collections.asr as nemo_asr


class datagenerator():
    
    #initialization of file locations and pre-trained model for audio generation
    def __init__(self, root, train, target, audio, transcribe):
        self.dataset = root #lang-8 en file location
        self.train = train #path to grammtically incorrect sentences
        self.target = target #path to grammatically correct sentences
        self.audio = audio #path to audio files storage location
        self.transcribe = transcribe #path to store transcibed sentences from audio files
        self.audio_files = []
        self.count = 1

        #text-to-speech can be done by converting text to spectrogram and then to audio.
        #tacoton-2 model is used to generate spectrogram and hifigan is used for audio file generation from spectrogram
        #self.spectro = SpectrogramGenerator.from_pretrained('tts_en_tacotron2').eval().cuda() #for spectrogram generation
        #self.vocoder = Vocoder.from_pretrained('tts_hifigan').eval().cuda() #for audio generation from spectrogram
        self.citrinet = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_en_citrinet_1024").eval().cuda() #asr pre-trained model

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
                ftt.write(orig+ "\n")
                ft.write(corr+ "\n")
    
    #function generates audio files from grammatically incorrect sentences
    def audio_process(self):
        with codecs.open(self.train, 'r', encoding='utf8') as fin:
            for line in fin:

                parsed = self.spectro.parse(line)
                spectrogram = self.spectro.generate_spectrogram(tokens=parsed)
                audio = self.vocoder.convert_spectrogram_to_audio(spec=spectrogram)[0]
                
                audio = audio.to('cpu').detach().numpy()
                audio_path = os.path.join(self.audio, str("audio_spec2wav_sample_"+ str("{:07d}".format(self.count))+ ".wav"))

                sf.write(audio_path, audio, 22050)

                self.count+= 1
        
    #function used for automatic speech recognition
    def speech_recognition(self):
        #obtain all audio files
        self.audio_files = sorted(glob(os.path.join(self.audio, '*.wav')), key= lambda x: x[-11:-4])
        batch_size = 1000

        with codecs.open(self.transcribe, 'w', encoding='utf8') as f:
            for i in range(len(self.audio_files)//batch_size):
                batch = self.audio_files[i*batch_size: (i+1)*batch_size]

                output = self.citrinet.transcribe(batch)

                for a in output:
                    f.write(a + '\n')


if __name__ == '__main__':

    files = ['train', 'test']

    for i in files:

        #path to lang-8 dataset, grammatically incorrect sentences, correct sentences and audio files respectively
        dataset_root = path+ i+ '/transcript/sample_entries'
        train = path+ i+ '/transcript/train'
        target = path+ i+ '/transcript/target'
        audio = path+ i+ '/audio'
        transcribe = path+ i+  '/transcribed/transcribed'


        #declaring object of class datagenerator
        obj = datagenerator(dataset_root, train, target, audio, transcribe)
        #function used to obtain grammatically incorrect and correct sentences from raw data
        #obj.text_process()
        #function used to convert incorrect sentences to audio format for training ASR model
        #obj.audio_process()
        #function used to transcribe audio files into text
        obj.speech_recognition()
