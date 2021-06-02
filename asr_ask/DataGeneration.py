import codecs
import soundfile as sf
from nemo.collections.tts.models.base import SpectrogramGenerator, Vocoder, TextToWaveform
import os

class dataloader():
    
    def __init__(self, root, train, target, audio):
        self.dataset = root
        self.train = train
        self.target = target
        self.audio = audio
        self.count = 1
        self.spectro = SpectrogramGenerator.from_pretrained('tts_en_tacotron2')
        self.vocoder = Vocoder.from_pretrained('tts_hifigan').eval().cuda()

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

                ft.write(orig + "\n")
                ftt.write(corr + "\n")
    
    def audio_process(self):
        with codecs.open(self.train, 'r', encoding='utf8') as f:
            for line in f:

                parsed = self.spectro.parse(line)
                spectrogram = self.spectro.generate_spectrogram(tokens=parsed)
                audio = self.vocoder.convert_spectrogram_to_audio(spec=spectrogram)[0]
                
                audio = audio.to('cpu').detach().numpy()

                sf.write(os.path.join(self.audio, str("audio_spec2wav_sample_"+ str(self.count)+ ".wav")), audio, 22050)

                self.count+= 1


if __name__ == '__main__':
    dataset_root = '/path/lang-8.train'
    train_train = '/path/train.train'
    target_train = '/path/target.train'
    audio_train = '/path/audio'

    obj = dataloader(dataset_root, train_train, target_train, audio_train)
    obj.text_process()
    obj.audio_process()