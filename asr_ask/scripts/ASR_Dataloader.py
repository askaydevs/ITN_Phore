#importing packages
import codecs
from glob import glob
import os
from torch.utils.data import Dataset
import librosa


class dataloader(Dataset):

    #initialization of location for audio and transcript files
    def __init__(self, root, mode= 'train'):
        self.root = root
        self.mode = mode
        self._init_dataset()
    
    #abstract audio and transcript data from the initialized location
    def _init_dataset(self):
        self.data = []
        self.label = []


        if self.mode == 'train':
            self.data = sorted(glob(os.path.join(self.root, self.mode, 'audio', '*.wav')), key= lambda x: x[-8:-4])
            with codecs.open(os.path.join(self.root, self.mode, 'transcript', 'train.train')) as f:
                for line in f:
                    self.label.append(line[:-2])
        elif self.mode == 'test':
            self.data = sorted(glob(os.path.join(self.root, self.mode, 'audio', '*.wav')))
            with codecs.open(os.path.join(self.root, self.mode, 'transcript', 'test.test')) as f:
                for line in f:
                    self.label.append(line[:-2])
        else:
            print("No Such Dataset Mode")
            return None

    #gets item when index is specified
    def __getitem__(self, index):
        audio = self.data[index]
        target = self.label[index]
        duration = librosa.core.get_duration(filename=audio)
        
        return audio, duration, target

    #gets length of dataset
    def __len__(self):
        return len(self.data)



#the below set of lines are used to check the functioning of the code
#if __name__ == '__main__':

#path where train and test folder (containing transcripts and audio files) are present
#    root = 'path'

#    obj = dataloader(root)
#    obj._init_dataset()

#    print(obj.__len__())
#    print(obj.__getitem__(14))