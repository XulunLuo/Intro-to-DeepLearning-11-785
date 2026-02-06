import torch
import numpy as np
import os
import torch.nn as nn
from torch.utils.data import Dataset

PHONEMES = [
    '[SIL]',   'AA',    'AE',    'AH',    'AO',    'AW',    'AY',
    'B',     'CH',    'D',     'DH',    'EH',    'ER',    'EY',
    'F',     'G',     'HH',    'IH',    'IY',    'JH',    'K',
    'L',     'M',     'N',     'NG',    'OW',    'OY',    'P',
    'R',     'S',     'SH',    'T',     'TH',    'UH',    'UW',
    'V',     'W',     'Y',     'Z',     'ZH',    '[SOS]', '[EOS]'
]

class AudioDataset(Dataset):
    def __init__(self, root, partition = "train-clean-100", context = 20, subset = 1.0):
        self.context = context
        self.phonemes = PHONEMES

        # Data paths
        self.mfcc_dir = os.path.join(root, partition, "mfcc")
        self.transcript_dir = os.path.join(root, partition, "transcript")

        # Get sorted file lists
        mfcc_names = sorted(os.listdir(self.mfcc_dir))
        transcript_names = sorted(os.listdir(self.transcript_dir))

        # Apply subset
        subset_size = int(len(mfcc_names) * subset)
        mfcc_names = mfcc_names[:subset_size]
        transcript_names = transcript_names[:subset_size]

        assert len(mfcc_names) == len(transcript_names), "Mismatch in number of MFCC and transcript files."

        self.mfccs, self.transcripts = [], []

        print(f"Loading {partition} data...")
        for i in range (len(mfcc_names)):
            # Load MFCC features
            mfcc = np.load(os.path.join(self.mfcc_dir, mfcc_names[i]))

            # Normalize MFCC features
            mfcc_normalized = (mfcc - np.mean(mfcc, axis = 0)) / (np.std(mfcc, axis = 0) + 1e-10)
            mfcc_normalized = torch.tensor(mfcc_normalized, dtype = torch.float32)

            # Load transcript phoneme indices
            transcript = np.load(os.path.join(self.transcript_dir, transcript_names[i]), allow_pickle = True)

            # Remove first and last
            transcript = transcript[1:-1]

            # Map phonemes 
            transcript_indices = [self.phonemes.index(p) for p in transcript]
            transcript_indices = torch.tensor(transcript_indices, dtype = torch.long)

            self.mfccs.append(mfcc_normalized)
            self.transcripts.append(transcript_indices)

        # Concatenate all data
        self.mfcc = torch.cat(self.mfcc, dim = 0)
        self.transcripts = torch.cat(self.transcripts, dim = 0)
        self.length = len(self.mfccs)

        # Padding those zeros for context
        self.mfccs = nn.functional.pad(self.mfccs, (0, 0, self.context, self.context))
        print(f"Loaded {self.length} frames")

    def __len__(self):
        return self.length
    
    # Returns one training example
    def __getitem__(self, ind):
        # Get frames with context from before and after the center frame
        window_size = 2 * self.context + 1 

        start = ind
        end = ind + window_size

        frames = self.mfcc[start:end]









            
