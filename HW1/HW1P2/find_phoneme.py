import os
import numpy as np
from collections import Counter

ROOT = r"C:\Users\xulunl\Desktop\Intro to DeepLearning\HW1\HW1P2\hw1p2_data\archive\11785-spring-2026-hw1p2"

PHONEMES = [
    '[SIL]',   'AA',    'AE',    'AH',    'AO',    'AW',    'AY',
    'B',     'CH',    'D',     'DH',    'EH',    'ER',    'EY',
    'F',     'G',     'HH',    'IH',    'IY',    'JH',    'K',
    'L',     'M',     'N',     'NG',    'OW',    'OY',    'P',
    'R',     'S',     'SH',    'T',     'TH',    'UH',    'UW',
    'V',     'W',     'Y',     'Z',     'ZH',    '[SOS]', '[EOS]'
]

transcript_dir = os.path.join(ROOT, "dev-clean", "transcript")

print(f"Looking for transcript files in {transcript_dir}")
print(f"Directory exists: {os.path.exists(transcript_dir)}\n")

all_phonemes = []
for filename in os.listdir(transcript_dir):
    if filename.endswith(".npy"):
        filepath = os.path.join(transcript_dir, filename)
        
        # Load the transcript indices
        transcript_phonemes = np.load(filepath, allow_pickle=True)

        # Map indices to phonemes
        all_phonemes.extend(transcript_phonemes.tolist())

print(f"Total phonemes found: {len(all_phonemes)}\n")

phoneme_counts = Counter(all_phonemes)
sorted_phonemes = sorted(phoneme_counts.items(), key=lambda x: x[1])

print("="*50)
print("TOP 5 MOST COMMON PHONEMES:")
print("="*50)
for phoneme, count in sorted_phonemes[-5:]:
    print(f"{phoneme}: {count}")

print("\n" + "="*50)
print("TOP 5 LEAST COMMON PHONEMES:")
print("="*50)
for phoneme, count in sorted_phonemes[:5]:
    print(f"{phoneme}: {count}")

print("\n" + "="*50)
print(f"ANSWER TO QUESTION 6: {sorted_phonemes[0][0]}")
print(f"Count: {sorted_phonemes[0][1]}")
print("="*50)