import sys, os
repo_root = r'C:/FakeNewsDetector'
sys.path.append(repo_root)

from src.data.fakeddit_dataset import get_fakeddit_dataset

csv_path = r'C:/FakeNewsDetector/fakeddit_processed.csv'
image_root = r'C:/FakeNewsDetector/data/images'

dataset = get_fakeddit_dataset(csv_path, image_root)
sample = dataset[0]
print('Lexicon emotion vector shape:', sample['lexicon_emotion'].shape)
print('Distil emotion embedding shape:', sample['distil_emotion'].shape)
