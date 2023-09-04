# S4SpeechFormer

Extracting data from MELD:
1. Run extract_feature/mp4_to_wav.py on each of the dataset folders (dev, test, train)
2. Extract features with the chosen script in extract_feature directory
3. Run utils/lmdb_kit.py - make sure to run on all states