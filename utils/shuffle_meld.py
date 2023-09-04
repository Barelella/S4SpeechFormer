import pandas as pd
import os
import shutil
label_conveter = {'neutral': 0, 'anger': 1, 'joy': 2, 'sadness': 3, 'surprise': 4, 'disgust': 5, 'fear': 6}

def sample_df(df, num_samples):
    for label in label_conveter.keys():
        df_label = df[df.label == label]
        df_sampled = df_label.sample(n=num_samples//7)
        if label_conveter[label] == 0:
            output_df = df_sampled
        else:
            output_df = pd.concat([output_df, df_sampled])
    output_df = output_df.sample(frac=1)
    return output_df

def copy_wav_files_in_df(df, src_path, dst_path):
    for _, row in df.iterrows():
        file_name = row["name"] + ".wav"
        shutil.copy(src_path + file_name, dst_path + file_name)
        print(f"Copied {file_name}")
        
def sample_data_set(meta_csv_file, num_train_samples, num_test_samples, num_dev_samples):
    df = pd.read_csv(meta_csv_file)
    
    src_path = "C:/Users/Ella bar-El/Documents/Deep_Learning/SpeechFormerProject/MELD.Raw/train/wav/"
    dst_path = "C:/Users/Ella bar-El/Documents/Deep_Learning/SpeechFormerProject/MELD.Raw/train/wav_800/"
    # os.mkdir(dst_path) 
    df_train = df[df.state == "train"]
    sampled_train_df = sample_df(df_train, num_train_samples)
    copy_wav_files_in_df(sampled_train_df, src_path, dst_path)

    src_path = "C:/Users/Ella bar-El/Documents/Deep_Learning/SpeechFormerProject/MELD.Raw/test/wav/"
    dst_path = "C:/Users/Ella bar-El/Documents/Deep_Learning/SpeechFormerProject/MELD.Raw/test/wav_800/"
    os.mkdir(dst_path)
    df_test = df[df.state == "test"]
    sampled_test_df = sample_df(df_test, num_test_samples)
    copy_wav_files_in_df(sampled_test_df, src_path, dst_path)

    src_path = "C:/Users/Ella bar-El/Documents/Deep_Learning/SpeechFormerProject/MELD.Raw/dev/wav/"
    dst_path = "C:/Users/Ella bar-El/Documents/Deep_Learning/SpeechFormerProject/MELD.Raw/dev/wav_800/"
    os.mkdir(dst_path)
    df_dev = df[df.state == "dev"]
    sampled_dev_df = sample_df(df_dev, num_dev_samples)
    copy_wav_files_in_df(sampled_dev_df, src_path, dst_path)

    output_df = pd.concat([sampled_train_df, sampled_test_df, sampled_dev_df])
    output_df.to_csv("C:/Users/Ella bar-El/Documents/Deep_Learning/SpeechFormerProject/S4SpeechFormer/metadata/metadata_meld_800.csv")


meta_csv_file = "C:/Users/Ella bar-El/Documents/Deep_Learning/SpeechFormerProject/S4SpeechFormer/metadata/metadata_meld copy.csv"
num_train_samples = 800
num_test_samples = 100
num_dev_samples = 50

sample_data_set(meta_csv_file, num_train_samples, num_test_samples, num_dev_samples)