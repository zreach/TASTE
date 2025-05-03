import torch, librosa
from muq import MuQ

device = 'cuda'
muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
muq = muq.to(device).eval()

import os
import torch
import librosa
import numpy as np
from tqdm import tqdm
import pickle

def downsample_avg(arr, factor):
    # 假设T在第二维度
    original_length = arr.shape[1]
    new_length = original_length // factor
    arr_cliped = arr[:, :factor * new_length, :]
    # factor = original_length // new_length
    # assert original_length % new_length == 0, "原长度必须能被目标长度整除"
    new_shape = list(arr_cliped.shape)
    new_shape[1] = new_length
    new_shape.insert(2, factor)
    reshaped = arr_cliped.reshape(new_shape)
    return np.mean(reshaped, axis=2)

# Extract features and save as id:embedding in pkl format
def traverse_and_extract_features(folder_path, output_path, threshold_sec):
    # features_dict = {}
    names_ready = [os.path.splitext(file)[0] for file in os.listdir(output_path)]
    
    for root, _, files in os.walk(folder_path):
        for file in tqdm(files):
            if file.endswith('.mp3') or file.endswith('.wav'):
                file_prefix = os.path.splitext(file)[0]
                if file_prefix not in names_ready:
                    file_path = os.path.join(root, file)
                    wav, sr = librosa.load(file_path, sr=24000)
                    duration = librosa.get_duration(y=wav, sr=sr)
    
                    if duration > threshold_sec:
                        # 计算中间段的起始和结束时间
                        start_time = (duration - threshold_sec) / 2
                        end_time = start_time + threshold_sec
                        
                        # 转换为样本索引
                        start_idx = int(start_time * sr)
                        end_idx = int(end_time * sr)
                        
                        # 截取中间段
                        wav = wav[start_idx:end_idx]
                    wavs = torch.tensor(wav).unsqueeze(0).to(device)
                    with torch.no_grad():
                        audio_embeds = muq(wavs, output_hidden_states=False)
                    
                    last_embeds = audio_embeds.last_hidden_state
                    last_embeds = last_embeds.cpu().numpy()
                    last_embeds = downsample_avg(last_embeds, 10)
                    # if file_prefix in mapping_dict:
                    #     features_dict[mapping_dict[file_prefix]] = audio_embeds
                    # features_dict[file_prefix] = last_embeds
                    output_file = os.path.join(output_path, file_prefix + '.npy')
                    np.save(output_file, last_embeds)
    
    # Save features_dict to pkl
    # pkl_file_path = os.path.join(output_path, 'muq-last.pkl')
    # with open(pkl_file_path, 'wb') as f:
    #     pickle.dump(features_dict, f)

folder_path = '/data2/zhouyz/rec/MSD_old/'
output_path = '/data2/zhouyz/rec/MSD_old/muq-last-npy-10'
os.makedirs(output_path, exist_ok=True)

threshold_sec = 360.0

traverse_and_extract_features(folder_path, output_path, threshold_sec)
