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

# Extract features and save as id:embedding in pkl format
def traverse_and_extract_features(folder_path, output_path):
    features_dict = {}
    names_ready = [os.path.splitext(file)[0] for file in os.listdir(output_path)]
    
    for root, _, files in os.walk(folder_path):
        for file in tqdm(files):
            if file.endswith('.mp3') or file.endswith('.wav'):
                file_prefix = os.path.splitext(file)[0]
                if file_prefix not in names_ready:
                    file_path = os.path.join(root, file)
                    wav, sr = librosa.load(file_path, sr=24000)
                    wavs = torch.tensor(wav).unsqueeze(0).to(device)
                    with torch.no_grad():
                        audio_embeds = muq(wavs, output_hidden_states=False)
                    
                    last_embeds = audio_embeds.last_hidden_state
                    last_embeds = last_embeds.cpu().numpy()
                    # if file_prefix in mapping_dict:
                    #     features_dict[mapping_dict[file_prefix]] = audio_embeds
                    features_dict[file_prefix] = last_embeds
                    output_file = os.path.join(output_path, file_prefix + '.npy')
                    np.save(output_file, audio_embeds)
    
    # Save features_dict to pkl
    pkl_file_path = os.path.join(output_path, 'muq-last.pkl')
    with open(pkl_file_path, 'wb') as f:
        pickle.dump(features_dict, f)

folder_path = '/data2/zhouyz/rec/music4all/music4all/audios'
output_path = '/data2/zhouyz/rec/music4all/music4all/muq-last'
os.makedirs(output_path, exist_ok=True)

traverse_and_extract_features(folder_path, output_path)
