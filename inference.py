import torch
import numpy as np
import pandas as pd
import argparse
from glob import glob
import librosa
from model import get_model
from dataloader import get_dataloader
from metrics import LWLRAP, label_ranking_average_precision_score
from tqdm.auto import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--csv-file', help='The name of the csv file that contains the audio files to predict on')
parser.add_argument('--audio-path', help='The path to where the audio files are to predict on')
parser.add_argument('--model-path', default= "models/", help='The path to the pretrained model weights')
parser.add_argument('--output-csv', default= "output.csv", help='The path to output the results of the inference')
parser.add_argument('-class-txt', default='class_birds.txt', help='File to the classes file')

args = parser.parse_args()

#we load the csv file 
#which should only contain wav files
df = pd.read_csv(args.csv_file)

#load the classes text
classes = np.loadtxt(args.class_txt, dtype='str', delimiter='\n')
classes = list(classes)

df['recording_id'] = [f[:-4] for f in df.File]

# we check that there is no audio file with a duration less than 14.9 seconds
ids = []
for fn in df.recording_id:
    y,sr = librosa.load(f'{args.audio_path}/{fn}.wav', sr=None)
    if librosa.get_duration(y,sr) <14.9:
        ids.append(fn)

df = df[~df.recording_id.isin(ids)].reset_index(drop=True)

species_cols = [f'{classes[i]}' for i in range(len(classes))]

cv_preds = pd.DataFrame(columns=species_cols)
cv_preds['recording_id'] = df['recording_id'].drop_duplicates()
cv_preds.loc[:, species_cols] = 0
cv_preds = cv_preds.reset_index(drop=True)

#config file for adjusting batch loading, and spectrogram settings
class Config:
    batch_size = 8 
    num_workers = 0
    sliding_window = 2.98
    num_classes = len(classes)
    sr = 32_000
    duration = 2.98
    total_duration = 14.9
    nmels = 128 # changed from 128
    data_root = args.audio_path


# if there is a GPU we load the audio onto the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#loading model 
paths = glob(f'{args.model_path}/*.pth')
for path in paths:
    print(f'loading model {path}')
    model = get_model(path, len(classes))
    model.to(device)
    model.eval()
    #load the data into dataloaders
    dataloader = get_dataloader(df, config=Config(), mode='test')

    tk = tqdm(dataloader, total=len(dataloader))
    sub_index = 0
    with torch.no_grad():
        #we go through all of the data
        for i, im, in enumerate(tk):
            #pass it to the cpu or gpu
            im = im.to(device)
            #predict on the data
            for i, x_partial in enumerate(torch.split(im, 1, dim=1)):
                x_partial = x_partial.squeeze(1)
                if i == 0:
                    preds = model(x_partial)
                else:
                    # take max over predictions
                    preds = torch.max(preds, model(x_partial))
            #get the confidence score of each species and add it to the csv file
            o = preds.sigmoid().cpu().numpy()
            for val in o:
                cv_preds.loc[sub_index, species_cols] += val

                sub_index += 1
#divide by length of folds
cv_preds.loc[:, species_cols] /=len(paths)

print(f'Saving predictions to {args.output_csv}')
cv_preds.to_csv(args.output_csv, index=False)





    


