# import libraries

import torch
from torch.utils.data import Dataset, DataLoader
import yaml  # used by RawNet2 to read the configuration
import json  # used to read config file
import aasist  # official AASIST implementation from https://github.com/clovaai/aasist/blob/main/models/AASIST.py
import os
import IPython.display as ipd  # used to display audio
import aasist
from tqdm import tqdm  # progress bar
from model import  DownStreamLinearClassifier, RawNetEncoderBaseline, RawNetBaseline, SSDNet1D, SAMOArgs  # SSDNet is the Res-TSSDNet Model
from datautils import genSpoof_list, Dataset_ASVspoof2019_train  # ASVspoof dataset utils
# Used to get the evaluation metrics
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
from evaluate_tDCF_asvspoof19 import compute_eer
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# Manipulation classes used
from datautils import VolumeChange, AddWhiteNoise, AddEnvironmentalNoise, WaveTimeStretch, AddEchoes, TimeShift, PitchShift, AddFade, ResampleAugmentation, pad_or_clip_batch
import torchaudio.transforms



torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Evaluation Configurations
batch_size = 32
gpu = 0  # GPU id to use
torch.cuda.set_device(gpu)

# Load file
with open("/home/hwang-gyuhan/Workspace/ND/config.conf", "r") as f_json:
    config = json.loads(f_json.read())
    
def load_model(model_name:str, config:dict):
    if model_name == "CLAD":
        with open(config['aasist_config_path'], "r") as f_json:        
            aasist_config = json.loads(f_json.read())
        aasist_model_config = aasist_config["model_config"]
        aasist_encoder = aasist.AasistEncoder(aasist_model_config).to(device)
        downstream_model = DownStreamLinearClassifier(aasist_encoder, input_depth=160)
        checkpoint = torch.load(config['clad_model_path_for_evaluation'], map_location=device)
        downstream_model.load_state_dict(checkpoint["state_dict"])
        downstream_model = downstream_model.to(device)
        return downstream_model
    
# Calculate the metrics, if the threshold is not given, the threshold output by the EER calculation will be used.
def get_eval_metrics(score_save_path, plot_figure=True, given_threshold=None, print_result=True):
    cm_data = np.genfromtxt(score_save_path, dtype=str)
    cm_keys = cm_data[:, 2]
    # cm_keys = 'bonafide' means 1, 'spoof' means 0
    cm_keys = np.where(cm_keys == 'bonafide', 1, 0)
    cm_scores = cm_data[:, 3].astype(float)
    # Compute EER
    # Extract bona fide (real human) and spoof scores from the CM scores
    bona_cm = cm_scores[cm_keys == 1]
    spoof_cm = cm_scores[cm_keys == 0]
    eer_cm, threshold = compute_eer(bona_cm, spoof_cm)

    auc = roc_auc_score(cm_keys, cm_scores)
    if given_threshold is not None:
        threshold = given_threshold
    y_pred = np.where(cm_scores > threshold, 1, 0)
    f1 = f1_score(cm_keys, y_pred)
    acc = balanced_accuracy_score(cm_keys, y_pred)
    # compute False Acceptance Rate and False Rejection Rate
    FAR = np.sum(cm_keys[y_pred == 1] == 0) / np.sum(cm_keys == 0)
    FRR = np.sum(cm_keys[y_pred == 0] == 1) / np.sum(cm_keys == 1)
    if print_result == True:
        print(f"EER:{eer_cm}, auc:{auc}, F1 score:{f1}, acc:{acc}, threshold:{threshold}, FAR:{FAR}, FRR:{FRR}")
    if plot_figure == True:
        # ylgnbu_pal = sns.color_palette("YlGnBu", as_cmap=True)
        sns.histplot(bona_cm, kde=False, label='Real', stat="density", element="step", fill=False, bins='auto')
        sns.histplot(spoof_cm, kde=False, label='Deepfake', stat="density",element="step", fill=False, bins='auto')
        
        plt.legend()
        plt.xlabel('Prediction score')
        plt.title('Prediction score histogram')
    return (eer_cm, auc, f1, acc, threshold, FAR, FRR)

def evaluation_19_LA_eval(model, score_save_path, model_name, database_path, augmentations=None, augmentations_on_cpu=None, batch_size = 1024, manipulation_on_real=True, cut_length = 64600):
    # In asvspoof dataset, label = 1 means bonafide
    model.eval()
    device = "cuda"
    # load asvspoof 2019 LA eval dataset
    d_label_trn, file_eval, utt2spk = genSpoof_list(dir_meta=database_path+"ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt", is_train=False, is_eval=False)
    print('no. of ASVspoof 2019 LA evaluating trials', len(file_eval))
    asvspoof_LA_eval_dataset = Dataset_ASVspoof2019_train(list_IDs=file_eval, labels=d_label_trn, base_dir=os.path.join(
        database_path+'ASVspoof2019_LA_eval/'), cut_length=cut_length, utt2spk=utt2spk)
    asvspoof_2019_LA_eval_dataloader = DataLoader(asvspoof_LA_eval_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8, pin_memory=True)  # added num_workders param to speed up.
    with open(score_save_path, 'w') as file:  # This creates an empty file or empties an existing file
        pass
    
    if model_name == "SAMO":
        samo_args = SAMOArgs()
        samo_args.path_to_database = config['database_path'][:-3]  # SAMO default path does not include "/LA"
        samo_args.batch_size = batch_size
        samo_args.target = False  # we use all the evaluation data, rather than only the target data
        # print(samo_args)
        _, _, eval_data_loader, train_bona_loader, _, eval_enroll_loader, _ = get_loader(samo_args)
        samo = SAMO(samo_args.enc_dim, m_real=samo_args.m_real, m_fake=samo_args.m_fake, alpha=samo_args.alpha).to(device)
        if samo_args.val_sp:
            # define and update eval centers
            eval_enroll = update_embeds(samo_args.device, model, eval_enroll_loader)
        else:  # use training centers without eval enrollment
            if samo_args.one_hot:
                spklist = ['LA_00' + str(spk_id) for spk_id in range(79, 99)]
                tmp_center = torch.eye(samo_args.enc_dim)[:20]
                eval_enroll = dict(zip(spklist, tmp_center))
            else:
                eval_enroll = update_embeds(samo_args.device, model, train_bona_loader)
        samo.center = torch.stack(list(eval_enroll.values()))

    with torch.no_grad():
        for batch_idx, (audio_input, spks, labels) in enumerate(tqdm(asvspoof_2019_LA_eval_dataloader)):
            score_list = []  
            # audio_input = torch.squeeze(audio_input)
            audio_input = audio_input.squeeze(1)
            if augmentations_on_cpu != None:
                audio_input = augmentations_on_cpu(audio_input)
            
            audio_input = audio_input.to(device)

            if augmentations != None:
                if manipulation_on_real == False:
                    # note that some manipulation will change the length of the audio, so we need to clip or pad it to the same length
                    audio_length = audio_input.shape[-1]
                    # only apply the augmentation on the spoofed audio, and pad or clip it to the same length
                    audio_input[labels==0] = pad_or_clip_batch(augmentations(audio_input[labels==0]), audio_length, random_clip=False)

                else:
                    audio_input = augmentations(audio_input)  
            # check the length of the audio, if it is not the same as the cut_length, then repeat or clip it to the same length
            if audio_input.shape[-1] < cut_length:
                audio_input = audio_input.repeat(1, int(cut_length/audio_input.shape[-1])+1)[:, :cut_length]
            elif audio_input.shape[-1] > cut_length:
                audio_input = audio_input[:, :cut_length]
            
            if model_name == "ResTSSDNetModel":
                audio_input = audio_input.unsqueeze(1)  # pretrained ResTSSDNetModel takes 3D input(batch, channel, waveform), so we need to add a dimension for channels
            batch_out = model(audio_input)
            
            if model_name == "AASIST":
                batch_out = batch_out[1]  # the AASIST model output last_hidden_state and score and we only need the score
            # The ResTSSDNetModel output two scores, but the order seems to be reversed. So we need to reverse it back.
            # The ResTSSDNetModel takes bonafide as 0, reference: https://github.com/ghua-ac/end-to-end-synthetic-speech-detection/blob/main/data.py#L89
            elif model_name == "ResTSSDNetModel":  
                batch_out = batch_out[:, [1,0]]
            elif model_name == "SAMO":
                if samo_args.target:  # loss calculation for target-only speakers
                    _, score = samo(batch_out[0], labels, spks, eval_enroll, samo_args.val_sp)
                else:
                    _, score = samo.inference(batch_out[0], labels, spks, eval_enroll, samo_args.val_sp)
            if model_name == "mesonet_whisper_mfcc_finetuned":
                batch_score = (batch_out[:, 0]).data.cpu().numpy().ravel()
            elif model_name == "SAMO":
                batch_score = score.data.cpu().numpy().ravel()
            else:
                batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
            label_list = ['bonafide' if i==1 else 'spoof' for i in labels]
            score_list.extend(batch_score.tolist())

            with open(score_save_path, 'a+') as fh:
                for label, cm_score in zip(label_list,score_list):
                    fh.write('- - {} {}\n'.format(label, cm_score))
            fh.close()   
        print('Scores saved to {}'.format(score_save_path))
    return  get_eval_metrics(score_save_path=score_save_path, plot_figure=False)# TODO: return EER, AUC and other things when I implement

evaluation_results = {}  # create a empty dict to store the results
noise_dataset_path = config['noise_dataset_path']
manipulations = {
    "no_augmentation": None,
    "volume_change_50": torchaudio.transforms.Vol(gain=0.5,gain_type='amplitude'),
    "volume_change_10": torchaudio.transforms.Vol(gain=0.1,gain_type='amplitude'),
    "white_noise_15": AddWhiteNoise(max_snr_db = 15, min_snr_db=15),
    "white_noise_20": AddWhiteNoise(max_snr_db = 20, min_snr_db=20),
    "white_noise_25": AddWhiteNoise(max_snr_db = 25, min_snr_db=25),
    "env_noise_wind": AddEnvironmentalNoise(max_snr_db=20, min_snr_db=20, device="cuda", noise_category="wind", noise_dataset_path=noise_dataset_path),
    "env_noise_footsteps": AddEnvironmentalNoise(max_snr_db=20, min_snr_db=20, device="cuda", noise_category="footsteps", noise_dataset_path=noise_dataset_path),
    "env_noise_breathing": AddEnvironmentalNoise(max_snr_db=20, min_snr_db=20, device="cuda", noise_category="breathing", noise_dataset_path=noise_dataset_path),
    "env_noise_coughing": AddEnvironmentalNoise(max_snr_db=20, min_snr_db=20, device="cuda", noise_category="coughing", noise_dataset_path=noise_dataset_path),
    "env_noise_rain": AddEnvironmentalNoise(max_snr_db=20, min_snr_db=20, device="cuda", noise_category="rain", noise_dataset_path=noise_dataset_path),
    "env_noise_clock_tick": AddEnvironmentalNoise(max_snr_db=20, min_snr_db=20, device="cuda", noise_category="clock_tick", noise_dataset_path=noise_dataset_path),
    "env_noise_sneezing": AddEnvironmentalNoise(max_snr_db=20, min_snr_db=20, device="cuda", noise_category="sneezing", noise_dataset_path=noise_dataset_path),
    "pitchshift_up_110": PitchShift(max_pitch=1.10, min_pitch=1.10, bins_per_octave=12),
    "pitchshift_up_105": PitchShift(max_pitch=1.05, min_pitch=1.05, bins_per_octave=12),
    "pitchshift_down_095": PitchShift(max_pitch=0.95, min_pitch=0.95, bins_per_octave=12),
    "pitchshift_down_090": PitchShift(max_pitch=0.90, min_pitch=0.90, bins_per_octave=12),
    "timestretch_110": WaveTimeStretch(max_ratio=1.10, min_ratio=1.10, n_fft=128),
    "timestretch_105": WaveTimeStretch(max_ratio=1.05, min_ratio=1.05, n_fft=128),
    "timestretch_095": WaveTimeStretch(max_ratio=0.95, min_ratio=0.95, n_fft=128),
    "timestretch_090": WaveTimeStretch(max_ratio=0.90, min_ratio=0.90, n_fft=128),
    "echoes_1000_02": AddEchoes(max_delay=1000, max_strengh=0.2, min_delay=1000, min_strength=0.2),
    "echoes_1000_05": AddEchoes(max_delay=1000, max_strengh=0.5, min_delay=1000, min_strength=0.5),
    "echoes_2000_05": AddEchoes(max_delay=2000, max_strengh=0.5, min_delay=2000, min_strength=0.5),
    "time_shift_1600": TimeShift(max_shift=1600, min_shift=1600),
    "time_shift_16000": TimeShift(max_shift=16000, min_shift=16000),
    "time_shift_32000": TimeShift(max_shift=32000, min_shift=32000),
    "fade_50_linear": AddFade(max_fade_size=0.5,fade_shape='linear', fix_fade_size=True),
    "fade_30_linear": AddFade(max_fade_size=0.3,fade_shape='linear', fix_fade_size=True),
    "fade_10_linear": AddFade(max_fade_size=0.1,fade_shape='linear', fix_fade_size=True),
    "fade_50_exponential": AddFade(max_fade_size=0.5,fade_shape='exponential', fix_fade_size=True),
    "fade_50_quarter_sine": AddFade(max_fade_size=0.5,fade_shape='quarter_sine', fix_fade_size=True),
    "fade_50_half_sine": AddFade(max_fade_size=0.5,fade_shape='half_sine', fix_fade_size=True),
    "fade_50_logarithmic": AddFade(max_fade_size=0.5,fade_shape='logarithmic', fix_fade_size=True),
    "resample_15000": ResampleAugmentation([15000], device="cuda"),
    "resample_15500": ResampleAugmentation([15500], device="cuda"),
    "resample_16500": ResampleAugmentation([16500], device="cuda"),
    "resample_17000": ResampleAugmentation([17000], device="cuda"),
}

for model_name in ["CLAD"]:
    if model_name == "CLAD":
        cut_length = 64600
    model = load_model(model_name,config)
    for (manipulation_name, manipulation) in manipulations.items():
        filename_prefix = model_name
        evaluation_results[manipulation_name] = evaluation_19_LA_eval(model=model, model_name=model_name, database_path=config['database_path'], batch_size = batch_size, augmentations=manipulation, score_save_path=f"scores/{filename_prefix}_{manipulation_name}_eval_19_LA_score.txt", cut_length=cut_length)
        print(f"--------{manipulation_name} finished.--------")
# Show Result of fix threshold(selected by EER)
# The results of white noise injection may have a small difference as there are randomness in the white noise generation.
results = {}
for model_name in ["CLAD"]:
    results[model_name] = {}
    # get EER threshold
    print("-----Results for model:", model_name, "-----")
    results[model_name]["no_augmentation"] = get_eval_metrics(f"scores/{model_name}_no_augmentation_eval_19_LA_score.txt", plot_figure=False, print_result=False)
    for manipulation_name in manipulations:
        results[model_name][manipulation_name] = get_eval_metrics(f"scores/{model_name}_{manipulation_name}_eval_19_LA_score.txt", plot_figure=False, given_threshold=results[model_name]["no_augmentation"][4], print_result=False)
        print(f"{manipulation_name}: F1 score:{results[model_name][manipulation_name][2]:.2%}, FAR:{results[model_name][manipulation_name][5]:.2%}")