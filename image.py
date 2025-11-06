import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import math

N_FFT = 1728          
HOP_LENGTH = 130      
WINDOW_TYPE = 'blackman'
SR = 16000            
CUT_SAMPLES = 64000 

def create_lps(y, sr=SR):
    
    status = "Fixed"
    audio_length = y.shape[-1]
    
    if audio_length < CUT_SAMPLES:
        y = np.tile(y, int(CUT_SAMPLES / audio_length) + 1)[:CUT_SAMPLES]
        status = "Padded (Samples)"
        
    elif audio_length > CUT_SAMPLES:
        y = y[:CUT_SAMPLES]
        status = "Truncated (Samples)"
  
    stft_result = librosa.stft(
        y, 
        n_fft=N_FFT, 
        hop_length=HOP_LENGTH, 
        window=WINDOW_TYPE
    )

    magnitude_spectrogram = np.abs(stft_result)
    lps_db = librosa.amplitude_to_db(magnitude_spectrogram, ref=np.max)

    return lps_db, status 

def save_lps_image(lps_data, output_filepath):
   
    plt.figure(figsize=(10, 5)) 
    plt.imshow(lps_data, 
               aspect='auto', 
               origin='lower', 
               cmap='viridis',
               vmin=lps_data.min(), 
               vmax=lps_data.max())
    plt.axis('off')
    plt.tight_layout(pad=0) 
    plt.savefig(output_filepath, bbox_inches='tight', pad_inches=0)
    plt.close() 

def process_audio_directory(input_dir, output_dir):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"출력 디렉토리 생성: {output_dir}")
        
    audio_files = glob.glob(os.path.join(input_dir, '*.flac'))
    
    if not audio_files:
        print(f"\n오류: 입력 디렉토리 '{input_dir}'에서 .flac 파일을 찾을 수 없습니다.")
        return

    print(f"\n--- 총 {len(audio_files)}개의 음성 파일 처리 시작 ---")
    
    for audio_path in audio_files:
        filename_base = os.path.splitext(os.path.basename(audio_path))[0]
        output_filepath = os.path.join(output_dir, f'{filename_base}_lps.png')
        
        try:
            y, sr = librosa.load(audio_path, sr=SR)
            lps_data, status = create_lps(y, sr)
            save_lps_image(lps_data, output_filepath)
            print(f"성공: {filename_base}.flac -> {status} -> {output_filepath}")

        except Exception as e:
            print(f"오류: {filename_base}.flac 처리 중 문제 발생: {e}")

if __name__ == '__main__':
    # ----------------------------------------------------
    # TODO
    # ----------------------------------------------------
    INPUT_AUDIO_DIR = '/home/cnrl/Workspace/ND/Dataset/asvspoof_5/flac_T'   # .flac
    OUTPUT_IMAGE_DIR = '/home/cnrl/Workspace/ND/Dataset/asvspoof_5/image/lps_T'       # .png
    # ------------------------------------------------------------------------

    process_audio_directory(INPUT_AUDIO_DIR, OUTPUT_IMAGE_DIR)
    print("\n모든 처리가 완료되었습니다.")