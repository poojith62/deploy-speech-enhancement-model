


import os
import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json
import soundfile as sf
from helper import audio_files_to_numpy, numpy_audio_to_matrix_spectrogram, scaled_in, inv_scaled_ou, matrix_spectrogram_to_numpy_audio

# Define global variables (if they're not already defined in helper.py)
sample_rate = 8000
frame_length = 8064
hop_length_frame = 8064
min_duration = 1.0
n_fft = 255
hop_length_fft = 63

   
import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json
import soundfile as sf

def prediction(weights_path, audio_dir_prediction, dir_save_prediction, audio_input_prediction,
audio_output_prediction):
   

    
    json_file = open('weightsmodel_unet.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    loaded_model.load_weights('weights/model_unet_best.keras')
    print("Loaded model from disk")

    
    audio = audio_files_to_numpy(audio_dir_prediction, audio_input_prediction, sample_rate,
                                 frame_length, hop_length_frame, min_duration)

    
    dim_square_spec = int(n_fft / 2) + 1
    print(dim_square_spec)

    
    m_amp_db_audio,  m_pha_audio = numpy_audio_to_matrix_spectrogram(
        audio, dim_square_spec, n_fft, hop_length_fft)

    
    X_in = scaled_in(m_amp_db_audio)
   
    X_in = X_in.reshape(X_in.shape[0],X_in.shape[1],X_in.shape[2],1)
    
    X_pred = loaded_model.predict(X_in)
    
    inv_sca_X_pred = inv_scaled_ou(X_pred)
    
    X_denoise = m_amp_db_audio - inv_sca_X_pred[:,:,:,0]
    
    print(X_denoise.shape)
    print(m_pha_audio.shape)
    print(frame_length)
    print(hop_length_fft)
    audio_denoise_recons = matrix_spectrogram_to_numpy_audio(X_denoise, m_pha_audio, frame_length, hop_length_fft)
   
    nb_samples = audio_denoise_recons.shape[0]
    
    denoise_long = audio_denoise_recons.reshape(1, nb_samples * frame_length)*10
    
    sf.write(dir_save_prediction + audio_output_prediction, denoise_long[0, :], 8000, 'PCM_24')
    



