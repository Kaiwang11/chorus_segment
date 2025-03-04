import os
import warnings
import numpy as np
import joblib
import librosa
import argparse
warnings.filterwarnings("ignore", category=UserWarning)

SR = 22050
N_FFT = 2048
N_HOP = 512
N_MEL = 128

data_path = 'feature_folder'

def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract spectrogram features from audio files')
    parser.add_argument('-i', '--input', required=True, 
                       help='Input audio file or directory')
    parser.add_argument('-o', '--output', default='features.joblib',
                       help='Output joblib file path')
    return parser.parse_args()
def load_labels(label_list):
    labels = {}
    round_labels = {}
    for label_joblib in label_list:
        labels.update(joblib.load(label_joblib))
    for key, label in labels.items():
        round_labels[key] = []
        for row in label:
            round_labels[key] += [[round(float(row[0])), round(float(row[1]))]]
    return round_labels


def load_features(files):
    features = {}
    for feature_file in files:
        features.update(joblib.load(feature_file))
    return features


def extract_features(file):

    y, _ = librosa.load(file, sr=SR)
    data = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=N_HOP,
                                          power=1, n_mels=N_MEL, fmin=20, fmax=5000)
    data = np.reshape(data, (data.shape[0], data.shape[1], 1))
    return data


def folder_to_joblib(base_folder, feature_file):
    feature_set = {}
    if os.path.isfile(base_folder):
        key = '{}'.format(base_folder.replace('.mp3', ''))
        features = extract_features(base_folder)
        feature_set[key] = features
        print(key, np.shape(features))
    else:    
        for (dirpath, _, filenames) in os.walk(base_folder):
            for file in [f for f in filenames if f.endswith('.mp3') or f.endswith('.wav')]:
                key = '{}'.format(file.replace('.mp3', ''))

                features = extract_features(os.path.join(dirpath, file))
                feature_set[key] = features
                print(key, np.shape(features))

    joblib.dump(feature_set, feature_file)
    print(len(feature_set), 'features saved to', feature_file)


if __name__ == '__main__':
    # folder_to_joblib('../mp3/', 'feature_name.joblib')
    args = parse_arguments()
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input path does not exist: {args.input}")
    
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    folder_to_joblib(args.input, args.output)

