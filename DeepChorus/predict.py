import os
import argparse
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import joblib
import importlib
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from constant import *
from loader import load_labels, load_features
from evaluator import *
import soundfile as sf
import librosa
from pydub import AudioSegment
import io
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Chorus Detection')
    parser.add_argument('-n', '--network',default='DeepChorus' ,help='Network module name')
    parser.add_argument('-m', '--mark',default='Deepchorus_2021' ,help='Model mark/identifier')
    parser.add_argument('-i', '--input', default='../mp3', help='Input audio file or directory')
    parser.add_argument('-o', '--output', default='../chorus', help='Output directory')
    parser.add_argument('--model_path', default='./model/Deepchorus_2021.h5', 
                       help='Path to model weights')
    return parser.parse_args()


def load_model(network_name, model_path):
    network_module = importlib.import_module('network.' + network_name)
    create_model = network_module.create_model
    
    model = create_model(input_shape=SHAPE, chunk_size=CHUNK_SIZE)
    model.compile(
        loss='binary_crossentropy', 
        optimizer=Adam(lr=LR),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(threshold=0.5), 
            tf.keras.metrics.Recall()
        ]
    )
    model.load_weights(model_path)
    return model


# Create chorus directory if it doesn't exist
def inference(model, feature):
    feature_length = len(feature)
    remain = feature_length % 9
    remain_np = np.zeros([128, 9 - remain, 1])
    feature_crop = np.concatenate((feature, remain_np), axis=1)    
    feature_crop = np.expand_dims(feature_crop, axis=0)
    predictions = model(feature_crop)
    return predictions
def get_result(model, features_dict):

    @tf.function(experimental_relax_shapes=True)
    def predict(t):
        return model(t)
    result={}

    for key in features_dict.keys():
        feature = features_dict[key]
        feature_length = len(feature[1])
        remain = feature_length % 9
        remain_np = np.zeros([128, 9 - remain, 1])
        feature_crop = np.concatenate((feature, remain_np), axis=1)

     
        feature_crop = np.expand_dims(feature_crop, axis=0)

        result_np = predict(feature_crop)[0]
        result[key]=np.argmax(result_np)
    return result

def main():
    SR = 22050
    N_FFT = 2048
    N_HOP = 512
    N_MEL = 128

    args = parse_arguments()

    model = load_model(args.network, args.model_path)
    features = load_features(test_feature_files)
  
    print('Testing...')
    # predictions_dict, target_dict = get_result_dict(model, features, labels)
    max_pos_all=get_result(model, features)
    data=[]
    # print(max_pos_all)
    for key, feature in features.items():
        audio,sr=librosa.load(f'../mp3/{key}.mp3', sr=SR)
        print(audio.shape)
        max_pos=max_pos_all[key]
        extracted_segment = audio[max_pos*SR:(max_pos+30)*SR]
        output_dir=f'{args.output}/{key}'
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{key}_deepchorus.mp3")
        sf.write(output_file, extracted_segment, SR, format='MP3')
        print(f"Saved {output_file}")

        data.append([key,'deepchorus' ,max_pos, max_pos+30])
        csv_path='../chorus/chorus_time.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        new_data = pd.DataFrame(data, columns=["name", "type", "start_time", "end_time"])
        df = pd.concat([df, new_data], ignore_index=True)
    else:
        df = pd.DataFrame(data, columns=["name", "type", "start_time", "end_time"])
    # Save to CSV
    df.to_csv(csv_path, index=False)


  
if __name__ == "__main__":
    main()