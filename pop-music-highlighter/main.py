from model import MusicHighlighter
from lib import *
import tensorflow as tf
import numpy as np
import os
import argparse
import pandas as pd
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = ''

def extract(fs,output_dir ,length=30 ,save_score=False, save_thumbnail=False, save_wav=True):
    with tf.Session() as sess:
        model = MusicHighlighter()
        sess.run(tf.global_variables_initializer())
        model.saver.restore(sess, 'model/model')
        data=[]
        for f,o in tqdm(zip(fs,output_dir)):
            name = os.path.split(f)[-1][:-4]
            audio, spectrogram, duration = audio_read(f)
            n_chunk, remainder = np.divmod(duration, 3)
            chunk_spec = chunk(spectrogram, n_chunk)
            pos = positional_encoding(batch_size=1, n_pos=n_chunk, d_pos=model.dim_feature*4)
            
            n_chunk = n_chunk.astype('int')
            chunk_spec = chunk_spec.astype('float')
            pos = pos.astype('float')
            
            attn_score = model.calculate(sess=sess, x=chunk_spec, pos_enc=pos, num_chunk=n_chunk)
            attn_score = np.repeat(attn_score, 3)
            attn_score = np.append(attn_score, np.zeros(remainder))

            # score
            attn_score = attn_score / attn_score.max()
            if save_score:
                np.save('{}_score.npy'.format(name), attn_score)

            # thumbnail
            attn_score = attn_score.cumsum()
            attn_score = np.append(attn_score[length], attn_score[length:] - attn_score[:-length])
            index = np.argmax(attn_score)
            highlight = [index, index+length]
            if save_thumbnail:
                np.save('{}_highlighter.npy'.format(name), highlight)

            if save_wav:
                librosa.output.write_wav('{}_pop.mp3'.format(o), audio[highlight[0]*22050:highlight[1]*22050], 22050)
            data.append([name,'highlighter' ,index, index+length])
        csv_path='../chorus/chorus_time.csv'
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            new_data = pd.DataFrame(data, columns=["name", "type", "start_time", "end_time"])
            df = pd.concat([df, new_data], ignore_index=True)
        else:
            df = pd.DataFrame(data, columns=["name", "type", "start_time", "end_time"])
        # Save to CSV
        df.to_csv(csv_path, index=False)

        # if os.path.exists(csv_path):
        #     df = pd.read_csv(csv_path)
        #     # Check if song already exists
        #     mask = (df['name'] == name) & (df['type'] == 'highlighter')
        #     if not mask.any():  # Only append if it doesn't exist
        #         df = pd.concat([df, new_data], ignore_index=True)
        #         df.to_csv(csv_path, mode='w', header=True, index=False)  # Overwrite file once
        #     df.to_csv(csv_path, mode='a', header=False, index=False)
        # else:
        #     new_data.to_csv(csv_path, mode='w', header=True, index=False)
if __name__ == '__main__':
    #fs = ['YOUR MP3 FILE NAME 1', 'YOUR MP3 FILE NAME 2'] # list
    parser=argparse.ArgumentParser()
    parser.add_argument('-i','--input', help='Input audio file or directory')
    parser.add_argument('-o','--output',default='../chorus' ,help='Input audio file or directory')

    args=parser.parse_args()
    fs=[]
    output_dir=[]
    fs = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith('.mp3')]
    # Generate output directories and file paths
    for f in fs:
        filename = os.path.splitext(os.path.basename(f))[0]  # Extract filename without extension
        output_dir_name = os.path.join(args.output, filename)
        os.makedirs(output_dir_name, exist_ok=True)
        output_dir.append(os.path.join(output_dir_name, filename+'_highlighter.mp3'))
    extract(fs, output_dir ,length=30, save_score=False, save_thumbnail=False, save_wav=True)
    