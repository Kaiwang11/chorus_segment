import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm
import os
import argparse
import pandas as pd
def parse_arguments():
    parser = argparse.ArgumentParser(description='Process audio energy')
    parser.add_argument('-i', '--input', default='../mp3', help='Input MP3 file path')
    parser.add_argument('-o', '--output', default='../chorus', help='Input MP3 file path')

    return parser.parse_args()

def create_output_path(input_path,output_path):
    # Get base filename without extension
    filename = os.path.splitext(os.path.basename(input_path))[0]
    # Create output directory with filename
    output_dir = os.path.join(output_path, filename)
    os.makedirs(output_dir, exist_ok=True)
    # Create output path with suffix
    output_path = os.path.join(output_dir, f"{filename}_energy.mp3")
    return output_path

def extract_energy_segment(input_file, output_file, segment_duration=30.0):
    # Load the audio file with progress bar
    print("Loading audio file...")
    with tqdm(total=100, desc="Loading") as pbar:
        y, sr = librosa.load(input_file, sr=None)
        pbar.update(100)

    # Calculate the RMS energy with progress bar
    print("Processing audio...")
    hop_length = 512
    frame_length = 2048
    
    # Calculate total frames for progress bar
    n_frames = 1 + (len(y) - frame_length) // hop_length
    rms = np.zeros(n_frames)
    
    with tqdm(total=n_frames, desc="Analyzing") as pbar:
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        pbar.update(n_frames)

    # Find the frame with the maximum RMS energy
    max_rms_index = np.argmax(rms)
    # Convert frame index to time
    max_rms_time = librosa.frames_to_time(max_rms_index, sr=sr, hop_length=hop_length)
    # Calculate the start and end times of the segment
    start_time = max(0, max_rms_time - segment_duration / 2)
    end_time = min(len(y) / sr, start_time + segment_duration)
    # Extract the segment
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    segment = y[start_sample:end_sample]

    # Save the segment to a new file
    print("Saving output file...")
    with tqdm(total=100, desc="Saving") as pbar:
        sf.write(output_file, segment, sr)
        pbar.update(100)
    song_name = os.path.splitext(os.path.basename(input_file))[0]
    return song_name,start_time, end_time
    

    

def main():
    args = parse_arguments()
    
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    data=[]
    if os.path.isdir(args.input):
        for f in os.listdir(args.input):
            if f.endswith('.mp3'):
                filename=f.split('.')[0]
                output_path = create_output_path(filename,args.output)
                filepath=os.path.join(args.input, f)
                song_name,start_time, end_time=extract_energy_segment(filepath, output_path)
                data.append([song_name,'energy',start_time,end_time])
                print(f"Processed audio saved to: {output_path}")
    else:
        output_path = create_output_path(args.input,args.output)
        extract_energy_segment(args.input, output_path)
        print(f"Processed audio saved to: {output_path}")

    csv_path='../chorus/chorus_time.csv'
    
    new_data = pd.DataFrame(data, columns=["name","type" ,"start_time", "end_time"])
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Check if song already exists
        df = pd.concat([df, new_data], ignore_index=True)
        df.to_csv(csv_path, index=False)

    else:
        new_data.to_csv(csv_path, mode='w', header=True, index=False)

if __name__ == "__main__":
    main()
