INPUT_DIR='../mp3'
OUTPUT_DIR='../chorus'

source ~/miniconda3/etc/profile.d/conda.sh

# highlighter
cd pop-music-highlighter

conda activate highlighter
python main.py -i $INPUT_DIR -o $OUTPUT_DIR

# Deepchorus
conda deactivate 
conda activate deepchorus
cd ../DeepChorus
python preprocess/extract_spectrogram.py -i $INPUT_DIR 
python predict.py -i $INPUT_DIR

# Energy
cd ../energy
python energy.py -i $INPUT_DIR 