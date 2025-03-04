# Chorus Segment

## Enviroment Install
DeepChorus and Highlighter need different version of tensorflow.Therefore we need to create two enviroments.Here we use miniconda with .yml of each enviroment :
```bash
conda env create -f deepchorus.yml
conda env create -f highlighter.yml
```

## Usage
1. Put the songs in ./mp3 directory
2. run all methods
```bash
./segment.sh
```


## Results
The segmented songs are in ./chorus and timestamp saved in ./chorus/chorus_time.csv

## To Be Solved
Data in ./chorus/chorus_time.csv might be duplicated if ./chorus/chorus_time.csv is not be removed before excution.

