#!/bin/bash
 
# read -p "Enter the stage to process (1: resample | 2: w2v | 3: f0 | 4: make filelist): " stage

# # Stage 1: resample audio to 16khz, 16bit, mono channel
# if [ "$stage" -eq 1 ]; then
#     echo "resampling..."
#     python3 resample.py -i "$input_wav_dir" 
# fi

# # Stage 2: extract w2v feature from MMS
# if [ "$stage" -eq 2 ]; then
#     echo "Extracting w2v features..."
#     python3 extract_w2v.py -i "$input_wav_dir" 
# fi

# # Stage 3: extract F0 using YAAPT
# if [ "$stage" -eq 3 ]; then
#     echo "Extracting F0..."
#     python3 extract_f0.py -i "$input_wav_dir"
# fi

# if [ "$stage" -eq 4 ]; then
#     echo "Making filelist..."
#     python3 prepare_filelist.py -i "$input_wav_dir" -o "$output_dir"
# fi

# echo "compledted $stage."


#!/bin/bash


read -p "Enter the absolute path to the data directory:" input_wav_dir
read -p "Enter the absolute path to the data feature data output directory:" output_dir

# Stage 1: resample audio to 16khz, 16bit, mono channel

read -p "Enter the origin sample rate:" org_sr
read -p "Enter the target sample rate:" target_sr
echo "resampling..."
python3 resample.py -i "$input_wav_dir" --org_sr "$org_sr" --target_sr "$target_sr"
echo "compledted resampling."

# Stage 2: extract w2v feature from MMS

echo "Extracting w2v features..."
python3 extract_w2v.py -i "$input_wav_dir" 
echo "compledted extracting w2v."

# Stage 3: extract F0 using YAAPT

echo "Extracting F0..."
python3 extract_f0.py -i "$input_wav_dir"
echo "compledted extracting f0."

# Stage 4: making filelist

echo "Making filelist..."
python3 prepare_filelist.py -i "$input_wav_dir" -o "$output_dir"
echo "Data preparation compledted !"
