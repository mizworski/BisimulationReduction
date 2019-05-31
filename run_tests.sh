#!/usr/bin/env bash

data_dir=data/
output_dir=output/

for file in $(ls data)
do
    echo "Processing" ${file}
    python2 bisim_reduction.py \
        --input_path ${data_dir}/${file} \
        --output_path ${output_dir}/reduced_${file} \
        -k 8 \
        -v 0
done