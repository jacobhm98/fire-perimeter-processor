#!/bin/sh

cd data/raw
for zip_file in *.zip; do
    dir_name="${zip_file%.zip}"
    mkdir -p "$dir_name"
    unzip -o "$zip_file" -d "$dir_name"
done
