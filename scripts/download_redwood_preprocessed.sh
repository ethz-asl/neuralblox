#!/bin/bash
mkdir -p data
cd data
echo "Start downloading ..."
wget -O redwood_2017_preprocessed.zip https://polybox.ethz.ch/index.php/s/86iC89KTKAETJW2/download
unzip redwood_2017_preprocessed.zip
echo "Done!"
