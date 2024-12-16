#!/bin/bash

python main_s1.py --config-file configs1/UIT-VSFC/sentiment/gru/config_s1_bpe.yaml --schema 1
python main_s1.py --config-file configs1/UIT-VSFC/sentiment/gru/config_s1_unigram.yaml --schema 1
python main_s1.py --config-file configs1/UIT-VSFC/sentiment/gru/config_s1_wordpiece.yaml --schema 1
python main_s1.py --config-file configs1/UIT-VSFC/sentiment/gru/config_s2_bpe.yaml --schema 2
python main_s1.py --config-file configs1/UIT-VSFC/sentiment/gru/config_s2_unigram.yaml --schema 2
python main_s1.py --config-file configs1/UIT-VSFC/sentiment/gru/config_s2_wordpiece.yaml --schema 2
