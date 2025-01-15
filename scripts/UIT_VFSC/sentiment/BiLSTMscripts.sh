#!/bin/bash

python main_s1.py --config-file configs/UIT_VFSC/sentiment/BiLSTM/config_bpe_BiLSTM_UIT_VFSC_sentiment.yaml --schema 1
python main_s1.py --config-file configs/UIT_VFSC/sentiment/BiLSTM/config_unigram_BiLSTM_UIT_VFSC_sentiment.yaml --schema 1
python main_s1.py --config-file configs/UIT_VFSC/sentiment/BiLSTM/config_wordpiece_BiLSTM_UIT_VFSC_sentiment.yaml --schema 1
python main_s1.py --config-file configs/UIT_VFSC/sentiment/BiLSTM/config_vipher_BiLSTM_UIT_VFSC_sentiment.yaml --schema 1
python main_s1.py --config-file configs/UIT_VFSC/sentiment/BiLSTM/config_bpe_BiLSTM_UIT_VFSC_sentiment.yaml --schema 2
python main_s1.py --config-file configs/UIT_VFSC/sentiment/BiLSTM/config_unigram_BiLSTM_UIT_VFSC_sentiment.yaml --schema 2
python main_s1.py --config-file configs/UIT_VFSC/sentiment/BiLSTM/config_wordpiece_BiLSTM_UIT_VFSC_sentiment.yaml --schema 2
