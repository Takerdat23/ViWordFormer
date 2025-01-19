#!/bin/bash

python main_s1.py --config-file configs/UIT_VFSC/sentiment/LSTM/s1/config_bpe_LSTM_UIT_VFSC_sentiment.yaml
python main_s1.py --config-file configs/UIT_VFSC/sentiment/LSTM/s1/config_unigram_LSTM_UIT_VFSC_sentiment.yaml
python main_s1.py --config-file configs/UIT_VFSC/sentiment/LSTM/s1/config_wordpiece_LSTM_UIT_VFSC_sentiment.yaml
python main_s1.py --config-file configs/UIT_VFSC/sentiment/LSTM/s1/config_vipher_LSTM_UIT_VFSC_sentiment.yaml
python main_s1.py --config-file configs/UIT_VFSC/sentiment/LSTM/s2/config_bpe_LSTM_UIT_VFSC_sentiment.yaml
python main_s1.py --config-file configs/UIT_VFSC/sentiment/LSTM/s2/config_unigram_LSTM_UIT_VFSC_sentiment.yaml
python main_s1.py --config-file configs/UIT_VFSC/sentiment/LSTM/s2/config_wordpiece_LSTM_UIT_VFSC_sentiment.yaml
