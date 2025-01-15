#!/bin/bash

python main_s1.py --config-file configs/UIT_VFSC/sentiment/LSTM/config_bpe_LSTM_UIT_VFSC_sentiment.yaml --schema 1
python main_s1.py --config-file configs/UIT_VFSC/sentiment/LSTM/config_unigram_LSTM_UIT_VFSC_sentiment.yaml --schema 1
python main_s1.py --config-file configs/UIT_VFSC/sentiment/LSTM/config_wordpiece_LSTM_UIT_VFSC_sentiment.yaml --schema 1
python main_s1.py --config-file configs/UIT_VFSC/sentiment/LSTM/config_vipher_LSTM_UIT_VFSC_sentiment.yaml --schema 1
python main_s1.py --config-file configs/UIT_VFSC/sentiment/LSTM/config_bpe_LSTM_UIT_VFSC_sentiment.yaml --schema 2
python main_s1.py --config-file configs/UIT_VFSC/sentiment/LSTM/config_unigram_LSTM_UIT_VFSC_sentiment.yaml --schema 2
python main_s1.py --config-file configs/UIT_VFSC/sentiment/LSTM/config_wordpiece_LSTM_UIT_VFSC_sentiment.yaml --schema 2
