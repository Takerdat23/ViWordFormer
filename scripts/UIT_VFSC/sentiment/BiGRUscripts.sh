#!/bin/bash

python main_s1.py --config-file configs/UIT_VFSC/sentiment/BiGRU/config_bpe_BiGRU_UIT_VFSC_sentiment.yaml --schema 1
python main_s1.py --config-file configs/UIT_VFSC/sentiment/BiGRU/config_unigram_BiGRU_UIT_VFSC_sentiment.yaml --schema 1
python main_s1.py --config-file configs/UIT_VFSC/sentiment/BiGRU/config_wordpiece_BiGRU_UIT_VFSC_sentiment.yaml --schema 1
python main_s1.py --config-file configs/UIT_VFSC/sentiment/BiGRU/config_vipher_BiGRU_UIT_VFSC_sentiment.yaml --schema 1
python main_s1.py --config-file configs/UIT_VFSC/sentiment/BiGRU/config_bpe_BiGRU_UIT_VFSC_sentiment.yaml --schema 2
python main_s1.py --config-file configs/UIT_VFSC/sentiment/BiGRU/config_unigram_BiGRU_UIT_VFSC_sentiment.yaml --schema 2
python main_s1.py --config-file configs/UIT_VFSC/sentiment/BiGRU/config_wordpiece_BiGRU_UIT_VFSC_sentiment.yaml --schema 2
