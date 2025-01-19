#!/bin/bash

python main_s1.py --config-file configs/UIT_VFSC/sentiment/GRU/s1/config_bpe_GRU_UIT_VFSC_sentiment.yaml
python main_s1.py --config-file configs/UIT_VFSC/sentiment/GRU/s1/config_unigram_GRU_UIT_VFSC_sentiment.yaml
python main_s1.py --config-file configs/UIT_VFSC/sentiment/GRU/s1/config_wordpiece_GRU_UIT_VFSC_sentiment.yaml
python main_s1.py --config-file configs/UIT_VFSC/sentiment/GRU/s1/config_vipher_GRU_UIT_VFSC_sentiment.yaml
python main_s1.py --config-file configs/UIT_VFSC/sentiment/GRU/s2/config_bpe_GRU_UIT_VFSC_sentiment.yaml
python main_s1.py --config-file configs/UIT_VFSC/sentiment/GRU/s2/config_unigram_GRU_UIT_VFSC_sentiment.yaml
python main_s1.py --config-file configs/UIT_VFSC/sentiment/GRU/s2/config_wordpiece_GRU_UIT_VFSC_sentiment.yaml
