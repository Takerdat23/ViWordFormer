#!/bin/bash

python main_s1.py --config-file configs/UIT_VFSC/topic/BiLSTM/s1/config_bpe_BiLSTM_UIT_VFSC_topic.yaml
python main_s1.py --config-file configs/UIT_VFSC/topic/BiLSTM/s1/config_unigram_BiLSTM_UIT_VFSC_topic.yaml
python main_s1.py --config-file configs/UIT_VFSC/topic/BiLSTM/s1/config_wordpiece_BiLSTM_UIT_VFSC_topic.yaml
python main_s1.py --config-file configs/UIT_VFSC/topic/BiLSTM/s1/config_vipher_BiLSTM_UIT_VFSC_topic.yaml
python main_s1.py --config-file configs/UIT_VFSC/topic/BiLSTM/s2/config_bpe_BiLSTM_UIT_VFSC_topic.yaml
python main_s1.py --config-file configs/UIT_VFSC/topic/BiLSTM/s2/config_unigram_BiLSTM_UIT_VFSC_topic.yaml
python main_s1.py --config-file configs/UIT_VFSC/topic/BiLSTM/s2/config_wordpiece_BiLSTM_UIT_VFSC_topic.yaml
