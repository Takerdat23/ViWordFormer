#!/bin/bash

python main_s1.py --config-file configs/UIT_VFSC/topic/BiGRU/s1/config_bpe_BiGRU_UIT_VFSC_topic.yaml
python main_s1.py --config-file configs/UIT_VFSC/topic/BiGRU/s1/config_unigram_BiGRU_UIT_VFSC_topic.yaml
python main_s1.py --config-file configs/UIT_VFSC/topic/BiGRU/s1/config_wordpiece_BiGRU_UIT_VFSC_topic.yaml
python main_s1.py --config-file configs/UIT_VFSC/topic/BiGRU/s1/config_vipher_BiGRU_UIT_VFSC_topic.yaml
python main_s1.py --config-file configs/UIT_VFSC/topic/BiGRU/s2/config_bpe_BiGRU_UIT_VFSC_topic.yaml
python main_s1.py --config-file configs/UIT_VFSC/topic/BiGRU/s2/config_unigram_BiGRU_UIT_VFSC_topic.yaml
python main_s1.py --config-file configs/UIT_VFSC/topic/BiGRU/s2/config_wordpiece_BiGRU_UIT_VFSC_topic.yaml
