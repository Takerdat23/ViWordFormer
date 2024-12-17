#!/bin/bash

python main_s1.py --config-file configs/UIT_VFSC/topic/GRU/config_bpe_GRU_UIT_VFSC_topic.yaml --schema 1
python main_s1.py --config-file configs/UIT_VFSC/topic/GRU/config_unigram_GRU_UIT_VFSC_topic.yaml --schema 1
python main_s1.py --config-file configs/UIT_VFSC/topic/GRU/config_wordpiece_GRU_UIT_VFSC_topic.yaml --schema 1
python main_s1.py --config-file configs/UIT_VFSC/topic/GRU/config_vipher_GRU_UIT_VFSC_topic.yaml --schema 1
python main_s1.py --config-file configs/UIT_VFSC/topic/GRU/config_bpe_GRU_UIT_VFSC_topic.yaml --schema 2
python main_s1.py --config-file configs/UIT_VFSC/topic/GRU/config_unigram_GRU_UIT_VFSC_topic.yaml --schema 2
python main_s1.py --config-file configs/UIT_VFSC/topic/GRU/config_wordpiece_GRU_UIT_VFSC_topic.yaml --schema 2
