#!/bin/sh

URL=https://emdhgcalae.nrp-nautilus.io/ttbar/data/HGCal22Data_signal_driven_ttbar_v11/nElinks_5/5Elinks_data.csv
wget $URL -P "$HAWQ_JET_TAGGING/data/econ" 
mv "$HAWQ_JET_TAGGING/data/econ/5Elinks_data.csv" "$HAWQ_JET_TAGGING/data/econ/HGCal22Data_signal_driven_ttbar_v11.csv" 

URL=https://emdhgcalae.nrp-nautilus.io/EleGun/low_pt_high_eta/data/nElinks_5/5Elinks_data.csv
wget $URL -P "$HAWQ_JET_TAGGING/data/econ" 
mv "$HAWQ_JET_TAGGING/data/econ/5Elinks_data.csv" "$HAWQ_JET_TAGGING/data/econ/low_pt_high_eta.csv" 
