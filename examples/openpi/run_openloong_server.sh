#!/bin/bash

cd ../../../OrcaGym_Openpi/3rd_party/openpi
python scripts/serve_policy.py policy:checkpoint --policy.config=pi0_orca_azureloong_lora --policy.dir=checkpoints/pi0_orca_azureloong_lora/azureloong_experiment/999