#!/bin/bash

AML_DIR=$(pwd)

AML_DATA='REPLACE THIS LINE WITH THE PATH TO AML_DATA'

AML_DATA='/home/ermanoarruda/Projects/aml_data' #gitignore

echo "Default AML_DATA set to ${AML_DATA}, change? (y/n)"
read answer
if echo "$answer" | grep -iq "^y" ;then
     echo -n "Enter path to AML_DATA > "
     read AML_DATA;
else
    echo No
fi



MODULES='aml_robot aml_dl aml_io aml_lfd aml_ctrl aml_perception aml_data_collec_utils aml_playground'

for module in $MODULES
do
	module_path=$AML_DIR/$module/src
	echo "adding module: $module_path"
	export PYTHONPATH=$module_path:$PYTHONPATH
done

export PYTHONPATH=$AML_DIR/aml_calib/scripts:$PYTHONPATH

export AML_DATA

#this is for tensor board
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:'/usr/local/cuda/extras/CUPTI/lib64'

