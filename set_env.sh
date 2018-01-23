#!/bin/sh

AML_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

AML_DATA=${AML_DIR%%/}/aml_data

if [ ! -d "$AML_DATA" ]
then
    echo "AML_DATA folder doesn't exist. Creating now"
    mkdir -p "$AML_DATA"
    echo "AML_DATA folder created : $AML_DATA"
else
    echo "AML_DATA folder exists"
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

