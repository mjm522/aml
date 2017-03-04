#!/bin/bash

AML_DIR=$(pwd)

AML_DATA='REPLACE THIS LINE WITH THE PATH TO AML_DATA'

AML_DATA='/home/baxter_gps/catkin_workspaces/baxter_ws/src/aml_data' #gitignore

MODULES='aml_robot aml_dl aml_io aml_lfd aml_ctrl aml_perception aml_data_collection_utils'

for module in $MODULES
do
	module_path=$AML_DIR/$module/src
	echo "adding module: $module_path"
	export PYTHONPATH=$module_path:$PYTHONPATH
done

export PYTHONPATH=$AML_DIR/aml_calib/scripts:$PYTHONPATH

export AML_DATA

