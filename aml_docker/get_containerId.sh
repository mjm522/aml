#!/bin/bash

export containerId=$(docker ps -l -q)
echo $containerId