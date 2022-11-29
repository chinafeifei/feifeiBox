#!/usr/bin/env sh

set -e

TMP_DIR=/tmp

# Install Intel® oneAPI Base Toolkit
echo "Install oneAPI"
cd $TMP_DIR
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
rm GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
echo "deb https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update
echo y | sudo apt install intel-basekit=2022.2.0-262

# Install Intel® Distribution of OpenVINO™ Toolkit
echo "install OpenVINO"
wget -c https://apt.repos.intel.com/openvino/2021/GPG-PUB-KEY-INTEL-OPENVINO-2021
sudo apt-key add GPG-PUB-KEY-INTEL-OPENVINO-2021
rm GPG-PUB-KEY-INTEL-OPENVINO-2021
echo "deb https://apt.repos.intel.com/openvino/2021 all main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2021.list
sudo apt update
echo y | sudo apt install intel-openvino-runtime-ubuntu20-2021.4.752

# If install GPU Driver
#sudo apt-get install -y gpg-agent wget
#wget -qO - https://repositories.intel.com/graphics/intel-graphics.key |  sudo apt-key add -
#sudo apt-add-repository  'deb [arch=amd64] https://repositories.intel.com/graphics/ubuntu focal main'
#sudo apt-get update
#echo y | sudo apt-get install  intel-opencl-icd  intel-level-zero-gpu level-zero  intel-media-va-driver-non-free libmfx1
#echo y | sudo apt-get install  libigc-dev  intel-igc-cm  libigdfcl-dev  libigfxcmrt-dev  level-zero-dev
#stat -c "%G" /dev/dri/render*
#groups ${USER}
#echo y | sudo apt install vainfo

