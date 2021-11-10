#!/usr/bin/bash
# License: MIT. See license file in root directory
# Copyright(c) qiuminghilllittlewan (2021/11/8)

if [ ! $# -eq 2 ];then
    echo  param error
    exit 1
fi
DIRNAME=$1
BUILD_CLASS=$2
CURRENT_DIR=$PWD


echo "the build folder is $DIRNAME"

if [ ! -d $DIRNAME ];then
    mkdir $DIRNAME
else
    echo dir exist
    exit 1
fi

cd third_party/cnpy
cmake -DCMAKE_INSTALL_PREFIX=../../libs/cnpy -DENABLE_STATIC=OFF . && make -j8 && make install
cd ../../

if [ $BUILD_CLASS -eq 0 ];then
    python $CURRENT_DIR/tools/gen_wts.py --config-file='./configs/DukeMTMC/bagtricks_R50-ibn.yml' \
    --wts_path='./bot_R50_ibn.wts'  \
    MODEL.WEIGHTS './checkpoints/bhjx/model_best.pth' \
    MODEL.DEVICE "cuda:0"
    cd $DIRNAME
    echo "People Build"
    cmake -DBUILD_FASTRT_ENGINE=ON -DBUILD_DEMO=ON -DUSE_CNUMPY=ON -DBUILD_FP16=ON -DBUILD_PYTHON_INTERFACE=ON -DBUILD_CLASS=ON ..
elif [ $BUILD_CLASS -eq 1 ];then
    python $CURRENT_DIR/tools/gen_wts.py --config-file='./configs/VERIWild/bagtricks_R50-ibn.yml' \
    --wts_path='./veri_wild_bot_R50_ibn.wts'  \
    MODEL.WEIGHTS './checkpoints/vbhjx/model_best.pth' \
    MODEL.DEVICE "cuda:0"
    cd $DIRNAME
    echo "Car Build"
    cmake -DBUILD_FASTRT_ENGINE=ON -DBUILD_DEMO=ON -DUSE_CNUMPY=ON -DBUILD_FP16=ON -DBUILD_PYTHON_INTERFACE=ON -DBUILD_CLASS=OFF ..
else
    echo param should be 0 or 1
fi
make -j8 && make install
./demo/fastrt -s
exit 0
