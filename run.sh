cp -r /home2/tensorflow-broad-product/oob_tf_models/gpuoob/wide_deep .

export DATASET_DIR=$PWD/wide_deep/datasets
export OUTPUT_DIR=$PWD/wide_deep/output
export CHECKPOINT_DIR=$PWD/wide_deep/ckpt

python models/recommendation/tensorflow/wide_deep_large_ds/training/train.py --batch_size=512 --data_location=${DATASET_DIR} --checkpoint=${CHECKPOINT_DIR} --output_dir=${OUTPUT_DIR}
