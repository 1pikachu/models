<!--- 50. AI Kit -->
## Run the model

Setup your environment using the instructions below, depending on if you are
using [AI Kit](/docs/general/tensorflow/AIKit.md):

<table>
  <tr>
    <th>Setup using AI Kit</th>
    <th>Setup without AI Kit</th>
  </tr>
  <tr>
    <td>
      <p>To run using AI Kit you will need:</p>
      <ul>
        <li>numactl
        <li>unzip
        <li>wget
        <li>Activate the `tensorflow` conda environment
        <pre>conda activate tensorflow</pre>
      </ul>
    </td>
    <td>
      <p>To run without AI Kit you will need:</p>
      <ul>
        <li>Python 3
        <li>intel-tensorflow>=2.5.0
        <li>git
        <li>numactl
        <li>unzip
        <li>wget
        <li>A clone of the Model Zoo repo<br />
        <pre>git clone https://github.com/IntelAI/models.git</pre>
      </ul>
    </td>
  </tr>
</table>

After your setup is done, download and unzip the pretrained model. The
path to this directory should be set as the `CHECKPOINT_DIR` before
running quickstart scripts.
```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/bert_large_checkpoints.zip
unzip bert_large_checkpoints.zip
export CHECKPOINT_DIR=$(pwd)/bert_large_checkpoints
```

Next, set environment variables with paths to the [dataset](#datasets),
checkpoint files, and an output directory, then run a quickstart script.
See the [list of quickstart scripts](#quick-start-scripts) for details
on the different options.

The snippet below shows how to run a quickstart script:
```
# cd to your model zoo directory
cd models

export DATASET_DIR=<path to the dataset being used>
export CHECKPOINT_DIR=<path to the unzipped checkpoints>
export OUTPUT_DIR=<path to the directory where log files will be saved>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

# Run a script for your desired usage
./quickstart/language_modeling/tensorflow/bert_large/inference/cpu/bfloat16/<script name>.sh
```
