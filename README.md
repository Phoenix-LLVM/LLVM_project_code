# LLVM_project_code

# Multi Task Reinforcement Learning Architecture Proposals
- To look at our proposed MTRL architectures using CLIP and Qwen-VL, you can navigate to Multi_Task_Reinforcement_Learning_Architecture_Proposals folder and look at the notebooks listed for different models


# Training
### Prepare the dataset:
- Follow the dataset_preparation/Training Data Preparation_100k.ipynb to create dataset for training.

### Training the model
- Clone the QwenVL repo ```git clone https://github.com/QwenLM/Qwen-VL.git```
- Clone the DeepSpeed repo(for distributed training) ```git clone https://github.com/microsoft/DeepSpeed.git```
- ```cd DeepSpeed```
- ```DS_BUILD_FUSED_ADAM=1 pip3 install .```
- ```cd ../Qwen-VL```
- ```pip3 install -r requirements.txt```
- ```cp ../Finetune_QwenVL_model/carla_finetune.sh finetune/```
- run carla_finetune.sh to finetune the Qwen-VL model using LoRA ```sh finetune/carla_finetune.sh```

## Setup

### Prerequisites:
- cuda toolkit 12.1
- docker
- nvidia-container-toolkit for docker

### Download and setup CARLA 0.9.10.1:
```
mkdir carla
cd carla
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.10.1.tar.gz
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/AdditionalMaps_0.9.10.1.tar.gz
tar -xf CARLA_0.9.10.1.tar.gz
tar -xf AdditionalMaps_0.9.10.1.tar.gz
rm CARLA_0.9.10.1.tar.gz
rm AdditionalMaps_0.9.10.1.tar.gz
cd ..
```
Also update the ~/.bashrc file:
export CARLA_ROOT=PATH_TO_CARLA_0.9.10.1

### Clone this repo and build the environment:

```
git clone https://github.com/s-suryakiran/LLVM_project_code.git
cd LLVM_project_code
```

```
export PYTHONPATH=$PYTHONPATH:PATH_TO_LLVM_project_code
```

### Create Dockerfile:
```Dockerfile
FROM carlasim/carla:0.9.10.1
WORKDIR /home/carla/Import
COPY ./AdditionalMaps_0.9.10.1.tar.gz ./
WORKDIR /home/carla
RUN ./ImportAssets.sh
```

### Build Dockerfile:
- ```docker build -t carla:eval .```

### Install Python3.7:
```
 sudo apt update
 sudo apt install software-properties-common
 sudo add-apt-repository ppa:deadsnakes/ppa
 sudo apt install python3.7
 sudo apt-get install python3.7-distutils
```

### Setup Python Virtual Environments:
```
sudo apt-get install python3-pip
python3.10 -m pip install virtualenv
python3.10 -m virtualenv modelenv
source ./modelenv/bin/activate
pip install -r model_env_requirements.txt

python3.7 -m pip install virtualenv
python3.7 -m virtualenv tcpenv
source ./tcpenv/bin/activate
pip install -r tcp_env_requirements.txt
```


# Benchmarking
### Run the carla docker server:
- ```sudo docker run --privileged --gpus 0 -e SDL_VIDEODRIVER=offscreen -e SDL_HINT_CUDA_DEVICE=0 -p 2000-2002:2000-2002 -v /tmp/.X11-unix:/tmp/.X11-unix:rw -it carla:eval /bin/bash ./CarlaUE4.sh```

### To benchmark the reference TCP model:
- cd PATH_TO_LLVM_project_code
- make sure you started the carla docker server.
- Download their model - ```wget https://storage.googleapis.com/carla_dataset_bucket/Eval_Uploads/best_model.ckpt```
- Modify the ```TEAM_CONFIG``` in /leaderboard/scripts/run_evaluation.sh
- run the run_evaluation.sh - ```sh leaderboard/scripts/run_evaluation.sh```


## To benchmark our model:
- Update ```CARLA_ROOT``` with the root folder of CARLA_0.9.10.1 in sh leaderboard/scripts/run_evaluation_carla.sh
- Enable modelenv and start jupyter notebook server.
- Download all the files in this google bucket using
- gsutil -m cp \
  "gs://carla_dataset_bucket/output_carla_25K_automated/README.md" \
  "gs://carla_dataset_bucket/output_carla_25K_automated/adapter_config.json" \
  "gs://carla_dataset_bucket/output_carla_25K_automated/adapter_model.bin" \
  "gs://carla_dataset_bucket/output_carla_25K_automated/qwen.tiktoken" \
  "gs://carla_dataset_bucket/output_carla_25K_automated/special_tokens_map.json" \
  "gs://carla_dataset_bucket/output_carla_25K_automated/tokenizer_config.json" \
  "gs://carla_dataset_bucket/output_carla_25K_automated/trainer_state.json" \
  "gs://carla_dataset_bucket/output_carla_25K_automated/training_args.bin" \
  .
- Open Control_Prediction.ipynb and follow the steps there and load the weights downloaded.
- After you run all the cells in the notebook, run
```sh leaderboard/scripts/run_evaluation_carla.sh```


# References:
- https://arxiv.org/abs/1711.03938
- https://arxiv.org/abs/2206.08129
- https://arxiv.org/abs/2308.12966
