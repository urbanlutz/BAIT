# BAIT "Learning to Learn"

## Installation
Dev environment is best set up via VS Code remote server (SSH & DevContainer Extensions)

### Building the VM (Docker Host)
1. Get access to https://apu.cloudlab.zhaw.ch
2. Create a Key Pair via Project/Compute/Key Pairs "Create Key Pair"
    - Name: My-SSH-KeyPair
    - Key Type: SSH
3. Create SSH Security Group via Project/Network/Security Group "Create Security Group"
    - Name: SSH
    -> Create Security Group
    Add Rule via "Add Rule":
        - Description: SSH
        - Directions: Ingress
        - Open Port: Port
        - Port: 22
        - Remote: CIDR
        - CIDR: 0.0.0.0/0
4. Launch Instance Project/Compute/Instances "Launch Instance"
    - Instance Name: any
    - Source: 2022-01_Ubuntu_Focal_nVidia-Cuda_Docker
    - Flavour: g1.xlarge.t4
    - Network: Allocate "internal"
    - Security Group: SSH
    - Key Pair: My-SSH-KeyPair
    -> Launch Instance
5. Add FloatingIp Project/Compute/Instances
    - On Instance: Action dropdown (far right) -> Allocate Floating IP
    - choose any
6. Reboot machine to avoid "Driver/ Library version mismatch" Error

### Dev Environment (VS Code DevContainer)
VS Code Remote - SSH & Remote - Containers Extensions necessary
Connect to Docker Host
1. Add private key (*.pem file) to VS Code
    - copy .pem file to ~/.ssh
2. Add SSH Host
    - ctrl-p -> Remote-SSH: Add new SSH Host
    - Add Floating IP from Docker Host
3. Connect
4. Pull git repo onto Docker host
5. configure git user name
```
git config --global user.email "lutzurb1@students.zhaw.ch"
git config --global user.name "Urban Lutz"
git config --global http.sslVerify false
```

Launch DevContainer
1. ctrl+p -> Remote Containers: Reopen in Container

# Experiment Specification

Experiments can be specified in CSV format in experiments.csv.
Valid columns are all parammeter available in any class from params.py.
If a parameter for a given class is omitted, the default value from params.py will be set.

The mechanism allows to specify an arbitrary number of pruning phases by prefixing the pruning parameter with "X_" (X -> phase number, starting at 1).
3 kinds of pruning phases are supported:
- "train": only training, no pruning, (set the train_epochs param)
- "one-shot": only pruning (set strategy, sparsity)
- "iterative": prune and train (set strategy, sparsity, prune_epochs, train_epochs)


# Dataset
lives on gdrive
https://medium.com/geekculture/how-to-upload-file-to-google-drive-from-linux-command-line-69668fbe4937

    