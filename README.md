# PlanNet
A framework for network game alterations using reinforcement learning.

## Prerequisites
Currently supported on macOS and Linux (tested on CentOS 7.4.1708, but should work out of the box on any standard Linux distro), as well as on Windows via [WSL](https://docs.microsoft.com/en-us/windows/wsl/install-win10).
Makes heavy use of Docker, see e.g. [here](https://docs.docker.com/engine/install/) for how to install. Tested with Docker 19.03. The use of Docker largely does away with dependency and setup headaches, making it significantly easier to reproduce the reported results.

## Configuration
The Docker setup uses Unix groups to control permissions. You can reuse an existing group that you are a member of, or create a new group `groupadd -g GID GNAME` and add your user to it `usermod -a -G GNAME MYUSERNAME`. 

Create a file `relnet.env` at the root of the project (see `relnet_example.env`) and adjust the paths within: this is where some data generated by the container will be stored. Also specify the group ID and name created / selected above.

Add the following lines to your `.bashrc`, replacing `/home/john/git/relnet` with the path where the repository is cloned. 

```bash
export RN_SOURCE_DIR='/home/john/git/relnet'
set -a
. $RN_SOURCE_DIR/relnet.env
set +a

export PATH=$PATH:$RN_SOURCE_DIR/scripts
```

Make the scripts executable (e.g. `chmod u+x scripts/*`) the first time after cloning the repository, and run `apply_permissions.sh` in order to create and permission the necessary directories.

## Managing the container
Some scripts are provided for convenience. To build the container (note, this will take a significant amount of time e.g. 2 hours, as some packages are built from source):
```bash
update_container.sh
```
To start it:
```bash
manage_container.sh up
```
To stop it:
```bash
manage_container.sh stop
```
To restart the container:
```bash
restart.sh
```

## Compiling objective function extension
(First-time setup only) with the container running (via `manage_container.sh up` above), execute the following command: 
```bash
docker exec -it relnet-manager /bin/bash -c "cd /relnet/relnet/objective_functions && make"
```

## Setting up synthetic graph data

Synthetic data will be automatically generated when the experiments are ran and stored to `$RN_EXPERIMENT_DIR/stored_graphs`.

## Accessing the services
There are several services running on the `manager` node.
- Jupyter notebook server: `http://localhost:8888` (make sure to select the `python-relnet` kernel which has the appropriate dependencies)
- Tensorboard (currently disabled due to its large memory footprint): `http://localhost:6006`

The first time Jupyter is accessed it will prompt for a token to enable password configuration, it can be grabbed by running `docker exec -it relnet-manager /bin/bash -c "jupyter notebook list"`.

## Running experiments
The file `relnet/experiment_launchers/run_rnet_dqn.py` contains the configuration to run the RNet-DQN algorithm on synthetic graphs. You may modify objective functions, hyperparameters etc. to suit your needs.
Example for how to run:
```bash
docker exec -it relnet-manager /bin/bash -c "source activate ucfadar-relnet && python relnet/experiment_launchers/run_rnet_dqn.py"
```
## Problems with jupyter kernel
In case the `python-relnet` kernel is not found, try reinstalling the kernel by running `docker exec -it relnet-manager /bin/bash -c "source activate ucfadar-relnet; python -m ipykernel install --user --name relnet --display-name python-relnet"`

<!-- ## Contact
If you face any issues or have any queries feel free to contact `v.darvariu@ucl.ac.uk` and I will be happy to assist. -->
