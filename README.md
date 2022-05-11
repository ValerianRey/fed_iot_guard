# fed_iot_guard
Detection of IoT devices infected by malwares from their network communications, using federated machine learning

Current paper available at https://www.sciencedirect.com/science/article/pii/S1389128621005582

Dataset comes from https://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT

## Setup
### Step 0: recommendations: 
* Using Linux (I use Ubuntu 20.04) so that you can more easily download and extract the dataset
* Using Python 3.9 for compatibility
* Using PyCharm (there is a free version called PyCharm Community which should be good enough), so that you instantly get the run configurations (containing the program arguments) that I used for my experiments. Otherwise you can understand how the program arguments work, and you can even run it from a terminal directly.

### Step 1: get the repository
Clone or download the repository to your machine. 

### Step 2: get the data
The .gitignore file contains the `data/` folder, so you have to manually create this folder, download and extract the dataset into it. 
* Create a `data/` folder in `fed_iot_guard/`
* Create a `N-BaIoT/` folder inside of `data/`
* From here, if you're on linux, open a terminal and run the following command: `wget -r -np -nH --cut-dirs=3 -R "index.html*" https://archive.ics.uci.edu/ml/machine-learning-databases/00442/`

  It will download the dataset. This command will take a few minutes to execute, depending on your internet connexion.
  If you're on another operating system than Linux, or if you cannot use wget for another reason, you need to find an alternative tool to recursively download files from https://archive.ics.uci.edu/ml/machine-learning-databases/00442/, or you need to manually download all of them.

* Now the data is downloaded, but some files inside of the inner folders are still in archive format (.rar extension). To fix that, I used the `unar` command, which can easily be installed on linux using `sudo apt install unar`. To run the command recursively, run `find ./ -name '*.rar' -execdir unar {} \;`
  Again, if you can't use unar for some reason, you can also manually unrar each .rar file in the dataset.

### Step 3: install the required python libraries.

* Using PyCharm: create a new python interpreter for the project ("File > Settings > Project: fed_iot_guard > Python Interpreter > Gear icon > Add > Virtualenv Environment or whatever you prefer > Select a location in which you'll create your new environment and select Python 3.9 as your base interpreter). When this is done, PyCharm will see that the requirements listed in requirements.txt are not satistfied, and it will ask to install said requirements (which you should do). Once you have done that, you can run any configuration from PyCharm (for example `fedavg_autoencoders_test`).

* Using virtualenv: Create a new virtual environment based on Python 3.9. Activate this environment. Install the requirements by moving to the `fed_iot_guard/` directory and running `pip install -r requirements.txt`. You can then run the main program from the terminal. For instance, you can run (from the `fed_iot_guard/ directory`): `python src/main.py decentralized autoencoder --test --fedavg --collaborative --verbose-depth=1`.

## Usage
Coming soon

## Modification
Coming soon


