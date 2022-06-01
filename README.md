# fed_iot_guard
Detection of IoT devices infected by malware from their network communications, using federated machine learning

This code allows to run experiments simulating different configurations of clients trying to train deep learning models for malware detection in their IoT device. A big part of the code is dedicated to running federated learning experiments so that the clients can collaboratively train their models without having to share any data. Since this is a simluation, everything runs locally on your machine, using the public dataset N-BaIoT (https://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT). The publication corresponding to this code is available at https://www.sciencedirect.com/science/article/pii/S1389128621005582 in open access.

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
* From the `N-BaIoT/` folder, if you're on linux, open a terminal and run the following command: `wget -r -np -nH --cut-dirs=3 -R "index.html*" https://archive.ics.uci.edu/ml/machine-learning-databases/00442/`

  It will download the dataset. This command will take a few minutes to execute, depending on your internet connexion.
  If you're on another operating system than Linux, or if you cannot use wget for another reason, you need to find an alternative tool to recursively download files from https://archive.ics.uci.edu/ml/machine-learning-databases/00442/, or you need to manually download all of them.

* Now the data is downloaded, but some files inside of the inner folders are still in archive format (.rar extension). To fix that, I used the `unar` command, which can easily be installed on linux using `sudo apt install unar`. To run the command recursively, run `find ./ -name '*.rar' -execdir unar {} \;` from the `N-BaIoT/` folder.
  Again, if you can't use unar for some reason, you can also manually unrar each .rar file in the dataset.

### Step 3: install the required python libraries.

* Using PyCharm: create a new python interpreter for the project ("File > Settings > Project: fed_iot_guard > Python Interpreter > Gear icon > Add > Virtualenv Environment or whatever you prefer > Select a location in which you'll create your new environment and select Python 3.9 as your base interpreter). When this is done, PyCharm will see that the requirements listed in requirements.txt are not satistfied, and it will ask to install said requirements (which you should do). Once you have done that, you can run any configuration from PyCharm (for example `fedavg_autoencoders_test`).

* Using virtualenv: Create a new virtual environment based on Python 3.9. Activate this environment. Install the requirements by moving to the `fed_iot_guard/` directory and running `pip install -r requirements.txt`. You can then run the main program from the terminal. For instance, you can run (from the `fed_iot_guard/`directory): `python src/main.py decentralized autoencoder --test --fedavg --collaborative --verbose-depth=1`.

## Usage

### Program arguments
To run an experiment, you have to run main.py with the appropriate arguments. Since there are a lot of different configurations to experiment with, and different approaches to all of them, there are quite a lot of parameters to allow us to do that rigorously.
* The first parameter determines whether you consider the clients' data decentralized (each client keeps its own data) or centralized (the data is put in common). The values accepted are `decentralized` and `centralized`.
* The second parameter determines whether you want to do supervised learning (assuming each client has access to some labeled data of both benign and malicious classes) using neural network classifiers, or if you want to do unsupervised learning (assuming each client just has access to some benign data) using autoencoders. The values accepted for this parameter are thus `classifier` and `autoencoder`.
* As the third parameter, you either indicate `--gs` if you want to run a grid search to select the best set of hyper-parameters (using training data to train the model and validation data to evaluate it), or `--test` if you want to train the model using train and validation data and evaluate it on the test set (to obtain the final test results after you have already selected which hyper-parameters to use).
* In case you have selected `--decentralized`, you can either let clients collaborate, using `--collaborative`, or not, using `--no-collaborative`. When running the grid search (`--gs`), collaboration allows the clients to select the hyper-parameters that give the best results on average among them (rather than all clients having its own set of best hyper-parameters. When running the final test (`--test`), collaboration allows the use of federated learning rather than having each client train its own model separately.
* In case you have selected `--test` and `--collaborative`, you can still decide which kind of federated algorithm you want to use. What is referred to as `Mini-batch aggregation` in the publication is called `fedsgd` in the code, and what is referred to as `Multi-epoch aggregation` in the publication is called `fedavg` in the code. The reason for that is that the naming used in the code is misleading (my versions of fedsgd and fedavg are not exactly equivalent to what is defined in "Communication-efficient learning of deep networks from decentralized data") so I decided to change their name in the publication, but I didn't update the code accordingly. Sorry for that. Long story short, use `--fedsgd` for `Mini-batch aggregation` or `--fedavg` for `Multi-epoch aggregation`.
* The last two parameters are about the prints in the console. `--verbose` activates printing in the console of the state of the training process (recommended since the experiments can be very long to run). `--no-verbose` deactivates all prints. Additionally, you can define the maximum depths of inner loops in which printing is enabled by using `--verbose-depth n` with `n` being an integer. A value of 1, 2 or 3 is recommended.

Examples (commands to run from a terminal in `fed_iot_guard/`, with the appropriate python environment activated):
* Collaborative grid search among decentralized clients for classifiers: `python src/main.py decentralized classifier --gs --collaborative --verbose-depth=2`
* Federated training and testing among decentralized clients for autoencoders, using `Multi-epoch aggregation`: `python src/main.py decentralized autoencoder --test --fedavg --collaborative --verbose-depth=2`

Note that if you're using PyCharm, you will directly have access to all the configurations that I used for my experiments. These configurations are saved in the `.idea/runConfigurations/` folder, in .xml files. If you do not use PyCharm, you can inspect these files and look at the `PARAMETERS` value to get all of the parameters of the configurations that I used.

### Hyper-Parameters
Many different hyper-parameters are defined in the code of the main function, inside of dictionaries. Some of them are fixed (inside of `constant_params`) and some of them vary during the grid search (inside of `varying_params`). Once you have run the grid search and obtained the results, you can use the jupyter notebook `GS Results.ipynb` to check which hyper-parameters are the best for each configuration. You can then copy the hyper-parameters obtained for each of the configurations inside of the appropriate `configuration_params` list (there is one such list for autoencoders and one for classifiers).

## Modification
To make my code easily extendable, here are a few explanations about its inner workings, file by file.
* `main.py` handles the program arguments, defines the hyper-parameters, creates the appropriate configuration of clients, and calls the appropriate experiment functions.
* `data.py` does everything related to data: reading, re-sampling and splitting for example.
* `supervised_data.py` contains the manipulations necessary to obtain the supervised datasets.
* `unsupervised_data.py` contains the manipulations necessary to obtain the unsupervised datasets.
* `architectures.py` defines the PyTorch architectures used (neural network classifier and autoencoder).
* `ml.py` defines a few machine learning function that are common between the supervised and the unsupervised approaches (normalization stuff).
* `supervised_ml.py` contains the functions to train the PyTorch classifiers.
* `unsupervised_ml.py` contains the functions to train the PyTorch autoencoders.
* `supervised_experiments.py` contains all of the supervised setup experiments, local or federated. Note that local training handles both the centralized and the decentralized (without collaboration) cases.
* `unsupervised_experiments.py` contains all of the unsupervised setup experiments, local or federated. Note that local training handles both the centralized and the decentralized (without collaboration) cases.
* `federated_util.py` contains many useful functions for federated learning: aggregation functions, adversarial attacks.
* `grid_search.py` contains all the code specific to grid searches.
* `test_hparams.py` contains the code to perform the final training and evaluation of the model once a given set of hyper-parameters has been selected.
* `metrics.py` defines a class to quickly compute binary classification metrics (like accuracy of F1-score) from the numbers of true positives, false positives, etc...
* `print_util.py` defines all print functions in the program, such that they appear in well-organized columns. Note that we used context-printer to keep the prints organized (https://pypi.org/project/context-printer/).
* `saving.py` contains all the code that saves the results.
 


