# SETUP
[Download sample datasets from Google Drive](https://drive.google.com/drive/folders/1QoT9-IfC4W5KzNpU-ONkLtyKxBDeNQnd?usp=sharing)

```   
git clone git@github.com:JudeWells/RocheHackathon2024.git
cd RocheHackathon2024
```
Move the downloaded datasets to the data folder in the project directory.
Directory structure should look something like this:
```
└ RochHackathon2024
    ├── README.md
    ├──src        
        ├── data_loader.py
        ├── model.py
        ├── requirements.txt
        ├── supervised_results.csv
        ├── train.py
        └── utils
            └── add_modulo_fold.py
    ├──data
        ├── HUMAN.tar.gz
        └── MOUSE.tar.gz 

# Create virtual environment and install dependencies
```
```
python3 -m venv venv_prot_gym
source venv_prot_gym/bin/activate
pip install -r requirements.txt
# check that the software is running
python src/train.py
```
If all working correctly you should see:
```
Training model on eval_files/MOUSE
Epoch 1/50, Loss: 1.3608, MAE: 0.9498, R2: -1.0875, Spearman R: 0.1025
Epoch 2/50, Loss: 0.5581, MAE: 0.6077, R2: 0.1484, Spearman R: 0.4036
.
.
.
Epoch 50/50, Loss: 5.2112
Test performance metrics:
mse: 5.2851
mae: 2.1600
r2: 0.0177
spearman_r: 0.3390
Metrics for 3 experiments saved to outputs/supervised_results.csv
        mse       mae        r2  spearman_r           DMS_id
0  0.230559  0.389087  0.642256    0.878918           MOUSE
2  1.415424  1.044317  0.291035    0.745408           HUMAN
```
