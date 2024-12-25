import torch
import os
import numpy as np
from tqdm import tqdm

from models import *
from database import save_distance

#--Definitionen------------------------------------------#
#--------------------------------------------------------#
device = torch.device('mps')

#Runs (ID aus Datenbank) angeben. Bei nur einem, dann von x bis x, sonst von x bis y
von     = int(input("von:" ))
bis     = int(input("bis: "))
bis     += 1
ds      = int(input("DS [0:MNIST][1:WBC]"))

if ds == 0:
    model_a = MNIST_Model().to(device)
    model_b = MNIST_Model().to(device)
else:
    model_a = BCW_Model().to(device)
    model_b = BCW_Model().to(device)
#--Funktionen--------------------------------------------#
#--------------------------------------------------------#

# Schlüssel in der state_dict zu bereinigen - hat mit opacus zu tun
def clean_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_module."):
            new_state_dict[k[len("_module."):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

# Anzahl von Dateien in einem Ordner
def count_files_in_directory(directory_path):
    return len([name for name in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, name))])

# Modelle in NumPy-Arrays umwandeln
def model_to_numpy(model):
    return np.concatenate([param.detach().cpu().numpy().flatten() for param in model.parameters()])

def l2_distance(i,j,model1, model2):
    # Berechnung
    l2_distance = np.linalg.norm(model1 - model2)

    #Speicherung in DB
    save_distance(run,i,j,l2_distance)

    #Ausgabe
    #print(f"L2-Distanz zwischen {i} und {j}: {l2_distance:.6f}")

# Anzahl der Modelle
for run in range(von,bis):
    models_directory_path = f"/Volumes/bachelor/models/{run}"

    num_models = count_files_in_directory(models_directory_path)
    print(f'Es gibt {num_models} Dateien für run {run}.')

    total_iterations = sum(1 for i in range(num_models) for j in range(i + 1, num_models))

    with tqdm(total=total_iterations, desc="") as pbar:
        for i in range(0,num_models):
            for j in range(i+1,num_models):
                # Modelle laden und die state_dict-Schlüssel bereinigen
                state_dict_a = torch.load(f"{models_directory_path}/{run}_run{i+1}.pth", map_location=device)
                state_dict_b = torch.load(f"{models_directory_path}/{run}_run{j+1}.pth", map_location=device)

                model_a.load_state_dict(clean_state_dict(state_dict_a))
                model_b.load_state_dict(clean_state_dict(state_dict_b))

                model_a_np = model_to_numpy(model_a)
                model_b_np = model_to_numpy(model_b)

                # L2-Distanz berechnen
                l2_distance(i,j,model_a_np,model_b_np)

                # Progressbar updaten
                pbar.update(1)