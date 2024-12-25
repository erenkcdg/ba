import torch
import torch.optim as optim
import numpy as np
import os
import json
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset
from opacus import PrivacyEngine
from sklearn.datasets import load_breast_cancer
from tqdm import tqdm
from colorama import Fore, Style
from collections import Counter
import pandas as pd
#-own packages-------------------------------------------#
from get_params import *
from models import *
from database import *
from custom_classes import *
from custom_functions import *
from epoch_data import *

#--Allgemeine Config laden
with open("config.json", 'r') as config_file:
    config = json.load(config_file)

#--Einstellungen
log                 = config['settings']['log']
save_to_db          = config['settings']['save_to_db']
ds                  = config['settings']['current_dataset']
experiment          = config['settings']['experiment']
device              = torch.device(config['settings']['device'])

current_dataset, num_epochs, batch_size, learning_rate, general_seed, num_trainings, length_dataset, outliers_indices, current_model, noising, clipping, delta, target_epsilon = get_params(config, ds)

#--Konfiguration des Experiments laden
with open(f"config/{experiment}.json",'r') as exp_config_file:
    exp_config = json.load(exp_config_file)

make_private, data_processing, dataset_mode, init_mode, dp_mode = get_exp_params(exp_config)

final_loss          = 0.0
var_seeds           = []
eps_batching        = 0.0

torch.use_deterministic_algorithms(True)
    
#--Funktionen--------------------------------------------#
#--------------------------------------------------------#
def train(model, train_loader, optimizer, epoch, device, test_loader):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    final_losses = []
    for _batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        final_loss = criterion(output, target)     
        final_loss.backward()
        optimizer.step()
        final_losses.append(final_loss.item())

    final_loss = np.mean(final_losses)

    if log == 1:
        accuracy = evaluate_accuracy(model,test_loader,device)
        save_epoch_data(epoch,final_loss,accuracy)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_general_seed():
    set_seed(general_seed)

#--Datenvorbereitung-------------------------------------#
#--------------------------------------------------------#
#--Seed setzen
set_general_seed()

#--wenn random dp/init erzeuge weitere Seeds
if dp_mode == 1 or init_mode == 1:
    var_seeds = torch.randperm(num_trainings)

#--Datensatz laden
if ds == 0:  # MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset   = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset    = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_dataset, original_indices     = filter_mnist(train_dataset, classes=[3, 5])
    outliers_indices                    = [original_indices.index(idx) for idx in outliers_indices if idx in original_indices]

    test_dataset, _                     = filter_mnist(test_dataset, classes=[3, 5])

else:  # Breast Cancer Wisconsin (Diagnostic)
    set_seed(0)
    X, y = load_breast_cancer(return_X_y=True)

    label_0 = [(x, label) for x, label in zip(X, y) if label == 0]
    label_1 = [(x, label) for x, label in zip(X, y) if label == 1]

    np.random.shuffle(label_0)
    np.random.shuffle(label_1)

    train_label_0 = label_0[:162]
    train_label_1 = label_1[:166]

    test_label_0 = label_0[162:212]
    test_label_1 = label_1[166:216]

    train_data = train_label_0 + train_label_1
    test_data = test_label_0 + test_label_1

    np.random.shuffle(train_data)
    np.random.shuffle(test_data)

    train_X, train_y = zip(*train_data)
    test_X, test_y = zip(*test_data)

    train_dataset = BreastCancerDataset(train_X, train_y)
    test_dataset = BreastCancerDataset(test_X, test_y)

    set_seed(general_seed)

#--Datensatz vorbereiten
all_indices         = list(set(range(len(train_dataset))) - set(outliers_indices))          # alle Indizes (ohne vorausgewählte outliers)
num_select_indices  = num_trainings - len(outliers_indices)                                 # Anzahl der zufällig auszuwählenden Indizes
selected_indices    = torch.randperm(len(all_indices))[:num_select_indices].tolist()        # zufällige Auswahl der restlichen Indizes
selected_indices    = outliers_indices + [all_indices[i] for i in selected_indices]         # ausgewählte Datenpunkte 

remaining_indices   = [i for i in all_indices if i not in selected_indices]                 # übrige Indizes
train_indices       = torch.randperm(len(remaining_indices))[:length_dataset].tolist()      # definiere Indizes des Hauptdatensatzes
train_indices       = [remaining_indices[i] for i in train_indices]                      

#print("Schnittmenge: ", np.intersect1d(train_indices, selected_indices))

# speichern der Trainingsparameter in die Datenbank
if save_to_db == 1:
    beschreibung  = experiment + "." + str(general_seed)
    run  = save_settings(config, exp_config, beschreibung)

#--Ausgabe aktueller Einstellungen-----------------------#
#--------------------------------------------------------#
print_table(current_model, current_dataset, save_to_db, dataset_mode, num_trainings, num_epochs, length_dataset, general_seed)

if save_to_db == 1:
    print(Fore.BLUE + f"\nID des runs in der Datenbank: {Fore.YELLOW}{run}\n" + Style.RESET_ALL)

#--Training----------------------------------------------#
#--------------------------------------------------------#
for i in tqdm(range(0, num_trainings)):
    set_general_seed()
    if log == 1:
        remove_existing_file()

    # wenn Initialisierung randomisiert sein soll, ändere den seed
    if ds == 0:
        if init_mode == 1:
            set_seed(var_seeds[i])
            model = MNIST_Model().to(device)
            set_general_seed()
        else:
            model = MNIST_Model().to(device)
            reset_all_weights(model)
    else:
        if init_mode == 1:
            set_seed(var_seeds[i])
            model = BCW_Model().to(device)
            set_general_seed()
        else:
            model = BCW_Model().to(device)
            reset_all_weights(model)
    
    optimizer       = optim.SGD(model.parameters(), lr=learning_rate)
    privacy_engine  = PrivacyEngine(accountant="rdp")
    model.train()
    final_loss      = 0
  
    if dataset_mode == 1:
        dp_index = selected_indices[i]
    else:
        dp_index = selected_indices[0]
    
    final_indices   = train_indices + [dp_index]    
    final_dataset   = Subset(train_dataset, final_indices)

    if dp_mode == 1:
        batching_g = torch.Generator()
        batching_g.manual_seed(int(var_seeds[i]))
        train_loader    = DataLoader(final_dataset, batch_size=batch_size, shuffle=True, generator=batching_g)
    else:
        train_loader    = DataLoader(final_dataset, batch_size=batch_size, shuffle=True)
    

    private_loader      = train_loader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # privacy engine - opacus
    if make_private == 1:
        model, optimizer, private_loader = privacy_engine.make_private_with_epsilon(
            module          = model,
            optimizer       = optimizer,
            data_loader     = train_loader,
            epochs          = num_epochs,
            target_epsilon  = target_epsilon,
            target_delta    = delta,
            max_grad_norm   = clipping,
        )

    for epoch in range(1, num_epochs + 1):
        loader = train_loader if data_processing == 1 else private_loader   # sampling
        train(model, loader, optimizer, epoch, device=device, test_loader=test_loader)
        if make_private == 1:
            eps_batching = get_epsilon_batching(
            noise_multiplier    = optimizer.noise_multiplier,  # type: ignore
            epochs              = epoch,
            delta               = delta,
        )  

    if log == 1 and save_to_db == 0:
        print(evaluate_accuracy(model, test_loader, device))

    weight = sum([np.prod(p.size()) for _, p in model.named_parameters()])
    print(weight)
    
    # speichern der Messergebnisse
    if save_to_db == 1:
        test_loader     = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        final_accuracy  = evaluate_accuracy(model, test_loader, device)
        if make_private == 1:
            final_epsilon   = privacy_engine.get_epsilon(delta)
        else:
            final_epsilon   = 0.0
        
        if log == 1:
            print(final_accuracy)

        # speichern des Modells
        directory       = f"/Volumes/bachelor/models/{run}"
        model_name      = f"{run}_run{i+1}.pth"

        if i == 0:
            os.makedirs(directory, exist_ok=True)

        model_directory = f"{directory}/{model_name}"
        torch.save(model.state_dict(), model_directory)

        save_training(run, i, final_loss, final_accuracy, final_epsilon, eps_batching, model_name)

#print_epoch_data()