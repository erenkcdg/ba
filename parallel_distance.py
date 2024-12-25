import torch
import os
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Lock, Manager

from models import *
from database import save_distance

# --Definitionen------------------------------------------#
device = torch.device('cpu')  # CPU erzwingen

# Konstanten
von = 171   # Start-Run
bis = 191   # End-Run
ds  = 1     # Dataset [0:MNIST][1:WBC]

# Modelle initialisieren
def init_model(ds):
    if ds == 0:
        return MNIST_Model().to(device)
    else:
        return BCW_Model().to(device)

# --Funktionen--------------------------------------------#
# Schlüssel in der state_dict bereinigen
def clean_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_module."):
            new_state_dict[k[len("_module."):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

# PyTorch-basierte L2-Distanzberechnung auf der CPU
def l2_distance_cpu(tensor1, tensor2):
    return torch.norm(tensor1 - tensor2, p=2).item()

# Funktion für L2-Distanzberechnung eines Runs
def parallel_l2_distance(run, models_directory_path, ds, lock, print_lock):
    model_a = init_model(ds)

    # Lade alle Modelle als Tensoren auf die CPU
    model_list = []
    num_models = len([name for name in os.listdir(models_directory_path) 
                      if os.path.isfile(os.path.join(models_directory_path, name))])
    with print_lock:
        print(f'Prozess {mp.current_process().name}: Es gibt {num_models} Dateien für Run {run}.')

    for i in range(num_models):
        state_dict = torch.load(f"{models_directory_path}/{run}_run{i+1}.pth", map_location=device)
        model_a.load_state_dict(clean_state_dict(state_dict))
        model_tensor = torch.cat([param.flatten() for param in model_a.parameters()]).to(device)
        model_list.append(model_tensor)

    # Berechne paarweise Distanzen auf der CPU
    for i in range(num_models):
        for j in range(i + 1, num_models):
            distance = l2_distance_cpu(model_list[i], model_list[j])
            save_distance(run, i, j, distance)

    with print_lock:
        print(f'Prozess {mp.current_process().name}: Run {run} abgeschlossen.')

# Worker-Funktion für die Prozesse
def worker(run_queue, ds, lock, print_lock):
    while not run_queue.empty():
        run = None
        with lock:
            if not run_queue.empty():
                run = run_queue.get()

        if run is not None:
            models_directory_path = f"/Volumes/bachelor/models/{run}"
            parallel_l2_distance(run, models_directory_path, ds, lock, print_lock)

# Hauptprogramm
if __name__ == "__main__":
    manager = Manager()
    run_queue = manager.Queue()
    lock = Lock()
    print_lock = Lock()  # Lock für synchronisiertes Printing

    # Runs in die Queue laden
    for run in range(von, bis):
        run_queue.put(run)

    # Anzahl der parallelen Prozesse
    num_processes = min(mp.cpu_count(), (bis - von))

    # Prozesse starten
    processes = []
    for _ in range(num_processes):
        p = mp.Process(target=worker, args=(run_queue, ds, lock, print_lock))
        processes.append(p)
        p.start()

    # Warten, bis alle Prozesse abgeschlossen sind
    for p in processes:
        p.join()