import torch
import torch.nn as nn
import pandas as pd
from tabulate import tabulate
from colorama import Fore, Style
from torch.utils.data import Dataset, DataLoader, Subset
from opacus.accountants.rdp import RDPAccountant
from opacus.accountants.analysis.rdp import get_privacy_spent

def filter_mnist(dataset, classes):
    indices = [i for i, target in enumerate(dataset.targets) if target in classes]
    return Subset(dataset, indices), indices

def get_epsilon_batching(noise_multiplier, epochs, delta):
    alphas = RDPAccountant.DEFAULT_ALPHAS
    rdp = [epochs * (alpha / (2 * noise_multiplier ** 2)) for alpha in alphas]
    eps, best_alpha = get_privacy_spent(orders=alphas, rdp=rdp, delta=delta)
    return float(eps)

def export_data_to_csv(dataset, name):
    data = []
    for x, y in dataset:
        data.append(list(x.numpy()) + [y.item()])
    columns = [f"Feature_{i}" for i in range(len(data[0]) - 1)] + ["Label"]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(f"{name}.csv", index=False)
    
def evaluate_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total

def reset_all_weights(model: nn.Module) -> None:
    @torch.no_grad()
    def weight_reset(m: nn.Module):
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    model.apply(fn=weight_reset)

def print_table(current_model, current_dataset, save_to_db, dataset_mode, num_trainings, num_epochs, length_dataset, seed):
    printed_data = [
        ["Modell", current_model],
        ["Datensatz", current_dataset],
        ["Datenbankspeicherung", (Fore.GREEN + "Ja" if save_to_db == 1 else Fore.RED + "Nein") + Style.RESET_ALL],
        ["Datensatzmodus", "changing" if dataset_mode == 1 else "fixed"],
        ["Anzahl der Trainings", num_trainings],
        ["Anzahl der Epochen pro Training", num_epochs],
        ["Länge des Hauptdatensatzes", length_dataset+1],
        ["Seed", seed]
    ]
    print(tabulate(printed_data, headers=["Parameter", "Wert"], tablefmt="pretty"))
