import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import FuncFormatter
from database import get_distances

sns.set_theme(style="whitegrid")

def custom_formatter(x, pos):
    # Wenn der Wert 0 ist, zeige '0'
    if x == 0:
        return '0'
    # Ansonsten auf drei Nachkommastellen runden
    return f'{x:.3f}'

def plot_distances(data, mode, label):
    plt.figure(figsize=(16, 9))

    if mode == 0:  # Violinplot
        sns.violinplot(data=data, x="Run", y="Distanz", bw=.5, cut=1, linewidth=1, palette="Set1")
        
        plt.xlabel('Run')
        plt.ylabel('Distanz')
        plt.title(label)

        # Nachkomma der y-Achse einstellen
        ax = plt.gca()
        ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))
        sns.despine(left=True, bottom=True)

    elif mode == 1:  # Histogramm
        sns.histplot(data, x="Distanz", hue="Run", bins=20, edgecolor="black", palette="Set1", multiple="stack",)
    
        plt.xlabel('Distanz')
        plt.ylabel('Menge')
        plt.title(label)
        plt.grid(True)
        
        # manuell x-Achse einstellen
        xmin, xmax = plt.xlim()
        xticks = np.arange(xmin, xmax, (xmax - xmin) / 10)
        if 0 not in xticks:
            xticks = np.append(xticks, 0)
            xticks.sort()
        plt.xticks(xticks)

        # Nachkomma der x-Achse einstellen
        ax = plt.gca()
        ax.xaxis.set_major_formatter(FuncFormatter(custom_formatter))

    elif mode == 2:  # Stripplot
        sns.stripplot(data=data, x="Run", y="Distanz", hue="Run", palette="Set1", jitter=True, size=5, dodge=True)

        plt.xlabel('Run')
        plt.ylabel('Distanz')
        plt.title(label)

        # Nachkomma der y-Achse einstellen
        ax = plt.gca()
        ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))
        sns.despine(left=True, bottom=True)
    
    plt.show()

# Daten aus der Datenbank abrufen
runs        = input("runs (kommagetrennt): ")
run_list    = [int(run.strip()) for run in runs.split(",")]

all_distances = []
for run in run_list:
    distances = get_distances(run)
    if distances:
        all_distances.extend([(distance, run) for distance in distances])

if all_distances:
    mode    = int(input("Art\t: [0]violin | [1]hist | [2]strip: "))
    label   = input("Label\t: ")
    df      = pd.DataFrame(all_distances, columns=['Distanz', 'Run'])
    print (df)
    plot_distances(df, mode, label)
else:
    print("Keine Distanzen für die angegebenen Runs gefunden.")