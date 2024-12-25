Eine empirische Bewertung des Einflusses einzelner Datenpunkte im DP-SGD
-------------------------
Eren Kocadag

Code zur Bachelorarbeit

----
### Installation
1. 
Es steht eine `env.yml`, mit der eine kompatible conda-Umgebung erstellt werden kann. Hierfür muss der folgende Befehl ausgeführt werden:
```bash
conda env create -f env. yml
``` 
Die conda-Umgebung lässt sich mit folgendem Befehl starten:
```bash
conda activate ba-ek
```
2.
#### a) mit Datenbank
Sollen die erzeugten Daten in einer Datenbank gespeichert werden, so muss zunächst eine Installation von PostgreSQL vorliegen. Genaueres ist der [PostgreSQL Website](https://www.postgresql.org/) zu entnehmen. Je nach Konfiguration muss dann die `connect()` - Funktion in `database.py` angepasst werden.
Anschließend kann mithilfe der in `/sql` vorliegenden Datei `ddl.sql` ein passendes Datenbankschema erzeugt werden. Zur Datenspeicherung muss der Parameter `save_to_db` in der `config.json` - Datei auf `1` gesetzt werden.
#### b) ohne Datenbank
Der Code kann auch ohne Datenbank ausgeführt werden. Dafür ist es notwendig, den Parameter `save_to_db` in der `config.json` - Datei auf `0` zu setzen.

-----
### Training
1. 
Die Eigenschaften des Trainings lassen sich über die `config.json` steuern. Um ein individuelles Training zu starten, muss das Experiment bei dem Parameter `experiment` ausgewählt werden. Die Nummerierung ist wiefolgt:

| Nr. | - | -
|:-----:|:-----:|:-------:|
| 1-1 | SGD | variabler Datensatz |
| 1-2 | SGD | variable Datenauswahl |
| 1-3 | SGD | variable Initialisierung |
| 2-1 | DP-SGD | variabler Datensatz | 
| 2-2 | DP-SGD | variable Datenauswahl |
| 2-3 | DP-SGD | variable Initialisierung |
| 3-1 | DP-SGD mit Batching | variabler Datensatz |
| 3-2 | DP-SGD mit Batching | variable Datenauswahl |
| 3-3 | DP-SGD mit Batching | variable Initialisierung |

2. 
Die weiteren Parameter sind wiefolgt zu setzen:
```json
    "settings": {
        "device": "mps", --> Device für torch: cpu: CPU, cuda: Grafikkarte, mps: Apple Silicon Grafikkerne
        "log": 0, --> Log in Terminal
        "save_to_db": 0, --> Datenbankspeicherung
        "current_dataset": 1, --> 0: MNIST, 1: BCW
        "experiment": "3-1" --> siehe oben
    }
```
Der globale Seed lässt sich, je nach Datensatz, in `parameters_mnist` bzw. in `parameters_bcw` mit dem Parameter `general_seed` variieren.

3. 
Ist eine Datenbankspeicherung vorgesehen, so muss in `train.py` der Dateipfad für die Speicherung der Modelle individuell angepasst werden:
```python
        directory       = f"/Volumes/bachelor/models/{run}"
```
4. 
a) Soll ein einfaches Training einer der Experimente durchgeführt werden, so ist `train.py` auszuführen.

b) Sollen alle Experimente automatisch an einem Datensatz durchgeführt werden, so muss der jeweilige Datensatz in `config.json` im Parameter `current_dataset` festgelegt werden und anschließend `start.py` ausgeführt werden. Dies führt alle Experimente der Tabelle an den in der Arbeit verwendeten 10 globalen Seeds aus.

----
### Distanzen 
Um die Distanzen zwischen den Modellen zu messen, muss zunächst ein mit eingeschalteter Datenbankspeicherung durchgeführtes Experiment vorliegen. Vor Messung der Distanzen muss auch in `distance.py` der Pfad, in dem das Training die Modelle gespeichert hat, individuell angepasst werden.
```python
models_directory_path = f"/Volumes/bachelor/models/{run}"
```
Anschließend kann `distance.py` ausgeführt werden. Anschließend sind die Runs (ID's aus der Datenbank) anzugeben, für die die Berechnung erfolgen soll. Die Distanzen werden dann in der Datenbanktabelle `distance` gespeichert und verwendet werden. Zur weiteren Verwendung der Distanzen mit python, sind in `database.py` Funktionen definiert, die die Ergebnisse in verschiedenen Formen aus der Datenbank beziehen.

Die Funktion `parallel_distance.py` parallelisiert die Distanzberechnung auf alle verfügbaren Ressourcen. Zur Verwendung müssen hier die Runs manuell im Code geändert werden:
```python
von = 171   # Start-Run
bis = 191   # End-Run
ds  = 1     # Dataset [0:MNIST][1:WBC]
```
Es ist zwingend `ds` anzupassen, mit welchem Datensatz der Run durchgeführt wurde.