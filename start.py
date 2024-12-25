import json
import subprocess

#--Arrays für die Schleifenwerte
experiments = ["1-1", "1-2", "1-3", "2-1", "2-2", "2-3", "3-1", "3-2", "3-3"]
seed_values = [8, 42, 123, 321, 678, 876, 1735, 2345, 4444, 8765]

#--Pfad zur Konfigurationsdatei
config_path = "config.json"

#--Lade die ursprüngliche Konfiguration
with open(config_path, "r") as file:
    config = json.load(file)

for exp in experiments:
    config["settings"]["experiment"] = exp
    print(f"Starte Experiment {exp}")
    for seed in seed_values:       
        config["parameters_wbc"]["general_seed"] = seed
        print(f"setzte general_seed auf {seed}")

        #--Speichere die geänderte Konfiguration zurück in die Datei
        with open(config_path, "w") as file:
            json.dump(config, file, indent=4)

        #--train.py aufrufen
        try:
            subprocess.run(["python", "train.py"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Fehler beim Ausführen von train.py mit Dataset {exp} und Seed {seed}: {e}")