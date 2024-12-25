import csv
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

csv_file = 'epoch_data.csv'
header = ['Epoch', 'Loss', 'Accuracy']

def save_epoch_data(epoch, loss, accuracy):
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
    
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, loss, accuracy])

def remove_existing_file():
    with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)

def print_epoch_data():
    df = pd.read_csv(csv_file)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(df['Epoch'], df['Loss'], color='tab:blue', label='Loss')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy (%)', color='tab:orange')
    ax2.plot(df['Epoch'], df['Accuracy'], color='tab:orange', label='Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    
    plt.title('Loss and Accuracy per Epoch')
    fig.tight_layout()
    
    plt.show()