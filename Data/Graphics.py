# Graphics.py :
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class Graphics:
    
    def __init__(self):
        pass
    
    def show_distribution(self,column,df):
        df[column].plot(kind='hist', bins=30, edgecolor='black')
        plt.title("Distribution of Your Column")
        plt.xlabel("Values")
        plt.ylabel("Frequency")
        plt.show()