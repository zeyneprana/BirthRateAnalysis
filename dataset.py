
# Kaggle'dan aldığım orijinal veri seti = UsBirths.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# UsBirths.csv dosyasını oku
df = pd.read_csv("UsBirths.csv")


def datasetControlFunctions():

    #İlk 5 satırı kontrol et
    print(df.head())

    #data type'larını verir
    print(df.dtypes)

    #konu başlıkları
    print(df.columns)

    #Boşluklara bak
    print(df.isnull().sum())

    print(df.info())



#Unknown olan satırı çıkar
df = df[df["Education Level Code"] != -9]







