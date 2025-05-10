import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# UsBirths.csv dosyasını oku
df = pd.read_csv("UsBirths.csv")




#                                      DATASET'E YENİ FEATURE'LAR EKLE

# 1
# İleri düzey eğitim mi? Yüksek lisans ve üstü mü?  1:Evet 0:Hayır 
# Education Level Code 7 veya 8 ise: 1 
df["IsAdvancedDegree"] = df["Education Level Code"].apply(lambda x: 1 if x in [7, 8] else 0)

# 2
# Average Age of Mother < 25 ise: 1 değilse: 0
df["IsYoungMother"] = df["Average Age of Mother (years)"].apply(lambda x: 1 if x < 25 else 0)

# 3
# Average Age of Mother > 35 ise: 1 değilse: 0
df["IsOlderMother"] = df["Average Age of Mother (years)"].apply(lambda x: 1 if x > 25 else 0)

# 4
# Doğum ağırlığı < 2500 ise: 1 değilse: 0
df["IsLowBirthWeight"] = df["Average Birth Weight (g)"].apply(lambda x: 1 if x < 2500 else 0)

# 5
# State bilgisine göre bölge ataması 
region_map = {
    # Northeast
    'CT': 'Northeast', 'ME': 'Northeast', 'MA': 'Northeast', 'NH': 'Northeast',
    'RI': 'Northeast', 'VT': 'Northeast', 'NJ': 'Northeast', 'NY': 'Northeast',
    'PA': 'Northeast',

    # Midwest
    'IL': 'Midwest', 'IN': 'Midwest', 'MI': 'Midwest', 'OH': 'Midwest',
    'WI': 'Midwest', 'IA': 'Midwest', 'KS': 'Midwest', 'MN': 'Midwest',
    'MO': 'Midwest', 'NE': 'Midwest', 'ND': 'Midwest', 'SD': 'Midwest',

    # South
    'DE': 'South', 'FL': 'South', 'GA': 'South', 'MD': 'South', 'NC': 'South',
    'SC': 'South', 'VA': 'South', 'DC': 'South', 'WV': 'South', 'AL': 'South',
    'KY': 'South', 'MS': 'South', 'TN': 'South', 'AR': 'South', 'LA': 'South',
    'OK': 'South', 'TX': 'South',

    # West
    'AZ': 'West', 'CO': 'West', 'ID': 'West', 'MT': 'West', 'NV': 'West',
    'NM': 'West', 'UT': 'West', 'WY': 'West', 'AK': 'West', 'CA': 'West',
    'HI': 'West', 'OR': 'West', 'WA': 'West'
}

df["Region"] = df["State Abbreviation"].map(region_map)

# 6
# Eğer anne genç VE yüksek eğitimliyse: 1 değilse: 0.

#Genç Anne: IsYoungMother == 1
#Yüksek Eğitimli: Education Level Code >= 6 (Yani: Lisans ve üzeri)
df["YoungHighlyEducated"] = ((df["IsYoungMother"] == 1) & (df["Education Level Code"] >= 6)).astype(int)



#                                      TARGET (HEDEF DEĞİŞKEN)

# Median eşiğini hesapla
threshold = df["Number of Births"].median()

# Hedef değişkeni oluştur
df["BirthRateClass"] = df["Number of Births"].apply(lambda x: "High" if x >= threshold else "Low")

# Değişiklikleri kontrol etmek için ilk birkaç satırı yazdır
print(df[["Number of Births", "BirthRateClass"]].head())

df.to_csv("dataset.csv", index=False)