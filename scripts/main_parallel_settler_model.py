# libraries
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import accuracy_score, classification_report

### TSS effluent error = TSS effluent lab measurement - TSS mech model

start_date = pd.to_datetime("2023-10-01")


# read TSS effluent lab measurement
TSS_eff_measurements = pd.read_csv("data/mech_model/TSS_effluent_WESTinput.txt", sep='\s+', index_col=0)
TSS_eff_measurements = TSS_eff_measurements.iloc[1:].astype(float)
TSS_eff_measurements.index = pd.to_numeric(TSS_eff_measurements.index, errors='coerce')
TSS_eff_measurements.index = start_date + pd.to_timedelta(TSS_eff_measurements.index, unit='D')

# read TSS mech model
TSS_eff_mech = pd.read_csv("data/mech_model/Project1.Dynamic.Simul.out.txt", sep='\s+', index_col=0)
TSS_eff_mech = TSS_eff_mech.iloc[1:].astype(float)
TSS_eff_mech.index = pd.to_numeric(TSS_eff_mech.index, errors='coerce')
TSS_eff_mech.index = start_date + pd.to_timedelta(TSS_eff_mech.index, unit='D')

# Interpolate

path_to_images='data/microscope_images'
TSS_eff_measurements = TSS_eff_measurements.reindex(listdir(path_to_images), method='nearest')
TSS_eff_mech_reindex = TSS_eff_mech.reindex(listdir(path_to_images), method='nearest')

# calculate error
TSS_eff_error=TSS_eff_measurements['TSS_effluent'].values-TSS_eff_mech_reindex['.SST_1.X_Out'].values

path_to_images='data/microscope_images'
image_folders = listdir(path_to_images)
i=0
labels = []
for folder in image_folders:
    path = f"{path_to_images}/{folder}/basin5/10x"
    images_list = listdir(path)
    print(folder)
    for image in images_list:
        labels.append(TSS_eff_error[i])
    i=i+1

### read embeddings = inputdata regression model
embeddings=pd.read_excel('results/dimension_reduction_results/UMAP_3D_embeddings.xlsx', header=None)


### make train and test dataset

n_train = int(len(embeddings) * 0.6)  # 80% van de data

# Train-test splits
X_train, X_test = embeddings[:n_train], embeddings[n_train:]
y_train, y_test = labels[:n_train], labels[n_train:]

# Random Forest model initialiseren
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Trainen van het model
rf.fit(X_train, y_train)

# Voorspellingen maken op de test set
y_pred = rf.predict(X_test)


plt.plot(y_pred)
plt.plot(y_test)

