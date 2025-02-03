# libraries
from utils.helpers import interpolate_time
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from xgboost import XGBRegressor
import pickle
import albumentations as A
import numpy as np
from skimage import feature
from concurrent.futures import ProcessPoolExecutor
from utils.helpers import extract_images

start_date = pd.to_datetime("2023-10-01")

# read TSS effluent lab measurement
TSS_eff_measurements = pd.read_csv("data/mech_model/lab_measurements_TSS_effluent.txt", sep='\s+', index_col=0)
TSS_eff_measurements = TSS_eff_measurements.iloc[1:].astype(float)
TSS_eff_measurements.index = pd.to_numeric(TSS_eff_measurements.index, errors='coerce')
TSS_eff_measurements.index = start_date + pd.to_timedelta(TSS_eff_measurements.index, unit='D')

# read TSS mech model
TSS_eff_mech = pd.read_csv("data/mech_model/Project1.Dynamic.Simul.out.txt", sep='\s+', index_col=0)
TSS_eff_mech = TSS_eff_mech.iloc[1:].astype(float)
TSS_eff_mech.index = pd.to_numeric(TSS_eff_mech.index, errors='coerce')
TSS_eff_mech.index = start_date + pd.to_timedelta(TSS_eff_mech.index, unit='D')

path_to_images='data/microscope_images'

#interpolate to dates on which images were taken
TSS_eff_measurements_reindex=interpolate_time(TSS_eff_measurements, listdir(path_to_images))
TSS_eff_mech_reindex=interpolate_time(TSS_eff_mech, listdir(path_to_images))

# calculate error
TSS_eff_error=TSS_eff_measurements_reindex['TSS_effluent'].values-TSS_eff_mech_reindex['.SST_1.X_Out'].values

### generate embeddings UMAP from augmented images and save labels
path_to_train_images='data/microscope_images_train'
train_images = extract_images(path_to_train_images, image_type='all', magnification=10)
train_image_folders = listdir(path_to_train_images)
i=0
train_labels=[]
for folder in train_image_folders:
    path = f"{path_to_train_images}/{folder}/basin5/10x"
    images_list = listdir(path)
    for image in images_list:
        train_labels.append(TSS_eff_error[i]) #give every image a label (=error TSS)
    i=i+1

# data augmentation techniques
augment = A.Compose([
    A.Blur(p=0.5, blur_limit=2),
    A.HorizontalFlip(p=0.5), 
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.1, p=0.8)
])

# load fitted umap
with open("results/dimension_reduction_results/umap_model_5dim_traindata.pkl", "rb") as f:
    loaded_reducer = pickle.load(f)

column_names = [f"Pixel_{i+1}" for i in range(540*450)]
X_train = []
y_train = []

# Start parallel processing
with ProcessPoolExecutor() as executor:
    for i, (image, label) in enumerate(zip(train_images, train_labels)):
        print(i+1)
        # resize and greyscale, convert to numpy array for further processing
        grey_image = image.resize((540, 450)).convert('L')
        grey_image = np.array(grey_image)

        # data augmentation (generate 5 new images)
        augmented_images = [grey_image]
        for _ in range(5):  
            augmented = augment(image=grey_image)['image']
            augmented_images.append(augmented)

        # edge detection and flattening
        processed_images = np.array([feature.canny(img, sigma=0.8) for img in augmented_images])
        flattened_images = processed_images.reshape(len(processed_images), -1)

        # umap recution
        reduced_features = loaded_reducer.transform(pd.DataFrame(flattened_images, columns=column_names))

        # save results
        X_train.extend(reduced_features)
        y_train.extend([label] * len(augmented_images))

pd.DataFrame(y_train).to_csv("results/labels_augmented.csv", index=False, header=False)
pd.DataFrame(X_train).to_csv("results/umap_features_train_augmented.csv", index=False, header=False)


#___________________________________
path_to_all_images='data/microscope_images'
all_images = extract_images(path_to_all_images, image_type='all', magnification=10)
all_image_folders = listdir(path_to_all_images)
i=0
all_labels=[]
for folder in all_image_folders:
    path = f"{path_to_all_images}/{folder}/basin5/10x"
    images_list = listdir(path)
    for image in images_list:
        all_labels.append(TSS_eff_error[i]) #give every image a label (=error TSS)
    i=i+1

all_features=[]
# Start parallel processing
for i, (image, label) in enumerate(zip(all_images, all_labels)):
    print(i+1)
    # resize and greyscale, convert to numpy array for further processing
    grey_image = image.resize((540, 450)).convert('L')
    grey_image = np.array(grey_image)

    # edge detection and flattening
    processed_image = np.array(feature.canny(grey_image, sigma=0.8))
    flattened_image = processed_image.reshape(1, -1)

    # umap recution
    reduced_features = loaded_reducer.transform(pd.DataFrame(flattened_image, columns=column_names))

    # save results
    all_features.extend(reduced_features)

pd.DataFrame(all_labels).to_csv("results/all_labels.csv", index=False, header=False)
pd.DataFrame(all_features).to_csv("results/all_umap_features.csv", index=False, header=False)


# Random Forest model initialiseren
#rf = RandomForestRegressor(n_estimators=100, random_state=42)
bst = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.3, objective='reg:squarederror')

# Trainen van het model
#rf.fit(X_train, y_train)
bst.fit(X_train, y_train)

y_pred = bst.predict(all_features)

#calc average predicted error
image_folders = listdir(path_to_images) 
average_error_preds = []
i=0
for folder in image_folders:
    path = f"{path_to_images}/{folder}/basin5/10x"
    images_list = listdir(path)
    temporary=[]
    for image in images_list:
        temporary.append(y_pred[i])
        i+=1
    average_error_preds.append(sum(temporary)/len(temporary))

TSS_hybridmodel=TSS_eff_mech_reindex['.SST_1.X_Out']+average_error_preds
TSS_hybridmodel.index = TSS_eff_measurements_reindex.index
# plot results
split_index = int(len(TSS_eff_error) * 0.7)

plt.figure(figsize=(10,4), dpi=150)
plt.plot(TSS_eff_measurements_reindex, '.-', label='Measurements', color='blue')
plt.plot(TSS_eff_mech['.SST_1.X_Out'], '-', linewidth=1.5, label='Mechanistic model output', color='green')
plt.plot(TSS_hybridmodel[:split_index], '.-', label='HM predictions (train)', color='orange')
plt.plot(TSS_hybridmodel[split_index:], '.-', label='HM predictions (test)', color='red')
plt.xlabel("Time")
plt.ylabel("TSS effluent (mg/L)")
plt.legend()
plt.show()