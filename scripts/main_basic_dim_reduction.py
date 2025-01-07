import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils.helpers import extract_images_and_labels
from src.images_preprocessing.images_preprocessing import preprocess_images

# extract images (either image_type='all','old', 'new' regarding to which microscope)
base_folder = "data/microscope_images"
all_images, image_labels = extract_images_and_labels("data/microscope_images", 'data/settling_tests/Batch_settleability_test_SVI.xlsx', image_type='all')

# preprocess images
image_df=preprocess_images(all_images=all_images, size=(540,450), method='edges', flatten=True, show_example=True)

#define parameters
n_neighbors=50
min_dist=0.5

#calculate umap
reducer = umap.UMAP(n_neighbors=n_neighbors, #default 15
                    min_dist=min_dist, #default 0.1
                    metric='l1', #default categorical
                    random_state=42,
                    n_components=3)
embedding = reducer.fit_transform(image_df)

fig = plt.subplots()
sc=plt.scatter(embedding[:, 0], embedding[:, 1], c=image_labels)
plt.title(f"UMAP Grayscaled images - neighbors={n_neighbors} - min_dist={min_dist}")
cbar = plt.colorbar(sc)
cbar.set_label('SSVI (mL/g)') 
# plt.savefig('results/dimension_reduction/UMAP/UMAP_all_images_contraststretch.png')
plt.show()



# Define threshold for outliers
threshold = 250

# Identify outliers
image_labels2 = image_labels.ravel()
outliers = (image_labels2 > threshold).ravel()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot non-outliers
sc = ax.scatter(embedding[~outliers, 0], embedding[~outliers, 1], embedding[~outliers, 2], c=image_labels[~outliers], cmap='viridis', label='SVI < 250')
# Plot outliers
ax.scatter(embedding[outliers, 0], embedding[outliers, 1], embedding[outliers, 2], c='red', label='SVI >= 250')

#plt.title(f"UMAP Grayscaled images - neighbors={n_neighbors} - min_dist={min_dist}")
cbar = plt.colorbar(sc)
cbar.set_label('SVI (mL/g)')
plt.legend()

# Adjust the view
ax.view_init(elev=20, azim=30)  # Change elevation and azimuth angles

# Set axis labels
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')

# Set axis limits if needed
ax.set_xlim(min(embedding[:, 0]), max(embedding[:, 0])*0.95)  # Example zoom for x-axis
ax.set_ylim(min(embedding[:, 1]), max(embedding[:, 1])*0.95)  # Example zoom for y-axis
ax.set_zlim(min(embedding[:, 2]), max(embedding[:, 2])*0.95)  # Example zoom for z-axis

plt.tight_layout() 
plt.savefig('results/dimension_reduction_results/UMAP/UMAP_all_images_edge_3d.png')




### I just did umap now but I have to generate results again for pca and tsne (oral exam is too close)





# pca = PCA(n_components=10)
# points = pca.fit_transform(image_df)

# plt.figure()
# sc=plt.scatter(points[:,0],points[:,1], c=image_labels)
# cbar = plt.colorbar(sc)
# cbar.set_label('SVI (mL/g)') 
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.title(f"PCA Grayscaled images")
# # plt.savefig('results/dimension_reduction/PCA/scatter_plot_all_images_contraststretch.png')
# plt.show()

# explained_variance = pca.explained_variance_ratio_

# plt.figure()
# plt.bar(range(1, len(explained_variance) + 1),
#          explained_variance*100) 
# plt.xlabel('Principal Component')
# plt.ylabel('Explained Variance (in %)')
# plt.title('Explained Variance by Principal Component')
# plt.xticks(range(1, len(explained_variance) + 1))
# # plt.savefig('results/dimension_reduction/PCA/explained_variance_all_images_contraststretch.png')
# plt.show()


# perplexity=2 #default 30
# tsne = TSNE(perplexity=perplexity, 
#             random_state=42)
# points = tsne.fit_transform(image_df)

# sc=plt.scatter(points[:,0], points[:,1], c=image_labels)
# cbar = plt.colorbar(sc)
# cbar.set_label('SVI (mL/g)') 
# plt.title(f"t-SNE Grayscaled images - perplexity={perplexity} ")
# # plt.savefig('results/dimension_reduction/tSNE/tSNE_all_images_contraststretch.png')