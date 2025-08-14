# PCA_UMAP_tSNE
A small project reducing image dimensionality using techniques like PCA, t-SNE, parametric UMAP and UMAP. Reduced images are subsequently visualised in 2 or 3 dimensions. The embeddings created with PCA and parametric UMAP are also used as input for a simple model to make predictions. For t-SNE and UMAP this is not possible as these techniques cannot be used to create embeddings in a test dataset.

Run the notebooks in `scripts` to perform dimensionality reduction on images. For the pilEAUte dataset all techniques are tried. For other datasets only UMAP is used for visualisation, no other techniques or predictions are tested.


