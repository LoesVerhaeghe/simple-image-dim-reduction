from setuptools import setup, find_packages

setup(
    name='PCA_tSNE_UMAP',
    version='0.1',
    packages=find_packages(include=['src', 'src.*', 'utils', 'utils.*', 'scripts']),
)