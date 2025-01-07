from setuptools import setup, find_packages

setup(
    name='serial_HM_pilEAUte_2024',
    version='0.1',
    packages=find_packages(include=['src', 'src.*', 'utils', 'utils.*', 'scripts']),
)