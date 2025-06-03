from setuptools import setup, find_packages
from typing import List


HYPEN = '-e .'
def get_requires(file_path: str) -> List[str]:
    """Read requirements from a file and return them as a list."""
    requirements = []
    try:
        with open(file_path, 'r') as file:
            requirements = [line.strip() for line in file if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. No requirements will be installed.")
    
    if HYPEN in requirements:
        requirements.remove(HYPEN)
    return requirements


setup(
    name= "Customer Churn",
    version= "0.1",
    author= "Kubilay ALICI",
    author_email= "kkubilay.alici@gmail.com",
    packages=find_packages(),
    install_requires= get_requires('requirements.txt'),
    description= "A package for analyzing and predicting customer churn using machine learning techniques.",
    long_description=open('README.md').read(),  
)