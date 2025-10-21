'''
The setup.py file is an essential part of packaging and 
distributing Python projects. It is used by setuptools 
(or distutils in older Python versions) to define the configuration 
of this project, such as its metadata, dependencies, and more
'''

from setuptools import find_packages,setup
from typing import List

def get_requirements()->List[str]:
    """
    Thiss function will return list of requirements
    
    """
    HYPEN_E_DOT = '-e .'
    requirement_lst:List[str]=[]
    try:
        with open('./requirements.txt','r') as file:
            lines=file.readlines()
            for line in lines:
                requirement=line.strip()
                ## ignore empty lines and -e .
                if requirement and requirement != HYPEN_E_DOT:
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found")

    return requirement_lst

setup(
    name="score_prediction_aws",
    version="0.0.1",
    author="Alejandro Garcia",
    author_email="garcialejan@gmail.com",
    packages=find_packages(where="."),
    install_requires=get_requirements()
)