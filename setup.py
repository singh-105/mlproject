from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

# FIX 1: Use List[str] instead of list[str]
def get_requirements(file_path: str) -> List[str]:
    '''
    this function will return the requirements 
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='harsh',
    author_email='manojrsingh1974@gmail.com',
    packages=find_packages(), # FIX 2: Changed 'package' to 'packages' (plural)
    install_requires=get_requirements('requirements.txt')
)