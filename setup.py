from setuptools import setup, find_packages
# The setup script for the mlproject package. It will find out all packages that will be available in the mlproject directory we created.

from typing import List
HYPEN_E_DOT = '-e .'
def get_requirements(file_path:str) -> list:
    '''
    This function will return the list of requirements from the given file path.
    '''
    requirements = []
    with open(file_path) as f:
        requirements = f.readlines()
        # requirements = [req.replace('\n', '') for req in requirements] # Every time it reads the requirements file line by line, automatically \n will be added, So to remove that we have added this line.
        requirements = [req.strip() for req in requirements] # This will remove any leading or trailing whitespace characters, including the newline character.     
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

# All the metadata about the package will be defined in this setup function. We will just define the name and version for now, but we can add more metadata like author, description, etc. if needed.
setup(
    name='mlproject',
    version='0.0.1',
    author='Mann',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),

)