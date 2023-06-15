import os
from setuptools import find_packages, setup


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths


extra_files = package_files('learning_fc/assets')

setup(name='learning_fc',
      version='0.1',
      description='Learning Force Control with 2-DoF Grippers',
      author='Luca Lach',
      author_email='llach@techfak.uni-bielefeld.de',
      url='https://github.com/llach/learning_fc',
      packages=[package for package in find_packages() if package.startswith("learning_fc")],
      package_data={'learning_fc': extra_files},
)