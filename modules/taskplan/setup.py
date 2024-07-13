from setuptools import setup, find_packages


setup(name='taskplan',
      version='1.0.0',
      description='Core code for task planning in ai2thor environment',
      license="MIT",
      author='Raihan Islam Arnob',
      author_email='rarnob@gmu.edu',
      packages=find_packages(),
      install_requires=['numpy', 'matplotlib'])