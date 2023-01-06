from setuptools import setup
from setuptools import find_packages

setup(name='gae',
      version='0.0.1',
      description='Implementation of (Variational) Graph Auto-Encoders in Tensorflow',
      author='Thomas Kipf',
      author_email='thomas.kipf@gmail.com',
      url='https://tkipf.github.io',
      download_url='https://github.com/tkipf/gae',
      license='MIT',
      install_requires=['numpy==1.19.2',
                        'tensorflow',
                        'networkx',
                        'scikit-learn',
                        'scipy',
                        'pandas',
                        'seaborn',
                        'yellowbrick'
                        ],
      extras_require={
          'visualization': ['matplotlib'],
      },
      package_data={'gae': ['README.md']},
      packages=find_packages())













##from setuptools import setup
##from setuptools import find_packages
##
##setup(name='gae',
##      version='0.0.1',
##      description='Implementation of (Variational) Graph Auto-Encoders in Tensorflow',
##      author='Thomas Kipf',
##      author_email='thomas.kipf@gmail.com',
##      url='https://tkipf.github.io',
##      download_url='https://github.com/tkipf/gae',
##      license='MIT',
##      install_requires=['pandas',
##                        'numpy',
##                        'tensorflow',
##                        'networkx',
##                        'scikit-learn',
##                        'scipy',
##                        'seaborn',
##                        'yellowbrick'
##                        ],
##      extras_require={
##          'visualization': ['matplotlib'],
##      },
##      package_data={'gae': ['README.md']},
##      packages=find_packages())
