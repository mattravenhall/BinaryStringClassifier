#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name='BinaryStringClassifier',
    version='0.0.1',
    description='Binary classification of strings, particularly SMILES strings.',
    long_description=open("README.md", 'r').read(),
    long_description_content_type="text/markdown",
    author='Matt Ravenhall',
    url='https://github.com/mattravenhall/BinaryStringClassifier',
    packages=["BinaryStringClassifier"],
    package_dir={"BinaryStringClassifier": "src"},
    include_package_data=True,
    package_data={
        "BinaryStringClassifier": ["src/mrdark.mplstyle"]
    },
    classifiers=[
          'Development Status :: 4 - Beta',
          'Environment :: Console',
          'Operating System :: POSIX :: Linux',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: MacOS :: MacOS X',
          'Programming Language :: Python :: 3',
          ],
    entry_points={
        "console_scripts": [
            'BSC-Model=BinaryStringClassifier.run:main',
            'BSC-Data=BinaryStringClassifier.data:main',
        ]
    },
    install_requires=open('requirements.txt', 'r').readlines(),
    python_requires='>=3.7',
)
