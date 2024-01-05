from setuptools import setup, find_packages


with open('requirements.txt', encoding='utf-8') as f:
    required = f.read().splitlines()

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='condconv',
    version='1.0.0',
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=required,
    url='https://github.com/nibuiro/CondConv-pytorch',
    license='MIT',
    author='nibuiro',
    author_email='immay1999@gmail.com',
    description='Implementation of condconv: Conditionally Parameterized Convolutions for Efficient Inference. '
)
