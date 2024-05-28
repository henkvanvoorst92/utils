#setup of pacakage
from setuptools import setup

setup(
    name='utils',
    version='0.0.1',
    author='Henk van Voorst',
    author_email='henkvanvoorst92',
    description='A lot of very usefull functions and classes used for all kinds of things',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/henkvanvoorst92/utils',
    py_modules=['ants_utils', 'dataframe', 'distances',
                'maskprocess', 'metrics', 'postprocessing',
                'preprocessing', 'registration', 'torch_utils',
                'utils', 'visualize'],
    install_requires=[
        "pytorch==2.1.0",
        "torchvision==0.16.0",
        "torchaudio==2.1.0",
        "pytorch-cuda==11.8",
        "simpleitk",
        "scikit-learn==0.24.2",
        "scipy=1.7.3",
        "pydicom"
        "seaborn==0.11",
        "pandas==1.5",
        "openpyxl",
        "tqdm",
        "antspyx"
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='3.11',
)
