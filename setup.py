# setup of pacakage
from setuptools import setup

setup(
    name='utils',
    version='0.0.1',
    author='Henk van Voorst',
    author_email='henkvanvoorst92',
    description='A lot of very usefull functions and classes used for all kinds of things',
    # long_description=open('README.md').read(),
    # long_description_content_type='text/markdown',
    url='https://github.com/henkvanvoorst92/utils',
    py_modules=['ants_utils', 'dataframe', 'distances',
                'maskprocess', 'metrics', 'postprocessing',
                'preprocessing', 'registration', 'torch_utils',
                'utils', 'visualize'],
    install_requires=[
        # "torch >= 2.1.0",
        # "torchvision >= 0.16.0",
        # "torchaudio >= 2.1.0",
        "simpleitk",
        "scikit-learn>=1",
        "numpy>=1.24",
        "scipy>=1.10",
        "aiohttp>=3.8.4",
        "pydicom",
        "seaborn==0.11",
        "pandas",
        "openpyxl",
        "tqdm",
        "antspyx",
        "totalsegmentator"
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='<3.10',
)
