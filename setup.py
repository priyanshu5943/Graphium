from setuptools import setup, find_packages

setup(
    name='graphium',
    version='0.1.0',
    description='A clean and efficient toolkit for static and interactive EDA visualizations.',
    
    author='Priyanshu Shukla',
    author_email='your.email@example.com',

    packages=find_packages(exclude=["legacy_univariate_viz"]),
    
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'plotly',
        'scipy',
        'ipython'
    ],

    python_requires='>=3.8',

    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Visualization',
    ],

    license='MIT'
)
