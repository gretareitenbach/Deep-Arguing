from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name='deeparguing',  # Replace with your package name
    version='0.0.1',
    author="Anonymous Authors",
    author_email="anonymous@example.com",
    description="Anonymous Implementation of Deep Arguing",
    long_description="file: README.md",
    long_description_content_type="text/markdown",
    url='https://github.com/anonymous/gradual-aacbr',
    project_urls = {
        "Bug Tracker": "https://github.com/anonymous/gradual-aacbr/issues",
    },
    classifiers=[
        " Programming Language :: Python :: 3",
        " License :: OSI Approved :: MIT License",
        " Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],    
    install_requires=[
        "matplotlib",
        "networkx",
        "numpy",
        "torch",
        "tqdm",
        "scikit-learn",
        "torchviz",
        "pandas",
        "optuna",
    ],
    dependency_links=[
        'https://download.pytorch.org/whl/cu118'
    ],
    python_requires=">=3.12",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
