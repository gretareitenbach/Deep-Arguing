from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name='deeparguing',  # Replace with your package name
    version='0.0.1',
    author="Adam Gould",
    author_email="adam.gould19@imperial.ac.uk",
    description="CLArg's Implementation of Deep Arguing",
    long_description="file: README.md",
    long_description_content_type="text/markdown",
    url='https://github.com/TheAdamG/gradual-aacbr',
    project_urls = {
        "Bug Tracker": "https://github.com/TheAdamG/gradual-aacbr/issues",
    },
    classifiers=[
        " Programming Language :: Python :: 3",
        " License :: OSI Approved :: MIT License",
        " Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],    
    install_requires=[
        "matplotlib==3.9.2",
        "networkx==3.2.1",
        "numpy==2.1.1",
        "torch==2.3.0+cu118",
    ],
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
