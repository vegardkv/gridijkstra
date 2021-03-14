import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gridikjstra",
    version="0.1",
    author="Vegard Kvernelv",
    description="Python package wrapping scipy's dijkstra with a grid-based interface",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),    # List of all python modules to be installed
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    py_modules=["gridikjstra"],
    # package_dir={'':'quicksample/src'},     # Directory of the source code of the package
    # install_requires=[]                     # Install other dependencies if any
)
