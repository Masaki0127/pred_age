import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bertpredage",
    version="0.1.0",
    author="higashi-masaki",
    author_email="ls16287j@gmail.com",
    description="You can predict age from BERT with multi questions",
    url="https://github.com/M-H0127/pred_age",
    packages=setuptools.find_packages(),
    install_requires=["numpy", "pandas", "scikit-learn", "torch", "transformers", "tqdm", "scipy"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)