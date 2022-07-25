from setuptools import setup, find_packages

VERSION = "0.1.0"
DESCRIPTION = "GetContacts-analysis"
LONG_DESCRIPTION = "github.com/liedllab/Getcontacts-analysis/gc_analysis/"

setup(
        name="gc_analysis",
        version=VERSION,
        author="Janik Kokot",
        author_email="janik@kokot.cc",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[
            "cython",
            "matplotlib",
            "numpy",
            "pandas",
            "seaborn",
            "mdtraj",
            ],

        keywords=["python", "GetContacts", "fingerprint", "flareplot"],
        classifiers= [
            "Intended Audience :: scientific",
            "Programming Language :: Python :: 3",
            "Operating System :: Linux :: Debian",
            ],
        )
