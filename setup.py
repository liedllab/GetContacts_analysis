from setuptools import setup, find_packages

VERSION = "0.0.2"
DESCRIPTION = "GetContacts-analysis"
LONG_DESCRIPTION = "github.com/liedllab/Getcontacts-analysis"

setup(
        name="gc_analysis",
        version=VERSION,
        author="Janik Kokot",
        author_email="janik@kokot.cc",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[
            "matplotlib",
            "mdtraj",
            "numpy",
            "pandas",
            "seaborn",
            ],

        keywords=["python", "GetContacts", "fingerprint", "flareplot"],
        classifiers= [
            "Intended Audience :: scientific",
            "Programming Language :: Python :: 3",
            "Operating System :: Linux :: Debian",
            ],
        )
