# GetContacts_analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This library is primarily meant to ease the analysis and visualisation of 
GetContacts frequency files. 

## Installation
### Building with pip

Please ensure that you have a python version higher than 3.6 for this package to
work properly. First change to the directory containing the cloned repository. The
package can then be installed via

```bash
pip install .
```

The dependencies can be installed with

```bash
pip install -r requirements.txt
```

### Building with setup.py

Be aware that you are responsible for the dependencies, if you decide to build 
this way.

```bash
python setup.py build
python setup.py install
```

### Import

After this, the package is callable by using

```python
import gc_analysis
```

## Usage

For usage examples, please check the provided Jupyter Notebook.

<!---
## License Info

When using the code in a publication, please cite:
[Johannes Kraml, Florian Hofer, Patrick K. Quoika, Anna S. Kamenik, and Klaus R. Liedl
Journal of Chemical Information and Modeling 2021 61 (4), 1533-1538 
doi: 10.1021/acs.jcim.0c01375 ](https://pubs.acs.org/doi/10.1021/acs.jcim.0c01375)
--->
