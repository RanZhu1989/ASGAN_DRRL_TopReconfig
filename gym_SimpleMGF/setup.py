from setuptools import setup

# stablebaseline3 is optional
setup(
    name="gym_SimpleMGF",
    version="0.0.1",
    install_requires=['pandas',
                      'numpy',
                      'tqdm',
                      'netcdf4',
                      'xarray',
                      'numba',
                      'PyYAML',
                      'pandapower',
                      'torch>=2.0.1',
                      'gymnasium==0.28.1',
                      'matplotlib',
                      'openpyxl'                  
                        ],
)