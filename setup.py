import os
import sys
from setuptools import setup, find_packages

setup(
    name="pandas_db",
    version=0.1,
    description="Description here",
    license="Apache 2.0",
    packages=find_packages(),
    package_data={},
    scripts=[],
    install_requires=["pandas", "plotly", "dash", "dash-extensions", "click"],
    extras_require={"test": ["pytest", "pylint!=2.5.0"],},
    entry_points={"console_scripts": [
        'ddsp_run = ddsp.training.ddsp_run:console_entry_point',
        'pandasdb = pandas_db.browser:main'
    ],},
    classifiers=[],
    tests_require=["pytest"],
    setup_requires=["pytest-runner"],
    keywords="",
)
