"""Python setup.py for agents package"""

import io
import os
from setuptools import find_packages, setup

from src.agents import __version__


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("project_name", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


setup(
    name="agents",
    version=__version__,
    description="Utility code for deploying LLM serving systems and running LLM agents",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="Noppanat Wadlom",
    install_requires=read_requirements("requirements.txt"),
    package_data={"agents": ["py.typed"]},
)
