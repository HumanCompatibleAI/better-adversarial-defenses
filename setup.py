#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="ap_rllib",
    version=0.1,
    description="Adversarial Policies for Reinforcement Learning with RLLib",
    author="Sergei Volodin, Adam Gleave, et al",
    author_email="sergei.volodin@epfl.ch",
    python_requires=">=3.7.0",
    url="https://github.com/HumanCompatibleAI/better-adversarial-defenses",
    packages=["ap_rllib", "gym_compete_rllib", "frankenstein", "ap_rllib_experiment_analysis"],
    package_dir={},
    package_data={},
    # We have some non-pip packages as requirements,
    # see requirements-build.txt and requirements.txt.
    install_requires=[],
    include_package_data=True,
    license="MIT",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
)
