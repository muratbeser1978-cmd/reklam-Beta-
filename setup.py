from setuptools import setup, find_packages

setup(
    name="rkl",
    version="0.1.0",
    description="Consumer Learning and Market Dynamics Simulation",
    author="Research Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24.0,<2.0.0",
        "scipy>=1.11.0,<2.0.0",
        "matplotlib>=3.7.0,<4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "hypothesis>=6.82.0",
        ]
    },
)
