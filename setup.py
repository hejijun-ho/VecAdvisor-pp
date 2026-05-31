from setuptools import setup, find_packages

setup(
    name="vecadvisor",
    version="0.1.0",
    description="Filter-Aware Vector Index Selection and Tuning for PostgreSQL",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
)
