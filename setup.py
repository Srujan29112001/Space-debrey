"""
Space Debris Tracking & Autonomous Collision Prediction System
Setup configuration for package installation
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="space-debris-tracker",
    version="1.0.0",
    author="Space Debris Tracking Team",
    author_email="team@spacedebris.ai",
    description="AI-Powered Space Debris Tracking & Autonomous Collision Prediction System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/spacedebris/tracker",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "mypy>=1.7.1",
        ],
        "gpu": [
            "cupy-cuda11x>=12.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "space-tracker=space_debris_tracker.cli:main",
            "space-api=space_debris_tracker.api.server:run",
            "space-dashboard=space_debris_tracker.dashboard.app:main",
        ],
    },
)
