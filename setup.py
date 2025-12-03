# ============================================
# Package Setup
# ============================================

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="gender-age-classifier",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Multi-modal Gender and Age Classification System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gender-age-classifier",
    packages=find_packages(exclude=["tests", "notebooks", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "mypy>=1.7.1",
        ],
        "gpu": [
            "onnxruntime-gpu>=1.16.3",
        ],
    },
    entry_points={
        "console_scripts": [
            "gender-age-train=training.vision.train:main",
            "gender-age-infer=inference.api.main:main",
            "gender-age-collect=data_collection.main:main",
        ],
    },
)

