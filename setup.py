from setuptools import setup, find_packages

setup(
    name="forest_fire_detection",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Forest Fire Detection using ML and IoT Sensors - BCA Final Year Project",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.12.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "opencv-python>=4.7.0",
        "pyserial>=3.5",
        "pyyaml>=6.0",
        "joblib>=1.2.0"
    ]
)