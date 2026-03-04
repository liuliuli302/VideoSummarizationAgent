from setuptools import setup, find_packages

setup(
    name="video_agent",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "numpy",
        "opencv-python",
        "pyyaml",
    ],
)
