from setuptools import setup, find_packages

setup(
    name="tsm-realtime",
    version="0.1.3",
    description="Real-time audio time-scale modification with look-up approximation and full computation methods",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Sayema Lubis, Clark Peng, Jared Carreno",
    license="MIT",
    license_files=[],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering",
    ],
    keywords=["audio", "dsp", "tempo", "music", "real-time"],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pyaudio>=0.2.11",
        "librosa>=0.9.0",
        "pydub>=0.25.0",
    ],
    extras_require={
        "keyboard": ["pynput>=1.8.1"],
    },
    project_urls={
        "Homepage": "https://github.com/HMC-MIR/TSMRealTime",
        "Repository": "https://github.com/HMC-MIR/TSMRealTime",
        "Issues": "https://github.com/HMC-MIR/TSMRealTime/issues",
    },
)
