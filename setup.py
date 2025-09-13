from setuptools import setup, find_packages

setup(
    name="llm_server",
    version="0.1.0",
    description="A server for managing and deploying large language models.",
    author="Andreas Rasmusson",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.111.0",
        "uvicorn>=0.30.0",
        "torch>=2.3.0",
        "transformers>=4.56.0",
        "azure-identity>=1.15.0",
        "azure-keyvault-secrets>=4.8.0",
        "pyyaml>=6.0",
        "bitsandbytes>=0.43.2",
        "accelerate>=1.0.0"
    ],
    extras_require={
    "cu121": [
        "torch==2.3.1+cu121",
        "torchvision==0.18.1+cu121",
        "torchaudio==2.3.1+cu121",
    ],
    entry_points={
        "console_scripts": [
            "start_llm_server = llm_server.api.server:main",
        ]
    },
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
    ],
)
