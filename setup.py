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
]
,
    entry_points={
        "console_scripts": [
            "start_llm_server = llm_server.server:main",
        ]
    },
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
    ],
)
