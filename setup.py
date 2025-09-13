from setuptools import setup, find_packages

setup(
    name="llm_server",
    version="0.1.0",
    description="A server for managing and deploying large language models.",
    author="Andreas Rasmusson",
    packages=find_packages(),
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
