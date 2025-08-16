from setuptools import setup, find_packages

setup(
    name="seexsum-ai",
    version="1.0.0",
    description="LLM-assisted web search + synthesis with sources",
    packages=find_packages(exclude=["venv", "tests"]),
    install_requires=[
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "ddgs>=9.0.0",
        "crawl4ai>=0.1.0",
    ],
    entry_points={
        "console_scripts": [
            "seexsum-ai=seexsum_ai.cli:main",
        ]
    },
    python_requires=">=3.9",
)


