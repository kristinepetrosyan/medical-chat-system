from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="medical-chat-system",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered medical chat system for answering health-related queries",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/medical-chat-system",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/medical-chat-system/issues",
        "Source": "https://github.com/yourusername/medical-chat-system",
        "Documentation": "https://github.com/yourusername/medical-chat-system#readme",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Framework :: FastAPI",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.20.0",
        ],
        "gpu": ["faiss-gpu>=1.7.2"],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "medical-chat=scripts.run_chat:main",
            "medical-chat-train=scripts.train_model:main",
            "medical-chat-evaluate=scripts.evaluate_model:main",
        ],
    },
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
        "config": ["*.yaml", "*.yml"],
    },
    include_package_data=True,
    keywords="medical, chatbot, ai, healthcare, nlp, machine-learning",
)
