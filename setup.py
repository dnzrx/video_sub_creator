from setuptools import setup, find_packages

# Read the contents of the README file
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    version="1.0.0",
    name="video_sub_creator",
    packages=find_packages(),
    author="dnzrx",
    description="Automatically generate subtitle files (.srt and .vtt) for any video.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    install_requires=[],
    python_requires='>=3.9',
    entry_points={
        'console_scripts': ['video_sub_creator=video_sub_creator.main:main'],
    },
)
