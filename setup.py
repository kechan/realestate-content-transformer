from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
  long_description = fh.read()

setup(
  name="realestate_content_transformer",
  version="1.0.0",
  author="Kelvin Chan",
  author_email="kechan.ca@gmail.com",
  description="Real Estate Content Transformer",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/kechan/realestate-content-transformer",
  packages=find_packages(),
)
