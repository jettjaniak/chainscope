from setuptools import find_packages, setup

setup(
    name="chainscope",
    packages=find_packages(where="."),
    package_dir={"": "."},
    package_data={
        "chainscope": ["data/**/*"],
    },
    include_package_data=True,
)
