from setuptools import find_packages, setup

package_name = "expert_twist"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    description="Velocity-twist ground-truth expert for AIC cable-insertion data collection.",
    license="Apache-2.0",
)
