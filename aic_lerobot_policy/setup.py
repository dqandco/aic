from setuptools import find_packages, setup

package_name = "aic_lerobot_policy"

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
    description="Generic LeRobot-policy runtime for the AIC framework.",
    license="Apache-2.0",
)
