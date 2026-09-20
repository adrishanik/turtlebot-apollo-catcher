from setuptools import setup

package_name = 'apollo_pursuit'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Participant',
    maintainer_email='participant@techzephyr.com',
    description='Apollonius Predictive Pursuit Controller for TurtleBot 4 Lite',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'catcher_node = apollo_pursuit.apollo_catcher_node:main',
        ],
    },
)
