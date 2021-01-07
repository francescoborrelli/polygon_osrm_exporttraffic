from setuptools import find_packages, setup
from pkg_resources import parse_requirements

name = 'ws_maps'
version = '0.0'
release = '0.0.0'
with open('README.rst') as readme_file:
    readme = readme_file.read()

# fixme this is not recommended (install requirements should be a minimal list)
requirements = []
with open('requirements.txt', 'r') as f:
    for r in parse_requirements(f.read()):
        if r.specs:
            specs = ','.join(''.join(list(s)) for s in r.specs)
            requirements.append(r.name + specs)
        else:
            requirements.append(r.name)

setup(
    name=name,
    packages=find_packages(),
    version=version,
    description='A short description of the project.',
    long_description=readme,
    author='Jacopo Guanetti',
    license='',
    # these are optional and override conf.py settings
    command_options={
        'build_sphinx': {
            'project': ('setup.py', name),
            'version': ('setup.py', version),
            'release': ('setup.py', release),
            'source_dir': ('setup.py', 'docs')}},
    install_requires=requirements
)
