from distutils.core import setup


setup(
    name='buildtools',
    version='0.11',
    packages=['buildtools',],
    license='MIT',
    long_description=open('README.md').read(),
    url='https://github.com/thomaspingel/buildtools',

    author='Thomas Pingel',
    author_email='thomas.pingel@gmail.com',

    classifiers=['Development Status :: 4 - Beta',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.4',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 'Programming Language :: Python :: 3.9',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Geographic Information Science',
                 'Intended Audience :: Science/Research',
                 'Operating System :: OS Independent',
                 'License :: OSI Approved :: MIT License'],
    keywords='GIS lidar',
	install_requires=['scipy','pandas','rasterio','numpy'],
    )
