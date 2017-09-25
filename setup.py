from distutils.core import setup

setup(name='NetComp',
      version='0.1',
      description='Network Comparison',
      license='MIT',
      author='Peter Wills',
      author_email='peter.e.wills@gmail.com',
      packages=[
          'netcomp'
      ],
      install_requires=[
          'numpy>=1.11.3',
          'scipy>=0.18',
          'networkx==1'
      ]
     )
