c2raytools
==========

A simple Python module for reading and analyzing data from C2Ray and CubeP3M data files.

See the `example.py` file for some simple usage, `example_pv.py` for how to calculate peculiar velocity distortions and `example_freqbox.py` for how to make a time evolving box.

Installation
-------------
Find the file `setup.py` in the root directory. To install in the standard directory, run:
```
python setup.py install
```
If you do not have write permissions, or you want to install somewhere else, you can specify some other installation directory, for example:
```
python setup.py install --home=~/mydir
```
To see more options, run
```
python setup.py --help-commands
```
Or look [here](http://docs.python.org/2/install/) for more details.
