import os
#from setuptools import setup
from distutils.core import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "SelectScript_OpenRAVE",
    version = "1",
    author = "Andr"+unichr(233)+" Dietrich",
    author_email = "dietrich@ivs.cs.uni-magdeburg.de",
    description = ("Implementation of the query-language SelectScript for OpenRAVE."),
    license = "BSD",
    keywords = "openrave, query, language, SelectScript",
    url = "https://pypi.python.org/pypi/SelectScript_OpenRAVE/1",
    packages=['SelectScript_OpenRAVE', 'SelectScript_OpenRAVE.examples'],
    long_description=read('README'),
    install_requires=['antlr_python_runtime','SelectScript'],
    include_package_data=True,
    package_data={ "SelectScript_OpenRAVE.examples" : 
                  ['resources/can.wrl',
                   'resources/chair.wrl',
                   'resources/coffe_machine.wrl',
                   'resources/_CorrogateShiny_.jpg',
                   'resources/cup.wrl',
                   'resources/irobot.png',
                   'resources/kitchen.env.xml',
                   'resources/kitchen.wrl',
                   'resources/knife_block.wrl',
                   'resources/Metal_Corrogated_Shiny.jpg',
                   'resources/microwave.wrl',
                   'resources/mixer.wrl',
                   'resources/pin.wrl',
                   'resources/plate.wrl',
                   'resources/roomba625.wrl',
                   'resources/shaker.wrl',
                   'resources/Stone_Vein_Gray.jpg',
                   'resources/table.wrl',
                   'resources/toaster.wrl',
                   'resources/_Translucent_Glass_Safety_1.jpg',
                   'resources/Translucent_Glass_Sky_Reflection.jpg',
                   'resources/Wood_Floor_Light.jpg',
                   'resources/Wood_Floor_Parquet.jpg']},
    #dependency_links = [],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)
