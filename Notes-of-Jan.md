# Notes scratchpad file

## calculate distances from phase changes

deltaPhi(t) = 2*pi*(f1-f2)*t
-> Phi(t2) - Phi(t1) = 2*pi*(f1-f2) * (t2-t1)
-> deltaPhi/deltaT = 2*pi*(f1-f2)
-> deltaF = (deltaPhi / deltaT) / 2*pi 




## some methods to make the access to radarconfig more handy:

### 1: create a dataclass from dict
a dataclass can extend on runtime
~~~
from dataclasses import make_dataclass
def dict2dataclass(classname:str, data:dict) -> object:
    '''
    creates a dataclassobject of class classname with data from dict
    :param classname:   name of the dataclass to create
    :param data:        data to fill the fields of the dataclass-object
    :return:    dataclass-object filled with values from dict
    '''
    return make_dataclass( classname, ((k, type(v)) for k, v in data.items()) )(**data)
~~~

### 2: create a SimpleNamespace-Object from dict
~~~
from types import SimpleNamespace
data = SimpleNamespace(**dict)
~~~

### 3: create a nametuple from dict
~~~
from collections import namedtuple
radarConfigClass = namedtuple("radarConfigClass", dict)
data = radarConfigClass(**dict)


all create an object witch direct access to the fields be the fieldnames

fStart = data.start_frequency_Hz
fEnd   = data.end_frequency_Hz
~~~