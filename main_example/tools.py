import sys, numpy
import numpy as np


###########################################################################
class argsList:
    #A simple class to parse an arg list of the form param1=value1, etc., with 
    #spaces delineating arguments. e.g., execute:
    #python run_sim.py param1=0 param2=1000 param3=True
    #Help option specific to my compartment sims, but basic structure can be reused.

    def __init__(self, **kwargs):
        self.arg_dict= {}
        self.parseArgs(**kwargs)

    def parseArgs(self, pdict=None, hdict=None):
        """pdict is keys of param dict"""
        if "?" in sys.argv:
            print("""
                Command line format:
                   param1=value1 param2=value2 ...etc...
                """)
            if pdict is not None: 
                print("Options:")
                print([k for k in pdict])
            if hdict is not None:
                print("Explanation:")
                for  k,v in zip(hdict.keys(), hdict.values()):
                    print(str(k)+": "+str(v))
            exit()
        if len(sys.argv) > 1:
            for element in sys.argv[1:]:
                var_name=""
                var_value=""

                if "=" not in element: #assume end of commands
                    break

                for k, char in enumerate(element):
                    if k < element.index("="):
                        var_name= var_name+char
                    elif k > element.index("="):
                        var_value= var_value+char

                self.arg_dict[var_name]=var_value

