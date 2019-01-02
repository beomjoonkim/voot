#!/usr/bin/env python
#__builtins__.__openravepy_version__ = '0.9'
import os
from openravepy import *
from time import sleep
import numpy as np
from SelectScript.SelectScript import SelectScript
from SelectScript_OpenRAVE.interpreter import interpreter

from pyswip import Prolog

class interpeter_extended(interpreter):
    def __init__(self):
        interpreter.__init__(self)
                
    def evalAs(self, AS, SELECT, FROM, FROM_n, WHERE):
        if AS == "prolog":
            return self.prolog(AS, SELECT, FROM, FROM_n, WHERE)
        return interpreter.evalAs(self, AS, SELECT, FROM, FROM_n, WHERE)
        
    def prolog(self, AS, SELECT, FROM, FROM_n, WHERE):
        results = set()
        mark = True 
        for elem in FROM:
            if WHERE != []:
                mark = self.eval(WHERE, elem, FROM_n)
            if mark:
                for f in SELECT:
                    if f[1] == 'to' :
                        result, relation = self.eval(f, elem, FROM_n)
                        pp = f[2][0][2]
                    else:
                        relation = f[1]
                        result = self.eval(f, elem, FROM_n)
                        pp = f[2]
                        
                    if result != False:
                        clause = relation + "("                    
                        for p in pp:
                            p = self.eval(p, elem, FROM_n)
                            if isinstance(p, (openravepy_int.Robot, openravepy_int.Sensor, openravepy_int.KinBody)):
                                p = p.GetName()
                            clause += str(p) + ","                
                        clause = clause[:-1] + ")"
             
                        if result != True:
                            clause = clause[:-1] + ","+str(result)+")"                   
                        
                        results.add(clause)
  
        return sorted(results)

def start():
    kitchen_env =os.path.dirname(__file__)+"/resources/kitchen.env.xml"
    env = Environment()     # create the environment
    env.SetViewer('qtcoin') # start the viewer
    env.Load(kitchen_env)   # load a model

    ss = SelectScript(None, None)
    ssRave = interpeter_extended()
    ssRave.addVariable('kitchen', env)
        
    ## Example 1: Generate a map from the enitre environment 
    expr = "SELECT  above(a.this, b.this), \
                    below(a.this, b.this), \
                    isEnabled(a.this),     \
                    volume(a.this),        \
                    position(a.this),      \
                    isRobot(a.this),       \
                    isKinbody(a.this)      \
            FROM a=kitchen, b=kitchen      \
            WHERE not (isSensor(a.this) or isSensor(b.this)) \
            AS prolog;"
            
    print "query:", expr
    prog = ss.compile(expr)
    prolog = Prolog()
    for result in ssRave.eval(prog):
        print result
        prolog.assertz(result)
        
    print list(prolog.query("above(table, X)"))
    print list(prolog.query("volume(Obj, V), V > 1.0"))
    print list(prolog.query("position(_, P)"))
    
    
if __name__ == "__main__":
    start()
    
    raw_input("\nPress Enter to exit ...")