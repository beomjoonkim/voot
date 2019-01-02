#!/usr/bin/env python
#__builtins__.__openravepy_version__ = '0.9'
import os, time, sys
from openravepy import *
from time import sleep

from SelectScript.SelectScript import SelectScript
from SelectScript_OpenRAVE.interpreter import interpreter


class interpeter_extended(interpreter):
    def __init__(self):
        interpreter.__init__(self)
                
    def evalAs(self, AS, SELECT, FROM, FROM_n, WHERE):
        if AS == "environment":
            return self.environment(AS, SELECT, FROM, FROM_n, WHERE)
        else:
            return interpreter.evalAs(self, AS, SELECT, FROM, FROM_n, WHERE)
        
    def environment(self, AS, SELECT, FROM, FROM_n, WHERE):
        mark = True
        newEnv = None
        for elem in FROM:
            if not newEnv: newEnv = elem[0].GetEnv().CloneSelf(1|8)
                
            if WHERE != []:
                mark = self.eval(WHERE, elem, FROM_n)            
                
            if not mark:
                object = self.eval(SELECT, elem, FROM_n)
                     
                if type(object) == Robot:
                    newEnv.Remove(newEnv.GetRobot(object.GetName()))
                elif type(object) == KinBody:
                    newEnv.Remove(newEnv.GetKinBody(object.GetName()))
                        
        return newEnv
    
def start():
    kitchen_env =os.path.dirname(__file__)+"/resources/kitchen.env.xml"
    env = Environment()     # create the environment
    #env.SetViewer('qtcoin') # start the viewer
    env.Load(kitchen_env)   # load a model

    ss = SelectScript(None, None)
    ssRaveX = interpeter_extended()
    ssRaveX.addVariable('kitchen', env)
    
    ## Example 1: Query for information about sensors and robots 
    expr  = "roomba = SELECT obj FROM kitchen WHERE id(this)=='roomba_625x' AS value;"
    expr += "SELECT obj FROM kitchen WHERE distance(roomba, this) <= 1.5 AS environment;"
    print "query:", expr
    prog = ss.compile(expr)
    
    newEnv = ssRaveX.eval(prog)
    newEnv.SetViewer('qtcoin')   
    
if __name__ == "__main__":
    start()
    
    raw_input("\n>>> ")#("\nPress Enter to exit ...")