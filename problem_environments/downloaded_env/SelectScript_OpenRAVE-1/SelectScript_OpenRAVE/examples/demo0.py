#!/usr/bin/env python
#__builtins__.__openravepy_version__ = '0.9'
import os
from openravepy import *
from SelectScript.SelectScript import SelectScript
from SelectScript_OpenRAVE.interpreter import interpreter

import time, thread

def switch(sensors): 
    while True:
        for sensor in sensors:
            time.sleep(3.33)
            for s in sensors:
                s.Configure(Sensor.ConfigureCommand.PowerOff)
                s.Configure(Sensor.ConfigureCommand.RenderDataOff)
            
            sensor.Configure(Sensor.ConfigureCommand.PowerOn)
            sensor.Configure(Sensor.ConfigureCommand.RenderDataOn)                

def on(o1, o2):
    ab  = o1.ComputeAABB()
    pos = o2.GetTransform()[:3,3]
    
    if ab.pos()[0] - ab.extents()[0] <= pos[0] <= ab.pos()[0] + ab.extents()[0]:
        if ab.pos()[1] - ab.extents()[1] <= pos[1] <= ab.pos()[1] + ab.extents()[1]:
            if pos[2] >= ab.pos()[2] + ab.extents()[2]:
                return True
    
    return False

def start():
    kitchen_env = os.path.dirname(__file__)+"/resources/kitchen.env.xml"
    env = Environment()     # create the environment
    env.Load(kitchen_env)   # load a model
    env.SetViewer('qtcoin') # start the viewer

    ss = SelectScript(None, None)
    ssRave = interpreter()
    # add the environment as variable to the interpreter
    ssRave.addVariable('kitchen', env)
    # ssRave.addVariable('durability', [5, "Feb", 2014])
    thread.start_new_thread( switch, (env.GetSensors(),) )
    
    ## Example 1: Query for information about sensors and robots 
    expr = "SELECT id, to(position(this),'pos'), type FROM kitchen WHERE isSensor(this) or isRobot(this) AS dict;"
    print "query 1:", expr, "\nresults:"
    prog = ss.compile(expr)
    for result in ssRave.eval(prog):
        print result

    ## Example 2: Query for the object with id table and store it in a variable
    expr = "table = SELECT obj FROM kitchen WHERE id(this) == 'table' as value;"
    print "\nquery 2:", expr, "\nresult:",
    prog = ss.compile(expr)
    print ssRave.eval(prog)

    ## Example 3: Query for all objects that are on the table and store them in a list
    ssRave.addFunction('on', on)
    expr = "objects_on_table = SELECT obj FROM kitchen WHERE on(table, this) AS list;"
    print "\nquery 3:", expr, "\nresults:"
    prog = ss.compile(expr)
    for result in ssRave.eval(prog):
        print result
    
    ## Example 4: Query for all objects not on the table
    expr = "SELECT obj FROM kitchen WHERE not on(table, this) AS list;"
    print "\nquery 4:", expr, "\nresults:"
    prog = ss.compile(expr)
    for result in ssRave.eval(prog):
        print result
    
if __name__ == "__main__":
    start()
    
    raw_input("\nPress Enter to exit ...")