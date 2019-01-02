#!/usr/bin/env python
#__builtins__.__openravepy_version__ = '0.9'
import os
from openravepy import *
from time import sleep
import matplotlib.pyplot as plt
import numpy as np
from SelectScript.SelectScript import SelectScript
from SelectScript_OpenRAVE.interpreter import interpreter

class interpeter_extended(interpreter):
    def __init__(self):
        interpreter.__init__(self)
                
    def evalAs(self, AS, SELECT, FROM, FROM_n, WHERE):
        if AS == "environment":
            return self.environment(AS, SELECT, FROM, FROM_n, WHERE)
        elif AS == "roombaGrid":
            return self.roombaGrid(AS, SELECT, FROM, FROM_n, WHERE)
        else:
            return interpreter.evalAs(self, AS, SELECT, FROM, FROM_n, WHERE)
        
    def roombaGrid(self, AS, SELECT, FROM, FROM_n, WHERE):
        # only for visualizing:
        env = FROM.next()[0].GetEnv()
        
        if WHERE == []:
            newEnv = FROM.next()[0].GetEnv()
        else:
            newEnv = self.environment(AS, SELECT, FROM, FROM_n, WHERE)
        envmin = []
        envmax = []
        for b in newEnv.GetBodies():
            ab = b.ComputeAABB()
            envmin.append(ab.pos()-ab.extents())
            envmax.append(ab.pos()+ab.extents())
            
        envmin = np.floor(np.min(np.array(envmin), 0)*100.) / 100
        envmax = np.ceil( np.max(np.array(envmax), 0)*100.) / 100 
        size   = np.ceil((envmax - envmin) / 0.025)
                
        ogMap = RaveCreateModule(env,'occupancygridmap')
        ogMap.SendCommand('SetTranslation %f %f 0.22' % (envmin[0], envmin[1]))
        ogMap.SendCommand('SetSize %i %i 0.025' %(size[0]+1, size[1]+1) )
        ogMap.SendCommand('SetLineWidth 2.0')
    
        render = ogMap.SendCommand('Scan')
        
        ogMap.SendCommand('Render')
        sleep(10)
        
        return np.fromstring(render, dtype=bool, count=-1, sep='').reshape(int(size[0]+1), int(size[1]+1))
        
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
    RaveInitialize()
    
    defaultPath = "/home/andre/Workspace/Projects/ROS/eos-openrave-plugins/filter/lib/filter"
    
    filterPath = raw_input("please enter the path to the OpenRAVE-filter-plugin default (" + defaultPath + "):\n")
    
    if filterPath=="":
        RaveLoadPlugin(defaultPath)
    else:
        RaveLoadPlugin(filterPath)
    
    kitchen_env =os.path.dirname(__file__)+"/resources/kitchen.env.xml"
    env = Environment()     # create the environment
    env.SetViewer('qtcoin') # start the viewer
    env.Load(kitchen_env)   # load a model

    ss = SelectScript(None, None)
    ssRave = interpeter_extended()
    ssRave.addVariable('kitchen', env)
        
    ## Example 1: Generate a map from the enitre environment 
    expr = "SELECT obj FROM kitchen AS roombaGrid;"
    print "query:", expr
    prog = ss.compile(expr)
    map = ssRave.eval(prog)    
    
    # plot the map
    plt.imshow(map.astype(numpy.float), cmap=plt.cm.gray)
    plt.show()    
    
    ## Example 2: Generate a map from all objects that are close to roomba 
    expr  = "roomba = SELECT obj FROM kitchen WHERE id(this)=='roomba_625x' AS value;"
    expr += "SELECT obj FROM kitchen WHERE distance(roomba, this) <= 1.5 AS roombaGrid;"
    print "query:", expr
    prog = ss.compile(expr)
    map = ssRave.eval(prog)    
    
    plt.imshow(map.astype(numpy.float), cmap=plt.cm.gray)
    plt.show()
    
if __name__ == "__main__":
    start()
    
    raw_input("\nPress Enter to exit ...")