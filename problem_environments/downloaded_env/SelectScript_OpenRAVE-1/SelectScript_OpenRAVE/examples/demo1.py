#!/usr/bin/env python
#__builtins__.__openravepy_version__ = '0.9'
import os
from openravepy import *
from openravepy.examples.graspplanning import *
from SelectScript.SelectScript import SelectScript
from SelectScript_OpenRAVE.interpreter import interpreter

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
    env.SetViewer('qtcoin') # start the viewer
    env.Load(kitchen_env)   # load a model

    ss = SelectScript(None, None)
    ssRave = interpreter()
    ssRave.addVariable('kitchen', env)
    
 #   sensor_ids = ['barrett_4918_laser_scaner', 'roomba_625x_webcam_a40f', 'roomba_625x_dist1', 'roomba_625x_dist2', 'roomba_625x_dist3', 'roomba_625x_dist4']
 #   for ID in sensor_ids:
 #       env.GetSensor(ID).Configure(Sensor.ConfigureCommand.PowerOn)
 #       env.GetSensor(ID).Configure(Sensor.ConfigureCommand.RenderDataOn)

    ## Example: 
    ssRave.addFunction('on', on)
    robot = env.GetRobot("barrett_4918")
    ssRave.addVariable('robot', robot)
    expr  = "sink = SELECT obj FROM kitchen WHERE id(this) == 'sink' AS value;"
    expr += "table = SELECT obj FROM kitchen WHERE id(this) == 'table' as value;"    
    expr += "cleanUp = SELECT obj FROM kitchen WHERE on(table, this) AND distance(robot, this) < 1.3 AS list;"
    
    prog = ss.compile(expr)
    print ssRave.eval(prog)
    
    ## start grasping
    grasp = GraspPlanning(robot)
    grasp.graspables = []
    for object in ssRave.callVariable('cleanUp'):
        destination = grasp.setRandomDestinations( [object], 
                                                   ssRave.callVariable('sink'), 
                                                   randomize=True, preserverotation=False)
        
        gModel = databases.grasping.GraspingModel(robot=robot, target=object) 
        if not gModel.load():
            gModel.autogenerate()
        destination = [x.tolist() for x in destination[0]]
        
        grasp.graspables.append([gModel,destination])

    grasp.performGraspPlanning(withreplacement=True)    
        
if __name__ == "__main__":
    start()
    
    raw_input("\nPress Enter to exit ...")