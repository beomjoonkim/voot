#!/usr/bin/env python
#__builtins__.__openravepy_version__ = '0.9'
import os
from openravepy import *
#from openravepy.examples.simplenavigation import *
from SelectScript.SelectScript import SelectScript
from SelectScript_OpenRAVE.interpreter import interpreter
import numpy
from thread import start_new_thread

class SimpleNavigationPlanning:
    def __init__(self,robot,randomize=False,dests=None,switchpatterns=None):
        self.env = robot.GetEnv()
        self.robot = robot
        self.cdmodel = databases.convexdecomposition.ConvexDecompositionModel(self.robot)
        if not self.cdmodel.load():
            self.cdmodel.autogenerate()
        self.basemanip = interfaces.BaseManipulation(self.robot)
    def performNavigationPlanning(self):
        
        with self.env:
            envmin = []
            envmax = []
            for b in self.env.GetBodies():
                ab = b.ComputeAABB()
                envmin.append(ab.pos()-ab.extents())
                envmax.append(ab.pos()+ab.extents())
            abrobot = self.robot.ComputeAABB()
            envmin = numpy.min(numpy.array(envmin),0)+abrobot.extents()
            envmax = numpy.max(numpy.array(envmax),0)-abrobot.extents()
         
        bounds = numpy.array([[-0.55, -2.51, -3.14],
                  [ 2.50,  1.34,  3.14]])
        
        while True:
            with self.env:
                self.robot.SetAffineTranslationLimits(envmin,envmax)
                self.robot.SetAffineTranslationMaxVels([0.5,0.5,0.5])
                self.robot.SetAffineRotationAxisMaxVels(numpy.ones(4))
                self.robot.SetActiveDOFs([],DOFAffine.X|DOFAffine.Y|DOFAffine.RotationAxis,[0,0,1])
                # pick a random position
                with self.robot:
                    while True:
                        goal = bounds[0,:]+numpy.random.rand(3)*(bounds[1,:]-bounds[0,:])
                        self.robot.SetActiveDOFValues(goal)
                        if not self.env.CheckCollision(self.robot):
                            break
            print 'planning to: ',goal
            # draw the marker
            center = numpy.r_[goal[0:2],0.2]
            xaxis = 0.5*numpy.array((numpy.cos(goal[2]),numpy.sin(goal[2]),0))
            yaxis = 0.25*numpy.array((-numpy.sin(goal[2]),numpy.cos(goal[2]),0))
            h = self.env.drawlinelist(numpy.transpose(numpy.c_[center-xaxis,center+xaxis,center-yaxis,center+yaxis,center+xaxis,center+0.5*xaxis+0.5*yaxis,center+xaxis,center+0.5*xaxis-0.5*yaxis]),linewidth=5.0,colors=numpy.array((0,1,0)))
            if self.basemanip.MoveActiveJoints(goal=goal,maxiter=3000,steplength=0.1) is None:
                print 'retrying...'
                continue
            print 'waiting for controller'
            self.robot.WaitForController(0)

def callback(msg):    
    simulation_time = msg[0]
    active_sensors  = msg[1]
    print "sim_time:", simulation_time
    
    global env    
    
    for sensor in env.GetSensors():
        sensor.Configure(Sensor.ConfigureCommand.RenderDataOff)
        
    for sensor in set(numpy.concatenate(active_sensors)):
        sensor.Configure(Sensor.ConfigureCommand.RenderDataOn)

def start():
    global env
    kitchen_env = os.path.dirname(__file__)+"/resources/kitchen.env.xml"
    env = Environment()     # create the environment
    env.SetViewer('qtcoin') # start the viewer
    env.Load(kitchen_env)   # load a model

    ss = SelectScript(None, None)
    ssRave = interpreter()
    ssRave.addVariable('env', env)
    ssRave.addVariable('kitchen', env)

    robot1 = env.GetRobot('barrett_4918')
    robot2 = env.GetRobot('roomba_625x')

    for sensor in env.GetSensors():
        sensor.Configure(Sensor.ConfigureCommand.PowerOn)
        #sensor.Configure(Sensor.ConfigureCommand.RenderDataOn)

    nav1 = SimpleNavigationPlanning(robot1)
    nav2 = SimpleNavigationPlanning(robot2)
    
    expr = "measure{5} = select obj(env.this) \
               from env=kitchen, env2=kitchen \
                 where isSensor(env.this) and \
                       isRobot(env2.this) and \
               isSensing(env.this, env2.this) \
               as list; \
               [getTime(), measure{-2}];"
    prog = ss.compile(expr)
    ssRave.addTrigger('test', prog, 0.5, callback)
    
    start_new_thread( nav1.performNavigationPlanning, () )
    start_new_thread( nav2.performNavigationPlanning, () )
        
if __name__ == "__main__":
    start()
    
    raw_input("\nPress Enter to exit ...\n")