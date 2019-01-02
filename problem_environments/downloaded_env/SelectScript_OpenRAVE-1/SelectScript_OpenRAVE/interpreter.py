import operator, threading
import ssFunction
from openravepy import *
from SelectScript.SelectScript_Interpreter import *

class interpreter(SelectScript_Interpreter):
    def __init__(self):
        SelectScript_Interpreter.__init__(self)        
        self.initFunctions()        
        self.timer_list = {}
        
    def initFunctions(self):
        self.addFunction('id', ssFunction._identifer, "Returns the identifier of a sensor, robot, or kinbody.\nUsage: id(object) -> string")
        self.addFunction('type', ssFunction._type, "Returns the type of an object.\nUsage: type(object)")
        self.addFunction('position', ssFunction._position, "Returns a list of the x,y,z coordinates of an object.\nUsage: position(object, begin, end)\n       position(object) -> [x,y,z]\n       position(object, 0,2) -> [x,y]")
        self.addFunction('pose', ssFunction._pose, "Returns the pose of an object.\nUsage: pose(object) -> XXX")
        self.addFunction('distance', ssFunction._distance, "Calculates the euclidian distance of two objects.\nUsage: distance(object1, object2) -> float")
        
        self.addFunction('volume', ssFunction._volumeAABB, "Approximates the volume of an object by using AABB.\nUsage: volume(object) -> float")
        
        self.addFunction('isRobot', ssFunction._isRobot, "Checks if object is of type Robot.\nUsage: isRobot(object) -> bool")
        self.addFunction('isSensor', ssFunction._isSensor, "Checks if object is of type Sensor.\nUsage: isSensor(object) -> bool")
        self.addFunction('isKinbody', ssFunction._isKinbody, "Checks if object is of type Kinbody.\nUsage: isKinbody(object) -> bool")
        
        self.addFunction('robot', ssFunction._robot)
        self.addFunction('sensor', ssFunction._sensor)
        self.addFunction('kinbody', ssFunction._kinbody)
        
        self.addFunction('obj', ssFunction._object, "Returns the object itself.\nUsage: obj(object) -> object")
        
        self.addFunction('x', ssFunction._positionX, "Returns only the x position.\nUsage: x(object) -> float")
        self.addFunction('y', ssFunction._positionY, "Returns only the y position.\nUsage: y(object) -> float")
        self.addFunction('z', ssFunction._positionZ, "Returns only the z position.\nUsage: z(object) -> float")
        
        self.addFunction('isEnabled', ssFunction._isEnabled, "Checks if object is enabled.\nUsage: isEnabled(object) -> bool")
        self.addFunction('isVisible', ssFunction._isVisible, "Checks if object is visable.\nUsage: isVisable(object) -> bool")
        
        self.addFunction('isSensing', ssFunction._isSensing, "Checks if a sensor is currently sensing a certain object.\nUsage: isSensing(sensor, object) -> bool")
        self.addFunction('environmentID', ssFunction._getEnvironmentID, "Returns the OpenRAVE environment id of an object.\nUsage: environmentID(object) -> XXX") 
        
        self.addFunction('sensingAmount', ssFunction._sensingAmount, "Calculates how much of XXX")
        self.addFunction('sensingEnvIDs', ssFunction._sensingEnvironmentIDs)
        
        self.addFunction('above', ssFunction._above, "Checks if the position of an object (o2) is above object (o1), by using the axis-aligned bounding box of o1.\nUsage: above(o1, o2) -> bool")
        self.addFunction('below', ssFunction._below, "Checks if the position of an object (o2) is below object (o1), by using the axis-aligned bounding box of o1.\nUsage: below(o1, o2) -> bool")
        self.addFunction('within',ssFunction._within, "Checks if the position of an object (o2) is within object (o1), by using the axis-aligned bounding box of o1.\nUsage: within(o1, o2) -> bool")
        
        self.addFunction('getTime', self.getTime )
    
    def getListFrom(self, obj):
        if isinstance(obj, openravepy_int.Environment):
            return obj.GetSensors() + obj.GetBodies()
        else:
            return SelectScript_Interpreter.getListFrom(self, obj)
        
    def trigger(self, name, prog, t, callback):
        if self.timer_list.has_key(name):
            #prog, t, callback, old = self.timer_list[name]
            #print self.callVariable('env').GetSimulationTime() /200000.
            #print self.callVariable('env').IsSimulationRunning()

            old_result = self.timer_list[name]
        
            threading.Timer(t, self.trigger, [name, prog, t, callback]).start()
            
            result=self.eval(prog)
            
            if result != old_result:
                self.timer_list[name] = result
                callback(result) 
                   
    
    def addTrigger(self, name, prog, t, callback):
        self.timer_list[name] = None #[prog, t, callback, None]
        self.trigger(name, prog, t, callback)
        
    def delTrigger(self, name):
        self.timer_list.pop(name)
        
    def getTime(self):
        return self.callVariable('env').GetSimulationTime()/200000.
    