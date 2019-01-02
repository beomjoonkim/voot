import operator
from openravepy import *
import numpy

def _in(obj_, list_):
    return obj_ in list_

def _isRobot(entity):
    if type(entity) == openravepy_int.Robot:
        return True
    return False

def _isSensor(entity):
    if type(entity) == openravepy_int.Sensor:
        return True
    return False

def _isKinbody(entity):
    if type(entity) == openravepy_int.KinBody:
        return True
    return False

def _robot(entity):
    if type(entity) == openravepy_int.Robot:
        return entity

def _sensor(entity):
    if type(entity) == openravepy_int.Sensor:
        return entity

def _kinbody(entity):
    if type(entity) == openravepy_int.KinBody:
        return entity

def _type(entity):
    return type(entity)

def _position(entity, b=0, e=3):
    return list((poseFromMatrix(entity.GetTransform())[4:])[b:e])

def _positionX(entity):
    return _position(entity)[0]

def _positionY(entity):
    return _position(entity)[1]

def _positionZ(entity):
    return _position(entity)[2]

def _orientation(entity, b=0, e=4):
    return list((poseFromMatrix(entity.GetTransform())[:4])[b:e])

def _pose(entity, b=0, e=7):
    return list(poseFromMatrix(entity.GetTransform())[b:e])

def _isVisible(entity):
    return entity.IsVisible()

def _isEnabled(entity):
    return entity.IsEnabled()

def _identifer(entity):
    return entity.GetName()

def _object(entity):
    return entity

def _distance(entity1, entity2):
    p1 = _position(entity1)
    p2 = _position(entity2)
    
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)**(0.5)

def _isSensing(sensor, entity):
    if not _isSensor(sensor) or _isSensor(entity):
        return False
    
    envID = entity.GetEnvironmentId()
    res = sensor.SendCommand('collidingbodies')
    return envID in numpy.fromstring(res, dtype=int, sep=' ')

def _getEnvironmentID(entity):
    return entity.GetEnvironmentId()

def _sensingAmount(sensor, entity):
    if not _isSensor(sensor):
        return False
    envID = entity.GetEnvironmentId()
    res = sensor.SendCommand('collidingbodies')
    res = numpy.fromstring(res, dtype=int, sep=' ')
    res = res.tolist()
    
    return  float(res.count(envID)) / float(len(res))

def _sensingEnvironmentIDs(sensor):
    if not _isSensor(sensor):
        return None
    res = sensor.SendCommand('collidingbodies')
    res = list(set(res.tolist()))
    return res

def _volumeAABB(entity):
    if not _isKinbody(entity):
        return 0
    ab = entity.ComputeAABB()
    
    return ab.extents()[0] * ab.extents()[1] * ab.extents()[2] 

def _above(o1, o2):
    if o1 == o2:
        return False
    
    ab  = o1.ComputeAABB()
    pos = o2.GetTransform()[:3,3]
    
    if ab.pos()[0] - ab.extents()[0] <= pos[0] <= ab.pos()[0] + ab.extents()[0]:
        if ab.pos()[1] - ab.extents()[1] <= pos[1] <= ab.pos()[1] + ab.extents()[1]:
            if pos[2] >= ab.pos()[2] + ab.extents()[2]:
                return True
    
    return False

def _below(o1, o2):
    if o1 == o2:
        return False
    
    ab  = o1.ComputeAABB()
    pos = o2.GetTransform()[:3,3]
    
    if ab.pos()[0] - ab.extents()[0] <= pos[0] <= ab.pos()[0] + ab.extents()[0]:
        if ab.pos()[1] - ab.extents()[1] <= pos[1] <= ab.pos()[1] + ab.extents()[1]:
            if pos[2] < ab.pos()[2]:
                return True
    
    return False

def _within(o1, o2):
    if o1 == o2:
        return False
    
    ab  = o1.ComputeAABB()
    pos = o2.GetTransform()[:3,3]
        
    if ab.pos()[0] - ab.extents()[0] < pos[0] < ab.pos()[0] + ab.extents()[0]:
        if ab.pos()[1] - ab.extents()[1] < pos[1] < ab.pos()[1] + ab.extents()[1]:
            if ab.pos()[2] - ab.extents()[2] < pos[2] < ab.pos()[2] + ab.extents()[2]:
                return True
    
    return False
