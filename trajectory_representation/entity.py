class Entity:
    def __init__(self, entity_type, pose, shape, name):
        self.pose = pose
        self.shape = shape
        self.type = entity_type
        self.name = name
