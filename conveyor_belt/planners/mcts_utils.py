def make_action_hashable(action):
    if len(action) == 2:
        return tuple((tuple(action[0]), tuple(action[1])))
    else:
        if isinstance(action, list):
            return tuple(action)
        else:
            array = action.squeeze()
            return tuple(array.tolist())


def is_action_hashable(action):
    if len(action) == 2:
        return isinstance(action[0], tuple) and isinstance(action[1], tuple)
    else:
        return isinstance(action, tuple)
