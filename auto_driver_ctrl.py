def get_action(track_pos):
    tol = 0.1
    left = track_pos > 0
    if abs(track_pos) >= tol:
        if left:
            action = [-0.0031, 0, 1., 7]
        else:
            action = [0.0002, 0, 1., 7]
    else:
        action = [0., 0, 1., 7]

    return action
