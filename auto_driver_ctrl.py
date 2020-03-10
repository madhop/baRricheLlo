def get_action(track_pos):
    tol = 0.02
    left = track_pos > 0
    if abs(track_pos) >= tol:
        if left:
            action = [-0.008, 0, 0.3, 7]
        else:
            action = [0.008, 0, 0., 7]
    else:
        action = [0, 0, 0.3, 7]

    return action
