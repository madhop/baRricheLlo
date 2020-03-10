def get_action(track_pos):
    tol = 0.05
    left = track_pos > 0
    if abs(track_pos) >= tol:
        if left:
            action = [-0.005, 0, 0.2, 7]
        else:
            action = [0.005, 0, 0.2, 7]
    else:
        action = [0, 0, 0.2, 7]

    return action
