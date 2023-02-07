
def determine_multilr_milestones(epochs: int, steps: int):
    middle = epochs // 2
    step_size = middle // steps

    milestones = []
    for i in range(steps):
        milestones.append(middle + i*step_size)

    return milestones
