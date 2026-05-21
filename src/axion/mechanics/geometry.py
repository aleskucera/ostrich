import warp as wp


@wp.func
def orthogonal_basis(axis: wp.vec3):
    # Choose v as the unit vector along the axis with the smallest absolute component
    if wp.abs(axis.x) <= wp.abs(axis.y) and wp.abs(axis.x) <= wp.abs(axis.z):
        v = wp.vec3(1.0, 0.0, 0.0)
    elif wp.abs(axis.y) <= wp.abs(axis.z):
        v = wp.vec3(0.0, 1.0, 0.0)
    else:
        v = wp.vec3(0.0, 0.0, 1.0)

    # Compute b1 as the normalized cross product of axis and v
    b1 = wp.normalize(wp.cross(axis, v))

    # Compute b2 as the cross product of axis and b1
    b2 = wp.cross(axis, b1)

    return b1, b2