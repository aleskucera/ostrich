import warp as wp


@wp.func
def track_get_frame(
    u_in: wp.float32,
    r1: wp.float32,
    r2: wp.float32,
    dist: wp.float32
):
    # Re-compute geometry on the fly (stateless)
    # This avoids storing extra state per joint
    
    # Centers
    c1 = wp.vec2(0.0, 0.0)
    c2 = wp.vec2(dist, 0.0)
    
    # Derived geometry
    # D_vec = c2 - c1 (length is dist)
    dir_vec = wp.vec2(1.0, 0.0)
    
    # Beta
    val = (r1 - r2) / dist
    # Clip val to [-1, 1]
    if val > 1.0: val = 1.0
    if val < -1.0: val = -1.0
    beta = wp.acos(val)
    
    sin_beta = wp.sin(beta)
    cos_beta = wp.cos(beta)
    
    # Rotate function inline
    # n_top = rotate(dir_vec, beta)
    n_top = wp.vec2(cos_beta, sin_beta)
    
    # n_bot = rotate(dir_vec, -beta)
    n_bot = wp.vec2(cos_beta, -sin_beta)
    
    p1_top = c1 + n_top * r1
    p2_top = c2 + n_top * r2
    
    p1_bot = c1 + n_bot * r1
    p2_bot = c2 + n_bot * r2
    
    len_top = wp.length(p2_top - p1_top)
    len_bot = wp.length(p1_bot - p2_bot)
    
    # Angles
    ang_f_start = wp.atan2(n_top[1], n_top[0])
    ang_f_end = wp.atan2(n_bot[1], n_bot[0])
    
    two_pi = 2.0 * wp.PI
    
    sweep_front = (ang_f_start - ang_f_end) % two_pi
    # Warp modulo behavior can be tricky with negative numbers, ensure positive result
    if sweep_front < 0.0: sweep_front += two_pi
        
    len_arc_front = sweep_front * r2
    
    ang_r_start = wp.atan2(n_bot[1], n_bot[0])
    ang_r_end = wp.atan2(n_top[1], n_top[0])
    
    sweep_rear = (ang_r_start - ang_r_end) % two_pi
    if sweep_rear < 0.0: sweep_rear += two_pi
        
    len_arc_rear = sweep_rear * r1
    
    # Accumulators
    L1 = len_top
    L2 = L1 + len_arc_front
    L3 = L2 + len_bot
    total_len = L3 + len_arc_rear
    
    # Modulo u
    u = u_in % total_len
    if u < 0.0: u += total_len

    pos = wp.vec2(0.0)
    tan = wp.vec2(0.0)
    norm = wp.vec2(0.0)
    
    # 1. Top Line
    if u < L1:
        t = u / len_top
        pos = (1.0 - t) * p1_top + t * p2_top
        tan_dir = (p2_top - p1_top) / len_top
        tan = tan_dir
        norm = wp.vec2(-tan[1], tan[0])
        
    # 2. Front Arc
    elif u < L2:
        arc_u = u - L1
        angle = ang_f_start - (arc_u / r2)
        pos = c2 + wp.vec2(wp.cos(angle), wp.sin(angle)) * r2
        tan = wp.vec2(wp.sin(angle), -wp.cos(angle))
        norm = wp.vec2(wp.cos(angle), wp.sin(angle))
        
    # 3. Bottom Line
    elif u < L3:
        line_u = u - L2
        t = line_u / len_bot
        pos = (1.0 - t) * p2_bot + t * p1_bot
        tan_dir = (p1_bot - p2_bot) / len_bot
        tan = tan_dir
        norm = wp.vec2(-tan[1], tan[0])
        
    # 4. Rear Arc
    else:
        arc_u = u - L3
        angle = ang_r_start - (arc_u / r1)
        pos = c1 + wp.vec2(wp.cos(angle), wp.sin(angle)) * r1
        tan = wp.vec2(wp.sin(angle), -wp.cos(angle))
        norm = wp.vec2(wp.cos(angle), wp.sin(angle))
        
    return pos, tan, norm, total_len


@wp.func
def track_project(
    p: wp.vec2,
    r1: wp.float32,
    r2: wp.float32,
    dist: wp.float32
):
    # Re-compute geometry (stateless)
    c1 = wp.vec2(0.0, 0.0)
    c2 = wp.vec2(dist, 0.0)
    
    val = (r1 - r2) / dist
    if val > 1.0: val = 1.0
    if val < -1.0: val = -1.0
    beta = wp.acos(val)
    
    sin_beta = wp.sin(beta)
    cos_beta = wp.cos(beta)
    
    n_top = wp.vec2(cos_beta, sin_beta)
    n_bot = wp.vec2(cos_beta, -sin_beta)
    
    p1_top = c1 + n_top * r1
    p2_top = c2 + n_top * r2
    
    p1_bot = c1 + n_bot * r1
    p2_bot = c2 + n_bot * r2
    
    len_top = wp.length(p2_top - p1_top)
    len_bot = wp.length(p1_bot - p2_bot)
    
    ang_f_start = wp.atan2(n_top[1], n_top[0])
    ang_f_end = wp.atan2(n_bot[1], n_bot[0])
    two_pi = 2.0 * wp.PI
    
    sweep_front = (ang_f_start - ang_f_end) % two_pi
    if sweep_front < 0.0: sweep_front += two_pi
        
    len_arc_front = sweep_front * r2
    
    ang_r_start = wp.atan2(n_bot[1], n_bot[0])
    ang_r_end = wp.atan2(n_top[1], n_top[0])
    sweep_rear = (ang_r_start - ang_r_end) % two_pi
    if sweep_rear < 0.0: sweep_rear += two_pi
        
    len_arc_rear = sweep_rear * r1
    
    L1 = len_top
    L2 = L1 + len_arc_front
    L3 = L2 + len_bot
    
    # --- Projection Candidates ---
    
    # 1. Top Line
    v_top = p - p1_top
    line_vec_top = p2_top - p1_top
    t_top = wp.dot(v_top, line_vec_top) / (len_top * len_top)
    if t_top < 0.0: t_top = 0.0
    if t_top > 1.0: t_top = 1.0
    closest_top = p1_top + t_top * line_vec_top
    d_sq_top = wp.length_sq(p - closest_top)
    u_top = t_top * len_top

    # 2. Front Arc
    v_f = p - c2
    ang_p_f = wp.atan2(v_f[1], v_f[0])
    diff_f = (ang_f_start - ang_p_f) % two_pi
    if diff_f < 0.0: diff_f += two_pi
    
    if diff_f > sweep_front:
        # Closer to start (0) or end (sweep)?
        d_start = diff_f # Distance to 0 from right
        d_end = two_pi - diff_f # Distance to 0 from left (wrap) ?? No.
        
        # logic: diff_f is angle FROM start CW.
        # if diff_f > sweep, it's outside.
        # dist to start is diff_f (wrapped?) no.
        # simpler: clamp.
        
        dist_to_start = diff_f
        dist_to_end = two_pi - diff_f 
        # Actually comparing to sweep limits
        # Since it's circular, check angular distance
        
        # Re-eval:
        if (diff_f - sweep_front) < (two_pi - diff_f):
             diff_f = sweep_front
        else:
             diff_f = 0.0
             
    ang_clamped_f = ang_f_start - diff_f
    closest_f = c2 + wp.vec2(wp.cos(ang_clamped_f), wp.sin(ang_clamped_f)) * r2
    d_sq_f = wp.length_sq(p - closest_f)
    u_f = L1 + diff_f * r2
    
    # 3. Bottom Line
    v_bot = p - p2_bot
    line_vec_bot = p1_bot - p2_bot
    t_bot = wp.dot(v_bot, line_vec_bot) / (len_bot * len_bot)
    if t_bot < 0.0: t_bot = 0.0
    if t_bot > 1.0: t_bot = 1.0
    closest_bot = p2_bot + t_bot * line_vec_bot
    d_sq_bot = wp.length_sq(p - closest_bot)
    u_bot = L2 + t_bot * len_bot
    
    # 4. Rear Arc
    v_r = p - c1
    ang_p_r = wp.atan2(v_r[1], v_r[0])
    diff_r = (ang_r_start - ang_p_r) % two_pi
    if diff_r < 0.0: diff_r += two_pi
    
    if diff_r > sweep_rear:
        if (diff_r - sweep_rear) < (two_pi - diff_r):
             diff_r = sweep_rear
        else:
             diff_r = 0.0

    ang_clamped_r = ang_r_start - diff_r
    closest_r = c1 + wp.vec2(wp.cos(ang_clamped_r), wp.sin(ang_clamped_r)) * r1
    d_sq_r = wp.length_sq(p - closest_r)
    u_r = L3 + diff_r * r1
    
    # Select Best
    best_u = u_top
    min_d = d_sq_top
    
    if d_sq_f < min_d:
        min_d = d_sq_f
        best_u = u_f
        
    if d_sq_bot < min_d:
        min_d = d_sq_bot
        best_u = u_bot
        
    if d_sq_r < min_d:
        min_d = d_sq_r
        best_u = u_r
        
    return best_u
