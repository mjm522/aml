import numpy as np
import pybullet as p

def rotx(ang):
    return np.array([ [1., 0., 0.], [0., np.cos(ang), -np.sin(ang)], [0., np.sin(ang), np.cos(ang)] ])

def roty(ang):
    return np.array([ [np.cos(ang), 0., np.sin(ang)], [0., 1., 0.], [-np.sin(ang), 0. , np.cos(ang)] ]) 

def rotz(ang):
    return np.array([ [np.cos(ang), -np.sin(ang), 0.], [np.sin(ang), np.cos(ang), 0.], [0., 0., 1.] ])


def compute_f_cone_approx(contact_point_obj_frame, obj_rot_matrix=None, mu=0.6):
    """
    this computes a fixed cone pointed in the 
    z direction. We will transform it into different directions
    this assumes you only contact the lateral surface of the object
    this is a healthy assumption since the object is fixed
    """
    s_n =  np.array([1., 0., 0.])

    cone_angle = np.arctan(mu)

    rotation_to_z = np.arctan2(contact_point_obj_frame[1], contact_point_obj_frame[0]) + np.pi

    if obj_rot_matrix is None:
        obj_rot_matrix = np.eye(3)

    #now we rotate the s_n about z by same quantiy

    b1 = np.dot(obj_rot_matrix, np.dot(rotz(rotation_to_z), np.dot(rotz(cone_angle),  s_n)) )
    b2 = np.dot(obj_rot_matrix, np.dot(rotz(rotation_to_z),np.dot(roty(cone_angle),  s_n)) )
    b3 = np.dot(obj_rot_matrix, np.dot(rotz(rotation_to_z),np.dot(rotz(-cone_angle),  s_n)) )
    b4 = np.dot(obj_rot_matrix, np.dot(rotz(rotation_to_z),np.dot(roty(-cone_angle), s_n)) )

    return np.vstack([b1, b2, b3, b4]).T


#unit normal vector of plane defined by points a, b, and c
def unit_normal(a, b, c):
    x = np.linalg.det([[1,a[1],a[2]],
         [1,b[1],b[2]],
         [1,c[1],c[2]]])
    y = np.linalg.det([[a[0],1,a[2]],
         [b[0],1,b[2]],
         [c[0],1,c[2]]])
    z = np.linalg.det([[a[0],a[1],1],
         [b[0],b[1],1],
         [c[0],c[1],1]])
    magnitude = (x**2 + y**2 + z**2)**.5
    return (x/magnitude, y/magnitude, z/magnitude)

#area of polygon poly
def poly_area(poly):
    if len(poly) < 3: # not a plane - no area
        return 0
    total = [0, 0, 0]
    N = len(poly)
    for i in range(N):
        vi1 = poly[i]
        vi2 = poly[(i+1) % N]
        prod = np.cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
    result = np.dot(total, unit_normal(poly[0], poly[1], poly[2]))
    return abs(result/2)



def ray_cylinder_intersection(ray_origin, ray_direction, radius=0.3, height=0.65):
    """
    ray_origin and ray_direction should be in object frame
    assumes a cylinder along z axis
    assumes ray equation = ray_origin + ray_direction*t
    """

    ray_hit_point = None
    ray_hit_normal = None

    # Compute quadratic cylinder coefficients
    a = ray_direction[0]*ray_direction[0] + ray_direction[1]*ray_direction[1]
    b = 2*(ray_direction[0]*ray_origin[0] + ray_direction[1]*ray_origin[1])
    c = ray_origin[0]*ray_origin[0] + ray_origin[1]*ray_origin[1] - radius*radius

    #Solve quadratic equation for _t_ values
    try:
        
        t0, t1 = np.roots([a,b,c])

    except Exception as e:

        print "Exception: \t", e
        return ray_hit_point, ray_hit_normal

    if t0 > t1: tmp = t0; t0=t1; t1=tmp;

    z0 = ray_origin[2] + t0*ray_direction[2]
    z1 = ray_origin[2] + t1*ray_direction[2]

    if (z0<-height/2):

        if z1 < -height/2: return ray_hit_point, ray_hit_normal
        
        else:

            #hit the cap
            th = t0 + (t1-t0) * (z0+1) / (z0-z1)

            if th <= 0: return ray_hit_point, ray_hit_normal

            ray_hit_point  = ray_origin + (ray_direction*th)
            ray_hit_normal = np.array([0, -1, 0])

            return ray_hit_point, ray_hit_normal

    elif (z0>=-height/2 and z0<=height/2):
        #hit the cylinder bit

        if t0 <= 0: return ray_hit_point, ray_hit_normal

        ray_hit_point =  ray_origin + (ray_direction*t1)
        ray_hit_normal = np.array([ray_hit_point[0], 0, ray_hit_point[2]])
        ray_hit_normal /= np.linalg.norm(ray_hit_normal)

    elif z0 > height/2:

        if z1 > height/2: return ray_hit_point, ray_hit_normal

        else:
            # hit the cap
            th = t0 + (t1-t0) * (z0-1) / (z0-z1)

            if th <= 0 : return ray_hit_point, ray_hit_normal

            ray_hit_point  = ray_origin + (ray_direction*th)
            ray_hit_normal = np.array([0, -1, 0])

            return ray_hit_point, ray_hit_normal

    return ray_hit_point, ray_hit_normal


