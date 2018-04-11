import numpy as np
from Box2D import b2ContactListener, b2GetPointStates, b2_addState

class ContactSensor(b2ContactListener):
    
    def __init__(self):
        b2ContactListener.__init__(self)
        self._normal_impulse_mag = 0.
        self._tangent_impulse_mag = 0.
        self._normal_impulse_dir = None

    def BeginContact(self, contact):
        pass
    def EndContact(self, contact):
        pass
    def PreSolve(self, contact, old_manifold, display=False):
        pass

    def PostSolve(self, contact, impulse):
        world_manifold = contact.worldManifold
        contact_manifold = contact.manifold

        local_normal = contact_manifold.localNormal
        local_point  = contact_manifold.localPoint

        self._local_contact_normal = np.array([local_normal[0], local_normal[1]])
        self._local_contact_point  = np.array([local_point[0],  local_point[1]])
        self._normal_impulse_mag   = np.asarray(impulse.normalImpulses)
        self._tangent_impulse_mag  = np.asarray(impulse.tangentImpulses)

        # print "Normal point \t", self._local_contact_normal
        # print "Local  point \t", self._local_contact_point

        # print "Normal impulse magnitude \t", self._tangent_impulse_mag
        # print "Tangent impulse magnitude \t", self._tangent_impulse_mag

        # return self._local_contact_normal, self._local_contact_point, self._tangent_impulse_mag, self._tangent_impulse_mag