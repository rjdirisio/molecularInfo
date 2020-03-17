import numpy as np
import numpy.linalg as la

class molecInfo:
    def __init__(self,coordinates):
        """Takes xyz coordinates of an molecule, and calculates attributes associated with the molecule. This is built
        with DMC wave functions in mind, but can be used for any xyz atom. This class handles bond lengths, angles, etc.
        @param coordinates: nxmx3 array, n = number of geometries (or walkers), m = number of atoms, 3 is x y z
        @type coordinates: np.ndarray"""
        self.xx = coordinates
        if len(self.xx.shape) == 2: #if only one geometry, make two
            self.xx = np.expand_dims(self.xx,axis=0)

    @classmethod
    def dotPdt(cls,v1,v2):
        """Fancy threaded,vectorized dot product"""
        new_v1 = np.expand_dims(v1,axis=1)
        new_v2 = np.expand_dims(v2,axis=2)
        return np.matmul(new_v1, new_v2).squeeze()

    def bondLength(self, atm1, atm2):
        return la.norm(self.xx[:,atm1] - self.xx[:,atm2],axis=1)

    def bondAngle(self, atm1,atmMid,atm3):
        """Calculate atm1 - atmMid - atm3 angle"""
        vec1 = self.xx[:,atm1] - self.xx[:,atmMid]
        vec2 = self.xx[:,atm3] - self.xx[:,atmMid]
        #cos(x) = left.right/(|left|*|right|)
        dotV = np.arccos(self.dotPdt(vec1,vec2) /
                         (la.norm(vec1,axis=1),la.norm(vec2,axis=1)))
        return dotV

    def bisectingVector(self,atm1,atmM,atm3):
        """Get  normalized bisecting vector of two vectors (two bond lengths).
        It is assumed that the two flanking atoms have the same central atom.
        |b|*a + |a|*b"""
        b1 = self.xx[:,atm1] - self.xx[:,atmM]
        b2 = self.xx[:,atm3] - self.xx[:,atmM]
        bis = la.norm(b1,axis=1) * b2 + \
              la.norm(b2,axis=1) * b1
        return bis / la.norm(bis,axis=1)

    def getComponent(self,atm,xyz):
        """Get x, y, or z component of a vector that corresponds to a particular atom in the predetermined cooridnate
        system.
        @param atm: atom's index
        @type: int
        @param xyz: Either 0 (x), 1 (y), or 2 (z)
        @type xyz: int
        @return: """
        return self.xx[:,atm,xyz]

    def cartToSpherical(self,vecc):
        """Takes stack of vectors and translates them to stack of r theta phi coordinates,
        with the assumption that you are already in the coordinate system you desire."""
        r = la.norm(vecc, axis=1)
        th = np.arccos(vecc[:, -1] / r)  #z/r
        phi = np.arctan2(vecc[:, 1], vecc[:, 0]) #y/x
        return r,th,phi