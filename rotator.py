import numpy as np
import numpy.linalg as la

class molRotator:
        @classmethod
        def rotateGeoms(cls,rotMs,geoms):
            """Takes in a stack of rotation matrices and applies it to a stack of geometries."""
            new_geoms = np.expand_dims(geoms,-1) #nxmx3x1
            new_rotms = np.expand_dims(rotMs,1) #nx1x3x3
            rot_geoms = np.matmul(new_rotms, new_geoms).squeeze()
            return rot_geoms

        @classmethod
        def rotateVector(cls,rotMs,vecc):
            """Takes in a stack of rotation matrices and applies it to a stack of vector"""

        @classmethod
        def genXYZ(cls,theta,XYZ):
            """Generates the rotation matrix where you rotate about XYZ by theta radians"""
            theta = [theta] if isinstance(theta, float) else theta
            rotM = np.zeros(len(theta),3,3)
            zeroLth = np.zeros(len(theta))
            if XYZ == 0:
                rotM[:, 0] = np.tile([1, 0, 0], (len(theta), 1))
                rotM[:, 1] = np.column_stack((zeroLth, np.cos(theta), -1 * np.sin(theta)))
                rotM[:, 2] = np.column_stack((zeroLth, np.sin(theta), np.cos(theta)))
            elif XYZ == 1:
                rotM[:, 0] = np.column_stack((np.cos(theta),zeroLth, -1 * np.sin(theta)))
                rotM[:, 1] = np.tile([0,1,0],len(theta),1)
                rotM[:, 2] = np.column_stack((np.sin(theta),zeroLth, np.cos(theta)))
            return rotM


        @classmethod
        def genEckart(cls,geoms,refGeom,planar=False,retMat=True):
            """Rotate geometries to look *the most* like the reference geometry, following:
            Eckart vectors, Eckart frames, and polyatomic molecules - James D. Louck and Harold W. Galbraith
            Can do a planar one or not."""

        @classmethod
        def rotToXYPlane(cls,geoms,orig,xax,xyp,retMat=True):
            """Rotate geometries to XY plane, placing one atom at the origin, one on the xaxis,
            and one on the xyplane.  Done through successive X-Z-X rotations. There is some level of ambiguity in the linear
            algebra..."""
            if len(geoms.shape) == 2:
                geoms = np.expand_dims(geoms,0)
            geoms -= geoms[:,orig]
            geomZeros = np.zeros(len(geoms))
            # Rotation to x axis
            o2 = geoms[:, xax, :]
            x = o2[:, 0]
            y = o2[:, 1]
            z = o2[:, 2]
            theta = np.arctan2(-z, y)
            alpha = np.arctan2((-1 * (y * np.cos(theta) - np.sin(theta) * z)), x)
            r1 = cls.genXYZ(theta,0)
            r2 = cls.genXYZ(alpha,2)
            rotM = np.matmul(r2, r1)
            geoms = cls.rotateGeoms(rotM,geoms)
            # Rotation to xyplane
            xypVec = geoms[:, xyp]
            z = xypVec[:, 2]
            y = xypVec[:, 1]
            beta = np.arctan2(-1 * z, y)
            r3 = cls.genXYZ(beta,0)
            geoms = cls.rotateGeoms(r3,geoms)
            if retMat:
                return geoms,r3.dot(r2.dot(r1))
            else:
                return geoms

        @classmethod
        def genEulers(cls,x,y,z,X,Y,Z):
            """Takes in cartesian vectors and gives you the 3 euler angles that bring xyz to XYZ based on a 'ZYZ'
            rotation"""
            zdot = (z * Z).sum(axis=1) / (la.norm(z, axis=1) * la.norm(Z, axis=1))
            Yzdot = (Y * z).sum(axis=1) / (la.norm(Y, axis=1) * la.norm(z, axis=1))
            Xzdot = (X * z).sum(axis=1) / (la.norm(X, axis=1) * la.norm(z, axis=1))
            yZdot = (y * Z).sum(axis=1) / (la.norm(y, axis=1) * la.norm(Z, axis=1))
            xZdot = (x * Z).sum(axis=1) / (la.norm(x, axis=1) * la.norm(Z, axis=1))
            Theta = np.arccos(zdot)
            tanPhi = np.arctan2(Yzdot, Xzdot)
            tanChi = np.arctan2(yZdot, -xZdot)  # negative baked in
            return Theta, tanPhi, tanChi

        @classmethod
        def extractEulers(cls,rotMs):
            """From a rotation matrix, calculate the three euler angles theta,phi and Chi. This is based on
            a 'ZYZ' euler rotation"""
            # [x]    [. . .][X]
            # [y] =  [. . .][Y]
            # [z]    [. . .][Z]
            zdot = rotMs[:, -1, -1]
            Yzdot = rotMs[:, 2, 1]
            Xzdot = rotMs[:, 2, 0]
            yZdot = rotMs[:, 1, 2]
            xZdot = rotMs[:, 0, 2]
            Theta = np.arccos(zdot)
            tanPhi = np.arctan2(Yzdot, Xzdot)
            tanChi = np.arctan2(yZdot, xZdot)
            return Theta, tanPhi, tanChi