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
            new_vecc = np.expand_dims(vecc,-1) #nx3x1
            rot_vecs = np.matmul(rotMs, new_vecc).squeeze()
            return rot_vecs
        @classmethod
        def genXYZ(cls,theta,XYZ):
            """Generates the 3D rotation matrix about X, Y, or Z by theta radians"""
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
            elif XYZ == 2:
                rotM[:, 0, :] = np.column_stack((np.cos(theta), -1 *np.sin(theta), zeroLth))
                rotM[:, 1, :] = np.column_stack((np.sin(theta), np.cos(theta), zeroLth))
                rotM[:, 2, :] = np.tile([0,1,0],len(theta),1)
            return rotM


        @classmethod
        def genEckart(cls,geoms,refGeom,masses,planar=False):
            """Old code. Need to rework so there aren't so many transposes.
            Generate rotation matrices such that geometries to look *the most* like the reference geometry, following:
            Eckart vectors, Eckart frames, and polyatomic molecules - James D. Louck and Harold W. Galbraith
            Slightly modified algorithm if reference is planar or not.
            @param geoms: geometries to be rotated
            @type geoms: np.ndarray
            @param refGeom: the reference geometry. 2d numpy array (m atoms x 3)
            @type: refGeom: np.ndarray
            @param masses: the masses that correspond to to the atoms in geoms and refGeom
            @type masses: list
            @param planar: whether or not the reference geometry is planar
            @type planar: boolean
            """
            com = np.dot(masses, geoms) / np.sum(masses)
            refCOM = np.dot(masses, refGeom) / np.sum(masses)  # same as overal COM
            refGeom -= refCOM
            # First Translate:
            print('shifting molecules')
            ShiftedMolecules = geoms - com[:, np.newaxis, :]
            # Equation 3.1
            print('starting mathy math')
            fP = np.sum(
                ShiftedMolecules[:, :, :, np.newaxis] * refGeom[np.newaxis, :, np.newaxis, :] *
                masses[np.newaxis, :,np.newaxis,np.newaxis], axis=1)
            myF = np.transpose(fP, (0, 2, 1))
            if planar:
                print('planar')
                all3 = np.arange(3)
                indZ = np.where(np.around(myF, 4).any(axis=2))[1][:2]
                myFp = myF[:, indZ, :]
                myFF = np.matmul(myFp, myFp.transpose(0, 2, 1))
                peval, pevecs = la.eigh(myFF)
                invRootDiagF2 = 1 / np.sqrt(peval)
                singularMats = np.where(np.isnan(invRootDiagF2))[0]
                if len(singularMats) > 0:
                    raise ZeroDivisionError("this is bad, dude")
                pevecsT = np.transpose(pevecs, (0, 2, 1))
                eckVecs2 = np.zeros((len(invRootDiagF2), 3, 3))
                invRootF2 = np.matmul(invRootDiagF2[:, np.newaxis, :] * pevecs, pevecsT)
                eckVecs2[:, :, indZ] = np.matmul(np.transpose(myFp, (0, 2, 1)), invRootF2)
                if indZ[0] == 0 and indZ[1] == 2:  # then we have to cross Z x X
                    eckVecs2[:, :, np.setdiff1d(all3, indZ)[0]] = np.cross(eckVecs2[:, :, indZ[1]],
                                                                           eckVecs2[:, :, indZ[0]])
                else:  # others can use the same generic formula because it's a "forward" cross pdt (X x Y ; Y x Z)
                    eckVecs2[:, :, np.setdiff1d(all3, indZ)[0]] = np.cross(eckVecs2[:, :, indZ[0]],
                                                                           eckVecs2[:, :, indZ[1]])
            else:
                myFF = np.matmul(myF, fP)
                bigEvals, bigEvecs = la.eigh(myFF)
                bigEvecsT = np.transpose(bigEvecs, (0, 2, 1))
                invRootDiagF2 = 1.0 / np.sqrt(bigEvals)
                singularMats = np.where(np.isnan(invRootDiagF2))[0]
                if len(singularMats) > 0:
                    raise ZeroDivisionError("this is bad, dude")
                invRootF2 = np.matmul(invRootDiagF2[:, np.newaxis, :] * -bigEvecs, -bigEvecsT, )  # -bigEvecs
                eckVecs2 = np.matmul(np.transpose(myF, (0, 2, 1)), invRootF2)
            mas = np.where(np.around(la.det(eckVecs2)) == -1.0)[0] #inversion of coordinate system
            if mas > 0:
                raise Exception
            return com, eckVecs2.transpose(0,2,1)



        @classmethod
        def rotToXYPlane(cls,geoms,orig,xax,xyp,retMat=False):
            """Rotate geometries to XY plane, placing one atom at the origin, one on the xaxis,
            and one on the xyplane.  Done through successive X-Z-X rotations. There is some level of ambiguity in the linear
            algebra..."""
            if len(geoms.shape) == 2:
                geoms = np.expand_dims(geoms,0)
            #translation of orig to origin
            geoms -= geoms[:,orig]
            # Rotation of xax to x axis
            xaxVec = geoms[:, xax, :]
            x = xaxVec[:, 0]
            y = xaxVec[:, 1]
            z = xaxVec[:, 2]
            theta = np.arctan2(-z, y)
            alpha = np.arctan2((-1 * (y * np.cos(theta) - np.sin(theta) * z)), x)
            r1 = cls.genXYZ(theta,0)
            r2 = cls.genXYZ(alpha,2)
            rotM = np.matmul(r2, r1)
            geoms = cls.rotateGeoms(rotM,geoms)
            # Rotation or xyp to xyplane
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
            from molecularInfo import *
            zdot = molecInfo.dotPdt(z,Z) / (la.norm(z, axis=1) * la.norm(Z, axis=1))
            Yzdot = molecInfo.dotPdt(Y,z) / (la.norm(Y, axis=1) * la.norm(z, axis=1))
            Xzdot = molecInfo.dotPdt(X,z) / (la.norm(X, axis=1) * la.norm(z, axis=1))
            yZdot = molecInfo.dotPdt(y,Z) / (la.norm(y, axis=1) * la.norm(Z, axis=1))
            xZdot = molecInfo.dotPdt(x,Z) / (la.norm(x, axis=1) * la.norm(Z, axis=1))
            Theta = np.arccos(zdot)
            tanPhi = np.arctan2(Yzdot, Xzdot)
            tanChi = np.arctan2(yZdot, -xZdot)  # negative baked in
            return Theta, tanPhi, tanChi

        @classmethod
        def extractEulers(cls,rotMs):
            """From a rotation matrix, calculate the three euler angles theta,phi and Chi. This is based on
            a 'ZYZ' euler rotation"""
            zdot = rotMs[:, -1, -1]
            Yzdot = rotMs[:, 2, 1]
            Xzdot = rotMs[:, 2, 0]
            yZdot = rotMs[:, 1, 2]
            xZdot = rotMs[:, 0, 2]
            Theta = np.arccos(zdot)
            tanPhi = np.arctan2(Yzdot, Xzdot)
            tanChi = np.arctan2(yZdot, xZdot)
            return Theta, tanPhi, tanChi