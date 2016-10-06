# finite element code for a composite plate
# formulated from K contribution from each layer
# based on Huang, Friedmann :
# "An Integrated Aerothermalelastic Analysis Framework for
#  Predicting the Response of Composite Panels"
# 2nd order, linear model, with shear, w/o heat
# copyright@ Sicheng He, 2016

# Now we only support simple geometry:
#     square with side length for each element
#     global topology be a rectangular

# Now only supports fully fixed BC

import numpy as np
import scipy as sp
import scipy.linalg as la
import math


class layerElement:

    # gives the stiffness matrix of a single layer of an element

    def __init__(self,Evec,theta,hvec,scale_size):
        # independent var
        self.E1 = Evec[0]
        self.E2 = Evec[1]
        self.G12 = Evec[2]
        self.G23 = Evec[3]
        self.nu12 = Evec[4]

        self.theta = theta

        self.h_i = hvec[0]
        self.h_ip1 = hvec[1]

        self.Lx = scale_size[0]
        self.Ly = scale_size[1]

        # dependent
        self.nu21 = self.E2*self.nu12/self.E1
        self.G13 = self.G12


    def QmatFun(self):

        Q11 = self.E1/(1.0-self.nu12*self.nu21)
        Q22 = self.E2/(1.0-self.nu12*self.nu21)
        Q12 = self.nu12*self.E2/(1-self.nu12*self.nu21)
        Q66 = self.G12

        c = np.cos(self.theta)
        s = np.sin(self.theta)

        Qbar11 = Q11*c**4+Q22*s**4+2*(Q12+2*Q66)*s**2*c**2
        Qbar12 = (Q11+Q22-4.0*Q66)*c**2*s**2+Q12*(s**4+c**4)
        Qbar22 = Q11*s**4+Q22*c**4+2.0*(Q12+2.0*Q66)*s**2*c**2
        Qbar16 = (Q11-Q12-2.0*Q66)*c**3*s-(Q22-Q12-2.0*Q66)*c*s**3
        Qbar26 = (Q11-Q12-2.0*Q66)*c*s**3-(Q22-Q12-2.0*Q66)*c**3*s
        Qbar66 = (Q11+Q22-2.0*Q12-2.0*Q66)*c**2*s**2+Q66*(s**4+c**4)

        Qbarmat = np.matrix([[Qbar11,Qbar12,Qbar16],
                            [Qbar12,Qbar22,Qbar26],
                            [Qbar16,Qbar26,Qbar66]])

        return Qbarmat

    def SmatFun(self):

        kappa = 5.0/6.0

        C44 = 2.0*self.G23
        C55 = 2.0*self.G13

        c = np.cos(self.theta)
        s = np.sin(self.theta)

        C44_bar = C44*c**2+C55*s**2
        C45_bar = (C55-C44)*c*s
        C55_bar = C44*s**2+C55*c**2

        Cmat = np.array([[C44_bar,C45_bar],[C45_bar,C55_bar]])
        Qmat = 0.5*Cmat
        #Qmat = Cmat

        kappa_Qmat = kappa*Qmat

        return kappa_Qmat


    def f(self,x):

        f1 = 2.0*(x-0.5)*(x-1.0)
        f2 = -4.0*x*(x-1)
        f3 = 2.0*x*(x-0.5)

        f = [f1,f2,f3]

        return f


    def devf(self,x):

        devf1 = 4.0*x-3.0
        devf2 = -8.0*x+4.0
        devf3 = 4.0*x-1.0

        devf = [devf1,devf2,devf3]

        return devf

    def N(self,x,y):

        fx = self.f(x)
        fy = self.f(y)

        N_vec = np.zeros(9)
        for i in xrange(3):
            for j in xrange(3):
                N_vec[3*i+j] = fy[i]*fx[j]

        return N_vec

    def N_x(self,x,y):

        fy = self.f(y)
        fx_dx = self.devf(x)
        fx_dx = [loc_x/self.Lx for loc_x in fx_dx]

        N_x_vec = np.zeros(9)

        for i in xrange(3):
            for j in xrange(3):
                N_x_vec[3*i+j] = fy[i]*fx_dx[j]

        return N_x_vec

    def N_y(self,x,y):

        fx = self.f(x)
        fy_dy = self.devf(y)
        fy_dy = [loc_y/self.Ly for loc_y in fy_dy]

        N_y_vec = np.zeros(9)

        for i in xrange(3):
            for j in xrange(3):
                N_y_vec[3*i+j] = fy_dy[i]*fx[j]

        return N_y_vec

    def L1(self,x,y):

        L1mat = np.matrix(np.zeros((3,5*9)))

        N_y_vec = self.N_y(x,y)
        N_x_vec = self.N_x(x,y)

        for i in xrange(9):

            col_ind_0 = i*5

            L1mat[0,col_ind_0] = N_x_vec[i]
            L1mat[1,col_ind_0+1] = N_y_vec[i]
            L1mat[2,col_ind_0] = N_y_vec[i]
            L1mat[2,col_ind_0+1] = N_x_vec[i]

        return L1mat

    def L2(self,x,y):

        L2mat = np.matrix(np.zeros((3,5*9)))

        N_y_vec = self.N_y(x,y)
        N_x_vec = self.N_x(x,y)

        for i in xrange(9):

            col_ind_0 = i*5

            L2mat[0,col_ind_0+3] = N_x_vec[i]
            L2mat[1,col_ind_0+4] = N_y_vec[i]
            L2mat[2,col_ind_0+3] = N_y_vec[i]
            L2mat[2,col_ind_0+4] = N_x_vec[i]

        return L2mat

    def L3(self,x,y):

        L3mat = np.matrix(np.zeros((2,5*9)))

        N_y_vec = self.N_y(x,y)
        N_x_vec = self.N_x(x,y)

        N_vec = self.N(x,y)

        for i in xrange(9):

            col_ind_0 = i*5

            L3mat[0,col_ind_0+2] = N_y_vec[i]
            L3mat[0,col_ind_0+4] = N_vec[i]
            L3mat[1,col_ind_0+2] = N_x_vec[i]
            L3mat[1,col_ind_0+3] = N_vec[i]

        return L3mat


    def locABBD(self):

        Qloc = self.QmatFun()

        A = Qloc*(self.h_ip1-self.h_i)
        B = Qloc*(self.h_ip1**2/2.0-self.h_i**2/2.0)
        D = Qloc*(self.h_ip1**3/3.0-self.h_i**3/3.0)

        return [A,B,D]


    def locSfunc(self):

        Sloc = self.SmatFun()

        S = Sloc*(self.h_ip1-self.h_i)

        return S


    def stiffnessMat(self):

        [A,B,D] = self.locABBD()
        S = self.locSfunc()

        # for ABBD we apply 3*3 guassian quad

        gaussQuad_coord1d_3pt = np.array([-np.sqrt(3)/np.sqrt(5),
                                     0.0,
                                     np.sqrt(3)/np.sqrt(5)])
        gaussQuad_coord1d_3pt = gaussQuad_coord1d_3pt/2.0
        gaussQuad_coord1d_3pt = gaussQuad_coord1d_3pt+0.5


        gaussQuad_weight_3pt = np.array([5.0/18,4.0/9,5.0/18])

        Kloc = np.matrix(np.zeros((5*9,5*9)))

        Area = self.Lx*self.Ly

        for i in range(3):
            for j in range(3):

                xloc = gaussQuad_coord1d_3pt[i]
                yloc = gaussQuad_coord1d_3pt[j]

                weight_loc = gaussQuad_weight_3pt[i]*gaussQuad_weight_3pt[j]

                weight_loc = weight_loc*Area

                L1_loc = self.L1(xloc,yloc)
                L2_loc = self.L2(xloc,yloc)

                Kloc += np.transpose(L1_loc).dot(A).dot(L1_loc)*weight_loc
                Kloc += np.transpose(L1_loc).dot(B).dot(L2_loc)*weight_loc
                Kloc += np.transpose(L2_loc).dot(B).dot(L1_loc)*weight_loc
                Kloc += np.transpose(L2_loc).dot(D).dot(L2_loc)*weight_loc

        # for shear we apply 2*2 guassian point
        # (reduced integration to void shear locking)

        gaussQuad_coord1d_2pt = np.array([-1.0/np.sqrt(3.0),1.0/np.sqrt(3.0)])
        gaussQuad_coord1d_2pt = gaussQuad_coord1d_2pt/2.0
        gaussQuad_coord1d_2pt = gaussQuad_coord1d_2pt+0.5
        gaussQuad_weight_2pt = np.array([0.5,0.5])

        for i in range(2):
            for j in range(2):

                xloc = gaussQuad_coord1d_2pt[i]
                yloc = gaussQuad_coord1d_2pt[j]

                weight_loc = gaussQuad_weight_2pt[i]*gaussQuad_weight_2pt[j]

                weight_loc = weight_loc*Area

                L3_loc = self.L3(xloc,yloc)

                Kloc += np.transpose(L3_loc).dot(S).dot(L3_loc)*weight_loc


        return Kloc

    def strain_inplane_oprt(self,x,y,z):

        # in plane strain operator: Lu=epsilon
        L1_mat = self.L1(x,y)
        L2_mat = self.L2(x,y)

        return L1_mat+L2_mat*z

    def strain_outplane_oprt(self,x,y):

        # out of plane strain operator: Lu=epsilon
        L3_mat = self.L3(x,y)

        return L3_mat

    def stress_oprt(self,x,y,z):

        strain_inplane = self.strain_inplane_oprt(x,y,z)
        strain_outplane = self.strain_outplane_oprt(x,y)

        Qloc = self.QmatFun()
        Sloc = self.SmatFun()

        stess_inplane = Qloc.dot(strain_inplane)
        stress_outplane = Sloc.dot(strain_outplane)

        return np.concatenate((stess_inplane, stress_outplane), axis=0)

    def corner_stress(self):

        stress_oprt_list = []

        hvec = [self.h_i,self.h_ip1]
        xvec = [0.0,1.0]
        yvec = [0.0,1.0]

        for i in range(2):
            for j in range(2):
                for k in range(2):

                    stress_oprt_list.append(self.stress_oprt(xvec[k],yvec[j],hvec[i]))

        return stress_oprt_list

    def center_stress(self):

        # only the inplane stress considered at the midplane
        # curvature related strain will be zero there

        strain_inplane = self.strain_inplane_oprt(0.5,0.5,0.0)

        Qloc = self.QmatFun()

        stress_inplane = Qloc.dot(strain_inplane)

        return stress_inplane


    def fiber_coord_stress(self,flag):

        T_mat_inv = np.matrix(np.zeros((3,3)))

        s = sin(self.theta)
        c = cos(self.theta)

        s2 = s**2
        c2 = c**2
        sc = s*c

        T_mat_inv[0,0] = c2
        T_mat_inv[0,1] = s2
        T_mat_inv[0,2] = 2.0*sc
        T_mat_inv[1,1] = c2
        T_mat_inv[1,2] = -2*sc
        T_mat_inv[2,2] = c2-s2
        T_mat_inv[1,0] = T_mat_inv[0,1]
        T_mat_inv[2,0] = T_mat_inv[0,2]
        T_mat_inv[2,1] = T_mat_inv[1,2]

        if (flag=="center_stress"):
            stress_inplane = self.center_stress()
            return T_mat_inv.dot(stress_inplane)









    # following are the linearized Ku=f model contents
    # only applicable for one layer case now
    def stiffnessMat_devh(self):

        Q_loc = self.QmatFun()
        S_loc = self.SmatFun()

        # for ABBD we apply 3*3 guassian quad

        gaussQuad_coord1d_3pt = np.array([-np.sqrt(3)/np.sqrt(5),
                                     0.0,
                                     np.sqrt(3)/np.sqrt(5)])
        gaussQuad_coord1d_3pt = gaussQuad_coord1d_3pt/2.0
        gaussQuad_coord1d_3pt = gaussQuad_coord1d_3pt+0.5


        gaussQuad_weight_3pt = np.array([5.0/18,4.0/9,5.0/18])

        Kloc_sens = np.matrix(np.zeros((5*9,5*9)))

        Area = self.Lx*self.Ly

        for i in range(3):
            for j in range(3):

                xloc = gaussQuad_coord1d_3pt[i]
                yloc = gaussQuad_coord1d_3pt[j]

                weight_loc = gaussQuad_weight_3pt[i]*gaussQuad_weight_3pt[j]

                weight_loc = weight_loc*Area

                L1_loc = self.L1(xloc,yloc)
                L2_loc = self.L2(xloc,yloc)

                Kloc_sens += np.transpose(L1_loc).dot(Q_loc).dot(L1_loc)*weight_loc
                #Kloc_sens += np.transpose(L1_loc).dot(B).dot(L2_loc)*weight_loc # one layer so symmetric
                #Kloc_sens += np.transpose(L2_loc).dot(B).dot(L1_loc)*weight_loc # one layer so symmetric
                Kloc_sens += np.transpose(L2_loc).dot(Q_loc).dot(L2_loc)*weight_loc*(h_ip1-h_i)**2/4.0

        # for shear we apply 2*2 guassian point
        # (reduced integration to void shear locking)

        gaussQuad_coord1d_2pt = np.array([-1.0/np.sqrt(3.0),1.0/np.sqrt(3.0)])
        gaussQuad_coord1d_2pt = gaussQuad_coord1d_2pt/2.0
        gaussQuad_coord1d_2pt = gaussQuad_coord1d_2pt+0.5
        gaussQuad_weight_2pt = np.array([0.5,0.5])

        for i in range(2):
            for j in range(2):

                xloc = gaussQuad_coord1d_2pt[i]
                yloc = gaussQuad_coord1d_2pt[j]

                weight_loc = gaussQuad_weight_2pt[i]*gaussQuad_weight_2pt[j]

                weight_loc = weight_loc*Area

                L3_loc = self.L3(xloc,yloc)

                Kloc_sens += np.transpose(L3_loc).dot(S_loc).dot(L3_loc)*weight_loc

        return Kloc_sens
