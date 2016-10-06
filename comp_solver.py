# composite structure solver

import numpy as np
import scipy as sp
import scipy.linalg as la
from math import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from plate_layerwise import *

class comp_solver:

    def __init__(self,topo,BC,f,thickness_vec,scale_size,angle_vec,Evec):

        # node number and elem2node mapping
        self.node_N = topo[0]
        self.elem = topo[1]
        self.elem_N = len(self.elem)

        # boundary condition and load
        self.BC = BC
        self.f = f

        # thickness, length and orientation
        self.thickness_vec = thickness_vec
        self.scale_size = scale_size
        self.angle_vec = angle_vec

        # material info: Young's Modulus, Poisson ratio, Shear Modulus
        self.Evec = Evec

    def glo_stiffness(self):

        # generate the one layer composite local stiffness matrix

        self.set_Kloc()

        K_glo = np.matrix(np.zeros((5*self.node_N,5*self.node_N)))


        # mapping the local K to global K

        for i in xrange(self.elem_N):

            K_loc = self.K_loc_vec[i]
            elem_loc = self.elem[i]
            print(self.elem)

            for j in xrange(9):
                for k in xrange(9):

                    ind_row_glo = elem_loc[j]
                    ind_col_glo = elem_loc[k]

                    K_loc_block = K_loc[j*5:(j+1)*5,k*5:(k+1)*5]

                    print(K_loc_block.shape)
                    print(K_glo.shape)
                    print(ind_row_glo*5,(ind_row_glo+1)*5)

                    K_glo[ind_row_glo*5:(ind_row_glo+1)*5,ind_col_glo*5:(ind_col_glo+1)*5] += K_loc_block

        # BC

        BC_lambda = 10**9

        for BC_loc in self.BC:
            for j in xrange(5):
                K_glo[BC_loc*5+j,BC_loc*5+j] += BC_lambda

        self.K_glo = K_glo

        # plt.imshow(K_glo, interpolation='nearest', cmap=plt.cm.ocean, extent=(0.5,10.5,0.5,10.5))
        # plt.colorbar()
        # plt.show()

    def set_Kloc(self):

        self.K_loc_vec = []

        for i in xrange(self.elem_N):

            hvec = [-0.5*self.thickness_vec[i],0.5*self.thickness_vec[i]]
            local_elem = layerElement(self.Evec,self.angle_vec[i],hvec,self.scale_size)

            self.K_loc_vec.append(local_elem.stiffnessMat())


    def set_Ksens(self):

        self.K_loc_sens_vec = []

        for i in xrange(self.elem_N):

            hvec = [-0.5*self.thickness_vec[i],0.5*self.thickness_vec[i]]
            local_elem = layerElement(self.Evec,self.angle_vec[i],hvec,self.scale_size)

            self.K_loc_sens_vec.append(local_elem.stiffnessMat_devh())


    def get_Kloc(self):

        return self.K_loc_vec

    def get_Ksens(self):

        return self.K_loc_sens_vec

    def set_stress_oprt(self):

        self.stress_oprt = []

        for i in xrange(self.elem_N):
            hvec = [-0.5*self.thickness_vec[i],0.5*self.thickness_vec[i]]
            local_elem = layerElement(self.Evec,self.angle_vec[i],hvec,self.scale_size)

            self.stress_oprt.append(local_elem.fiber_coord_stress("center_stress"))

    def get_stress_oprt(self):

        return self.stress_oprt

    def solve(self):
        self.u = np.linalg.solve(self.K_glo,self.f)





class post_process:

    def __init__(self,u,elem):
        self.u = u # N*5 mat

        self.elem = elem

        self.elem_N = len(elem)
        self.node_N = int(u.shape[0]/5)

    def set_conn(self,conn):

        self.conn = conn

    def set_x(self,x):

        self.x = x # N*3 mat

    def plot_preprocess(self,x,conn):

        self.set_conn(conn)
        self.set_x(x)

    def set_u_trans(self):
        self.u_disp = np.matrix(np.zeros((self.node_N,3)))

        for i in xrange(self.node_N):
            self.u_disp[i,0] = self.u[i*5+0]
            self.u_disp[i,1] = self.u[i*5+1]
            self.u_disp[i,2] = self.u[i*5+2]

    def set_new_coord(self):
        self.x_new = self.x+self.u_disp

    def get_u_for_elem(self,ind_elem):
        # return a list of disp
        elem_loc = self.elem[ind_elem]
        u = []
        for i in xrange(9):
            node_ind = elem_loc[i]

            u_loc = self.u[node_ind*5:(node_ind+1)*5,0]

            for j in xrange(5):
                u.append(u_loc[j,0])

        return u

    def get_u_for_all_elem(self):
        self.u2elem = np.zeros((self.elem_N,9*5))
        for i in xrange(self.elem_N):
            u_elem = self.get_u_for_elem(i)
            self.u2elem[i][:] = u_elem[:]

    def plot(self):

        self.set_u_trans()
        self.set_new_coord()

        line_N = len(conn)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # original

        for i in xrange(line_N):

            ind1 = conn[i][0]
            ind2 = conn[i][1]

            x1 = self.x[ind1,0]
            y1 = self.x[ind1,1]
            z1 = self.x[ind1,2]

            x2 = self.x[ind2,0]
            y2 = self.x[ind2,1]
            z2 = self.x[ind2,2]

            x = [x1,x2]
            y = [y1,y2]
            z = [z1,z2]
            ax.plot(x, y, z,'-b')

        for i in xrange(line_N):

            ind1 = conn[i][0]
            ind2 = conn[i][1]

            x1 = self.x_new[ind1,0]
            y1 = self.x_new[ind1,1]
            z1 = self.x_new[ind1,2]

            x2 = self.x_new[ind2,0]
            y2 = self.x_new[ind2,1]
            z2 = self.x_new[ind2,2]

            x = [x1,x2]
            y = [y1,y2]
            z = [z1,z2]
            ax.plot(x, y, z,'-g')

        plt.show()







if (1==0):
    # test
    elem = [[0,1,2,5,6,7,10,11,12],
    [2,3,4,7,8,9,12,13,14],
    [10,11,12,15,16,17,20,21,22],
    [12,13,14,17,18,19,22,23,24]]
    topo = [25,elem]


    BC = [0,1,2,3,4]

    f = np.matrix(np.zeros((25*5,1)))
    f[20*5+2] = 10.0

    thickness_vec = [0.01,0.01,0.01,0.01]

    scale_size = [1.0,1.0]

    #angle_vec = [0,0,0,0]
    angle_vec = [np.pi/2,np.pi/2,np.pi/2,np.pi/2]

    E1 = 14.69*1e9
    E2 = 1.062*1e9
    G12 = 0.545*1e9
    G23 = 0.399*1e9
    nu12 = 0.33
    Evec = [E1,E2,G12,G23,nu12]

    four_elem_pro = comp_solver(topo,BC,f,thickness_vec,scale_size,angle_vec,Evec)
    four_elem_pro.glo_stiffness()
    four_elem_pro.solve()

    # u_disp = np.matrix(np.zeros((25,3)))
    # for i in xrange(25):
    #     u_disp[i,0] = four_elem_pro.u[i*5+0]
    #     u_disp[i,1] = four_elem_pro.u[i*5+1]
    #     u_disp[i,2] = four_elem_pro.u[i*5+2]
    x = np.matrix(np.zeros((25,3)))
    for i in xrange(5):
        ind0 = i*5

        x[ind0+0,0] = 0.0
        x[ind0+1,0] = 0.5
        x[ind0+2,0] = 1.0
        x[ind0+3,0] = 1.5
        x[ind0+4,0] = 2.0

        x[ind0+0,1] = i*0.5
        x[ind0+1,1] = i*0.5
        x[ind0+2,1] = i*0.5
        x[ind0+3,1] = i*0.5
        x[ind0+4,1] = i*0.5

    conn = [[0,1],
    [1,2],
    [0,5],
    [5,10],
    [10,11],
    [11,12],
    [2,7],
    [7,12],
    [2,3],
    [3,4],
    [4,9],
    [9,14],
    [13,14],
    [12,13],
    [10,15],
    [15,20],
    [12,17],
    [17,22],
    [20,21],
    [21,22],
    [14,19],
    [19,24],
    [22,23],
    [23,24]]

    fem_post = post_process(four_elem_pro.u,elem)
    fem_post.plot_preprocess(x,conn)
    fem_post.get_u_for_all_elem()
    print '++++++++++++++++fem_post.u2elem',fem_post.u2elem
    fem_post.plot()
