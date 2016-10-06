import numpy as np
import scipy as sp
import scipy.linalg as la
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.axes3d import get_test_data
import time

from gurobipy import *

from plate_layerwise import *

from opt import *

from comp_solver import *

# tests\
# basic case:
def topology_generator(N):

    # take in the total # of elements
    # return element topology and boundary (assume one side fixed)

    side_N = int(np.sqrt(N))
    node_perside = (side_N-1)*2+3

    elem = []

    for i in range(side_N):
        for j in range(side_N):

            elem_loc_mat = np.matrix(np.zeros((3,3),dtype=np.int16))

            row_index_0 = 2*i
            row_index = [row_index_0,row_index_0+1,row_index_0+2]

            col_index_0 = 2*j
            col_index = [col_index_0,col_index_0+1,col_index_0+2]

            for p in range(3):
                for q in range(3):
                    node_index = row_index[p]*node_perside+col_index[q]
                    elem_loc_mat[p,q] = node_index


            elem.append(elem_loc_mat)

    BC = []
    for i in range(node_perside):
        BC.append(i)

    node_N = node_perside**2

    return([elem,BC,node_N])




def basic_test(N,M,int_flag,wt_cmpl_flag,N_angle,cut_flag,solution_limit):

    # The basic tests will take:
    # i).    N as how many elements it has (1,4,9,16,25,...,K^2);
    # ii).   M how many layers it has;
    # iii).  scale combination: (mm,Gpa) vs. (m,pa)
    # as inputs and form a rectgular (in terms of geo & topo)
    # and optimize the structure

    [elem,BC_dof,node_N] = topology_generator(N)

    ########## FEM setup #############################################

    # parameter
    ## bound on disp
    # ulb = -3.0
    # uub = 3.0


    # topoGeo info
    ## topology
    ### dof per node
    dof_per_node = 5


    ## geometry
    ### z axis arrangement
    hvec = np.linspace(-1.0,1.0,M+1)
    scale_size = [1.0,1.0]

    topoGeo_cong = [node_N,dof_per_node,elem,hvec,scale_size]



    # material property
    ## mechanics property
    E1 = 14.69
    E2 = 1.062
    G12 = 0.545
    G23 = 0.399
    nu12 = 0.33


    if wt_cmpl_flag==0:
        mat1 = [E1,E2,G12,G23,nu12]
        mat2 = [10*E1,10*E2,10*G12,10*G23,nu12]
        mat3 = [0.001*E1,0.001*E2,0.001*G12,0.001*G23,nu12]

        mat_vec = [mat1,mat2,mat3]

        ## density property
        rho = 8.0
        rho_vec = [10.0*rho,rho,0.01*rho]

        ## von Mise yield strength
        stress_y = 0.215*1e9
        stress_y_vec = [10.0*stress_y,stress_y,0.00001*stress_y]

        mat_cong = [mat_vec,rho_vec,stress_y_vec]
    else:
        mat1 = [E1,E2,G12,G23,nu12]
        mat_vec = [mat1]

        rho = 1.0
        rho_vec = [rho]

        mat_cong = [mat_vec,rho_vec]



    # load
    f_local = 1.0
    f = np.zeros(node_N*dof_per_node)

    side_N = int(np.sqrt(node_N))
    f[(side_N-1)*side_N*dof_per_node+1] = f_local

    angle_vec = []
    for i in xrange(N):
        angle_vec.append(0.0)

    thickness_loc = 2.0
    thickness_vec = []
    for i in xrange(N):
        thickness_vec.append(2.0)

    # convert the data type
    elem_solver = []
    for i in xrange(len(elem)):
        loc_elem = elem[i]
        loc_elem = loc_elem.reshape(1,9)
        loc_elem_solver = []
        for j in xrange(9):
            loc_elem_solver.append(loc_elem[0,j])
        elem_solver.append(loc_elem_solver)



    topo_solver = [node_N,elem_solver]

    comp_prb = comp_solver(topo_solver,BC_dof,f,thickness_vec,scale_size,angle_vec,mat_vec[0])
    comp_prb.glo_stiffness()
    comp_prb.solve()

    uub_vec = abs(comp_prb.u)*3.0
    ulb_vec = -uub_vec

    uub_vec = uub_vec.reshape(node_N,5)
    ulb_vec = ulb_vec.reshape(node_N,5)

    parameter_cong = [ulb_vec,uub_vec]






    fem_cong = [parameter_cong, topoGeo_cong, mat_cong, BC_dof, f]

    # optimize
    #[u_mat,u_anl,dt] = optimization_prb(fem_cong,int_flag,wt_cmpl_flag,N_angle,cut_flag)
    dt = optimization_prb(fem_cong,int_flag,wt_cmpl_flag,N_angle,cut_flag,solution_limit)


    return dt



# N_vec = []
# t_vec_Gpamm = []
#basic_test(N,M,int_flag,wt_cmpl_flag,N_angle,cut_flag,solution_limit)
dt_Gpamm = basic_test(16,1,True,True,4,False,100)
print('time for optimization:',dt_Gpamm)
