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

np.set_printoptions(threshold=np.nan)


# optimization problem formulation
def optimization_prb(fem_cong,int_flag,wt_cmpl_flag,N_angle,cut_flag,solution_limit):

    # wt_cmpl_flag = 0: weight
    #              = 1: compliance

    # expand the congrugated inputs
    ## fem_cong
    parameter_cong = fem_cong[0]
    topoGeo_cong = fem_cong[1]
    mat_cong = fem_cong[2]
    BC_dof = fem_cong[3]
    f = fem_cong[4]

    ### parameter_cong
    ulb = parameter_cong[0]
    uub = parameter_cong[1]

    ### fem_cong
    node_N = topoGeo_cong[0]
    dof_per_node = topoGeo_cong[1]
    elem = topoGeo_cong[2]
    hvec = topoGeo_cong[3]
    scale_size = topoGeo_cong[4]

    ### mat_cong
    mat_vec = mat_cong[0]
    rho_vec = mat_cong[1]
    if wt_cmpl_flag==0:
        stress_y_vec = mat_cong[2]



    # problem size info
    elem_N = len(elem)
    layer_N = len(hvec)-1

    if wt_cmpl_flag==1:
        choice_N = N_angle
        angle_vec = (np.linspace(0.0,np.pi,choice_N+1))[:choice_N]
    else:
        choice_N = len(mat_vec)




    # construct the local stiffness mat/ local stress oprt
    K = []
    stress = []
    for i in range(elem_N):

        # get the nodes global index
        element2Node_loc = elem[i]

        K.append([])
        stress.append([])

        for j in range(layer_N):

            # get the j th layer z coord
            h_i = hvec[j]
            h_ip1 = hvec[j+1]

            hvec_loc = [h_i,h_ip1]

            K[i].append([])
            stress[i].append([])

            if wt_cmpl_flag==0:

                for k in range(choice_N):

                    # construct the local layer element
                    Evec_loc = mat_vec[k]

                    layer_element = layerElement(Evec_loc,0.0,hvec_loc,scale_size)

                    K[i][j].append(layer_element.stiffnessMat())
                    stress[i][j].append(layer_element.corner_stress())

            else:

                for k in range(choice_N):

                    Evec_loc = mat_vec[0]

                    angle_loc = angle_vec[k]

                    layer_element = layerElement(Evec_loc,angle_loc,hvec_loc,scale_size)

                    K[i][j].append(layer_element.stiffnessMat())
                    stress[i][j].append(layer_element.corner_stress())





    ########## OPtimization ##########################################
    # model
    if wt_cmpl_flag==0:
        m = Model("composite plate weight optimization SOCP")
    else:
        m = Model("composite plate compliance optimization SOCP")



    # add var
    ##
    if (cut_flag==True):
        s = np.zeros(elem_N).tolist()
        for i in range(elem_N):
            s[i] = m.addVar(vtype=GRB.CONTINUOUS,lb=0.0,name='s%i' %i)
        sig_aux = np.zeros((elem_N,5)).tolist()
        for i in range(elem_N):
            for j in range(5):
                sig_aux[i][j] = m.addVar(vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY,name='sig_aux%i%i' %(i,j))


    z = np.zeros((elem_N,layer_N,choice_N)).tolist()

    for i in range(elem_N):
        for j in range(layer_N):
            for k in range(choice_N):

                if (int_flag):
                    z[i][j][k] = m.addVar(vtype=GRB.BINARY,name='z%i%i%i' %(i,j,k))
                else:
                    z[i][j][k] = m.addVar(vtype=GRB.CONTINUOUS,name='z%i%i%i' %(i,j,k))

    ##
    u = np.zeros((node_N,dof_per_node)).tolist()

    for i in range(node_N):
        for j in range(dof_per_node):
            u[i][j] = m.addVar(vtype=GRB.CONTINUOUS,lb=ulb[i,j],ub=uub[i,j],name='u%i%i' %(i,j))

    ##
    psi = np.zeros((elem_N,layer_N,choice_N,dof_per_node*9)).tolist()

    for i in range(elem_N):
        for j in range(layer_N):
            for k in range(choice_N):
                for l in range(dof_per_node*9):

                    psi[i][j][k][l] = m.addVar(vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY,name='psi%i%i%i%i' %(i,j,k,l))

    ##
    # ordered as (sigma_xx,sigma_yy,sigma_xy,sigma_yz,sigma_xz)
    sig = np.zeros((elem_N,layer_N,1,5)).tolist()

    for i in range(elem_N):
        for j in range(layer_N):
            for k in range(1):
                for l in range(5):

                    sig[i][j][k][l] = m.addVar(vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY,name='sig%i%i%i%i' %(i,j,k,l))


    m.update()

    # add obj
    obj = 0.0
    if wt_cmpl_flag==0:

        for i in range(elem_N):
            for j in range(layer_N):
                for k in range(choice_N):

                    obj += z[i][j][k]*rho_vec[k]

    else:
        for i in range(node_N):
            for j in range(dof_per_node):
                if (f[i*dof_per_node+j]!=0.0):

                    obj += u[i][j]*f[i*dof_per_node+j]



    m.setObjective(obj, GRB.MINIMIZE)

    # add constraints
    #
    if (cut_flag==True):
        print('con0')
        for i in range(elem_N):
            stress_rt = stress_y_vec[1]/stress_y_vec[0]
            print('conic cut',stress_rt)
            m.addConstr(s[i]==stress_rt-(stress_rt-1.0)*z[0][0][0])

        mat_vM_sqrt1 = (np.sqrt(6.0)+np.sqrt(2.0))/4.0
        mat_vM_sqrt2 = (-np.sqrt(6.0)+np.sqrt(2.0))/4.0
        mat_vM_sqrt3 = np.sqrt(3.0/2.0)

        for i in range(elem_N):
            m.addConstr(sig_aux[i][0] == mat_vM_sqrt1*sig[i][0][0][0]+mat_vM_sqrt2*sig[i][0][0][1])
            m.addConstr(sig_aux[i][1] == mat_vM_sqrt2*sig[i][0][0][0]+mat_vM_sqrt1*sig[i][0][0][1])
            m.addConstr(sig_aux[i][2] == mat_vM_sqrt3*sig[i][0][0][2])
            m.addConstr(sig_aux[i][3] == mat_vM_sqrt3*sig[i][0][0][3])
            m.addConstr(sig_aux[i][4] == mat_vM_sqrt3*sig[i][0][0][4])

        for i in range(elem_N):
            m.addQConstr(sig_aux[i][0]*sig_aux[i][0]+\
            sig_aux[i][1]*sig_aux[i][1]+\
            sig_aux[i][2]*sig_aux[i][2]+\
            sig_aux[i][3]*sig_aux[i][3]+\
            sig_aux[i][4]*sig_aux[i][4]\
            <=s[i]*s[i])



    ##
    print('con1.1')
    f_sum = np.zeros(node_N*dof_per_node).tolist()
    for i in range(elem_N):

        element2Node_loc = elem[i]

        print('++++++++',i)

        f_loc = np.zeros(9*dof_per_node).tolist()

        for j in range(layer_N):
            for k in range(choice_N):

                Kijk = K[i][j][k]

                for p in range(9*dof_per_node):
                    for n in range(9*dof_per_node):

                        f_loc[p] += Kijk[p,n]*psi[i][j][k][n]

        for i in range(3): ######################bug!!!!
            for j in range(3):

                glo_index = element2Node_loc[i,j]
                loc_index = 3*i+j

                for k in range(dof_per_node):

                    f_sum[glo_index*dof_per_node+k] += f_loc[loc_index*dof_per_node+k]


    #Direct set disp zero???????????????????????????????????
    for ind_BC_loc in BC_dof:
        for j in range(dof_per_node):

            m.addConstr(u[ind_BC_loc][j]==0.0)

    print('con1.3')
    for i in range(node_N):
        for j in range(dof_per_node):
            if(i>=len(BC_dof)):
                index_loc = i*dof_per_node+j
                m.addConstr(f_sum[index_loc]==f[index_loc])



    #
    print('con2')
    for i in range(elem_N):

        elem_loc = elem[i]

        for j in range(layer_N):
            for k in range(choice_N):
                for ii in range(3):
                    for jj in range(3):

                        glo_index = elem_loc[ii,jj]
                        u_loc = u[glo_index][:]

                        loc_index = ii*3+jj
                        psi_loc = psi[i][j][k][loc_index*dof_per_node:(loc_index+1)*dof_per_node]

                        for kk in range(dof_per_node):

                            m.addConstr(psi_loc[kk]<=uub[glo_index,kk]*z[i][j][k])
                            m.addConstr(psi_loc[kk]>=ulb[glo_index,kk]*z[i][j][k])
                            m.addConstr(psi_loc[kk]<=u_loc[kk]-ulb[glo_index,kk]*(1-z[i][j][k]))
                            m.addConstr(psi_loc[kk]>=u_loc[kk]-uub[glo_index,kk]*(1-z[i][j][k]))

    #
    if wt_cmpl_flag==0:
        print('con3')
        for s in range(1):
            for i in range(elem_N):
                for j in range(layer_N):

                    stress_sum = (np.zeros(5)).tolist()

                    for k in range(choice_N):

                        stress_oprt_loc = stress[i][j][k][s]

                        for dim in range(5):
                            for l in range(9*dof_per_node):

                                stress_sum[dim] += (stress_oprt_loc[dim,l]*psi[i][j][k][l])/stress_y_vec[0]

                    for k in range(5):

                        m.addConstr(stress_sum[k]==sig[i][j][s][k])



        for i in range(elem_N):
            for j in range(layer_N):
                for s in range(1):# NOTICE!!!
                #for s in range(8):

                    sig11 = sig[i][j][s][0]
                    sig22 = sig[i][j][s][1]
                    sig12 = sig[i][j][s][2]
                    sig23 = sig[i][j][s][3]
                    sig13 = sig[i][j][s][4]

                    sig_vm = 0.5*((sig11-sig22)*(sig11-sig22)+(sig22)*(sig22)+(-sig11)*(-sig11))+\
                    3*(sig12*sig12+sig23*sig23+sig13*sig13)

                    stress_yield_sq = 0.0

                    for k in range(choice_N):

                        stress_yield_sq += z[i][j][k]*(stress_y_vec[k]/stress_y_vec[0])**2

                    m.addQConstr(sig_vm<=stress_yield_sq)


    #
    for i in range(elem_N):
        for j in range(layer_N):

            sum_z = 0.0

            for k in range(choice_N):
                sum_z += z[i][j][k]

            m.addConstr(sum_z==1.0)

    # m.addConstr(z[0][0][1]==1.0)
    # m.addConstr(z[1][0][1]==1.0)
    # m.addConstr(z[2][0][3]==1.0)
    # m.addConstr(z[3][0][3]==1.0)
    # m.addConstr(z[4][0][0]==1.0)
    # m.addConstr(z[5][0][1]==1.0)
    # m.addConstr(z[6][0][1]==1.0)
    # m.addConstr(z[7][0][3]==1.0)
    # m.addConstr(z[8][0][0]==1.0)
    # m.addConstr(z[9][0][1]==1.0)
    # m.addConstr(z[10][0][1]==1.0)
    # m.addConstr(z[11][0][3]==1.0)
    # m.addConstr(z[12][0][3]==1.0)
    # m.addConstr(z[13][0][3]==1.0)
    # m.addConstr(z[14][0][2]==1.0)
    # m.addConstr(z[15][0][3]==1.0)

    # optimize
    def mycallback(model, where):
        if (1==0):
            if where == GRB.Callback.MIPNODE:
                print "MIPNODE_OBJBND",model.cbGet(GRB.Callback.MIPNODE_OBJBND)

            if where == GRB.Callback.MIPSOL_SOL:
                print "MIPSOL_OBJBST",model.cbGet(GRB.Callback.MIPSOL_OBJBST)
                print "*****************MIPSOL_SOLCNT",model.cbGet(GRB.Callback.MIPSOL_SOLCNT)

            if where == GRB.Callback.MIP:
                print "MIP_CUTCNT",model.cbGet(GRB.Callback.MIP_CUTCNT)

        if(1==1):
            if where == GRB.Callback.MIP:
                solcnt = model.cbGet(GRB.Callback.MIP_SOLCNT)
                print solcnt






    #m.params.Method = 3
    #m.params.Heuristics = 0.5
    m.params.SolutionLimit = solution_limit
    start_time = time.time()
    m.optimize(mycallback)
    #m.optimize()
    end_time = time.time()
    dt = end_time-start_time
    print('dt!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11',dt)


    for v in m.getVars():
        print('%s %g' % (v.varName, v.x))


    #return [u_mat,u_anl,dt]
    return dt
