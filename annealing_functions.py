# Annealing helper function





def Generate_gp_Cooling_Schedule(cooling_param:list,steps:int)->list[float]:
    """
    generates a cooling schedule from the cooling_param as a list
    """
    T=[]
    for step in range(1,steps+1):

            c = cooling_param[1]
            T.append(c/(np.log(1+pow(step, 2.22))))