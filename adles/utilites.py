def binary_search(time, time_list):
    if len(time_list) == 2:
        if time<(time_list[0]+time_list[1])/2:
            return time_list[0]
        else:
            return time_list[1]
    elif len(time_list) == 3:
        if time<(time_list[0]+time_list[1])/2:
            return time_list[0]
        elif (time_list[0]+time_list[1])/2<= time and time < (time_list[1] + time_list[2])/2:
            return time_list[1]
        else:
            return time_list[2]
            
    ls = []
    if time < time_list[len(time_list)//2]:
        ls = time_list[0:len(time_list)//2]  
    else:
        ls = time_list[len(time_list)//2:]
        
    return binary_search(time, ls)