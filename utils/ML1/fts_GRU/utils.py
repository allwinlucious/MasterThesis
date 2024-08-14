
import numpy as np



count_to_deg = 360/(2**24)
radian_to_deg = 180/np.pi



def number_of_motor_turns(mot_enc,load_enc,offset_m,offset_l,gear_ratio = 101):
    return ((load_enc - offset_l)*gear_ratio + offset_m)//2**24 - mot_enc//2**24
    
def multiturn_compensation(df,offset_m,offset_l,gear_ratio=101):
    ##check if starting position is at 0°
    if np.isclose(0,-6.47897e-05,atol=1e-04): 
        n = number_of_motor_turns(df.encoder_motorinc_3[0],df.encoder_loadinc_3[0],offset_m,offset_l,gear_ratio)
        if n == 0:
            #print("no offset compensation required")
            return df
        else:
            #print("compensating offset")
            df.encoder_motorinc_3 += n * 2**24
            return df
    else:
        #print("starting data point is not near 0°")
        pass

def offset_compensation(df,offset_m,offset_l):
    #reduce offsets from df motorenc and load enc
    df.encoder_motorinc_3 -= offset_m
    df.encoder_loadinc_3 -= offset_l
    return df
