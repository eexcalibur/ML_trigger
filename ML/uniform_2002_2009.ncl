begin

data_fid=addfile("Arm_CF_2002_2009.nc","r")

p_srf_aver   = data_fid->p_srf_aver
T     = data_fid->T
q     = data_fid->q
T_adv_h  = data_fid->T_adv_h
q_adv_h  = data_fid->q_adv_h
s_adv_h  = data_fid->s_adv_h
q_adv_v = data_fid->q_adv_v
T_adv_v = data_fid->T_adv_v
s_adv_v = data_fid->s_adv_v
lev = data_fid->lev

u = data_fid->u
v = data_fid->v

data_out = addfile("Arm_CF_2002_2009_uniform.nc", "c")
data_out->p_srf_aver = p_srf_aver
data_out->T = T
data_out->q = q
data_out->T_adv_h = T_adv_h
data_out->q_adv_h = q_adv_h
data_out->s_adv_h = s_adv_h
data_out->T_adv_v = T_adv_v
data_out->q_adv_v = q_adv_v
data_out->s_adv_v = s_adv_v
data_out->lev = lev

data_out->u = u
data_out->v = v

end
