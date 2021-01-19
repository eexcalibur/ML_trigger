begin

data_fid=addfile("Arm_CF_1999_2001.nc","r")

p_srf_aver   = data_fid->Area_Mean_Ps(:,0,0)
T     = data_fid->Temp(:,:,0,0)
q     = data_fid->H2O_Mixing_Ratio(:,:,0,0)
T_adv_h  = data_fid->Horizontal_Temp_Advec(:,:,0,0)
q_adv_h  = data_fid->Horizontal_q_Advec(:,:,0,0)
s_adv_h  = data_fid->Horizontal_s_Advec(:,:,0,0)
q_adv_v = data_fid->Vertical_q_Advec(:,:,0,0)
T_adv_v = data_fid->Vertical_T_Advec(:,:,0,0)
s_adv_v = data_fid->Vertical_s_Advec(:,:,0,0)
lev = data_fid->lev

u = data_fid->u_wind(:,:,0,0)
v = data_fid->v_wind(:,:,0,0)


data_out = addfile("Arm_CF_1999_2001_uniform.nc", "c")
data_out->p_srf_aver = p_srf_aver
data_out->T = T(:,::-1)
data_out->q = q(:,::-1)
data_out->T_adv_h = T_adv_h(:,::-1)
data_out->q_adv_h = q_adv_h(:,::-1)
data_out->s_adv_h = s_adv_h(:,::-1)
data_out->T_adv_v = T_adv_v(:,::-1)
data_out->q_adv_v = q_adv_v(:,::-1)
data_out->s_adv_v = s_adv_v(:,::-1)
data_out->lev = lev(::-1)
data_out->u = u(:,::-1)
data_out->v = v(:,::-1)

end
