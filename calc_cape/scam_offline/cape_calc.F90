program cape_calc
    use netcdf
    use zm_conv, only: buoyan, buoyan_dilute,zm_convi,buoyan_dilute_1
    use shr_kind_mod,    only: r8 => shr_kind_r8
    use physconst,       only: cpair, epsilo, gravit, latice, latvap, tmelt, rair, &
                               cpwv, cpliq, rh2o
    implicit none
    
    integer status,NCID,i,j
    integer dimID,nlevOBS,ntimeOBS,varid
    real(r8), allocatable :: tmp_1d(:)
    real(r8), allocatable :: tmp_2d(:, :)
    real(r8), allocatable :: levOBS( : )
    real(r8), allocatable :: pfOBS( : )
    real(r8), allocatable :: psOBS( : )
    real(r8), allocatable :: capeOBS( : )
    real(r8), allocatable :: mxOBS( : )
    real(r8), allocatable :: qOBS( :, : )
    real(r8), allocatable :: sqOBS( :,: )
    real(r8), allocatable :: tOBS( :,: )
    real(r8), allocatable :: divtHOBS( :,: )
    real(r8), allocatable :: divqHOBS( :,: )
    real(r8), allocatable :: divtVOBS( :,: )
    real(r8), allocatable :: divqVOBS( :,: )
    real(r8), allocatable :: zOBS( :,: )

    real(r8) cape(1)
    real(r8) tl(1)
    real(r8) tpert(1)
    integer lcl(1)                  
    integer lel(1)                  
    integer lon(1)                  
    integer maxi(1)
    real(r8) pblt(1)
    integer msg
    real(r8) tp(1,45)
    real(r8) qstp(1,45)
    real(r8) rl
    real(r8) rgas
    real(r8) grav
    real(r8) cpres

    call zm_convi()

    rl = latvap
    rgas = rair 
    grav = gravit 
    cpres = cpair 
    tpert = 0.0
    pblt = 25
    msg = 1

    !status =NF90_OPEN("continuous_at_goamazon.nc",NF90_NOWRITE,NCID )
    status =NF90_OPEN("Arm_CF_1999_2009_uniform.nc",NF90_NOWRITE,NCID )
    !status =NF90_OPEN("goamazon_2014_2015.nc",NF90_NOWRITE,NCID )
    !status =NF90_OPEN("mao_IOP1_20140201.nc",NF90_NOWRITE,NCID )

    status = NF90_INQ_DIMID( ncid, 'lev', dimID )
    status = nf90_inquire_dimension( ncid, dimID, len=nlevOBS )
    status = NF90_INQ_DIMID( ncid, 'time', dimID )
    status = nf90_inquire_dimension( ncid, dimID, len=ntimeOBS )

    allocate(tmp_1d(nlevOBS))
    allocate(levOBS(nlevOBS))
    allocate(pfOBS(nlevOBS+1))
    allocate(psOBS(ntimeOBS))
    allocate(capeOBS(ntimeOBS))
    allocate(mxOBS(ntimeOBS))
    allocate(tmp_2d(nlevOBS,ntimeOBS))
    allocate(qOBS(nlevOBS,ntimeOBS))
    allocate(sqOBS(nlevOBS,ntimeOBS))
    allocate(tOBS(nlevOBS,ntimeOBS))
    allocate(divtHOBS(nlevOBS,ntimeOBS))
    allocate(divqHOBS(nlevOBS,ntimeOBS))
    allocate(divtVOBS(nlevOBS,ntimeOBS))
    allocate(divqVOBS(nlevOBS,ntimeOBS))
    allocate(zOBS(nlevOBS,ntimeOBS))

    !lev
    status = nf90_inq_varid( ncid, 'lev', varid ) !mb
    status = nf90_get_var (ncid, varid, tmp_1d)
    levOBS = tmp_1d(nlevOBS:1:-1)

    !surface pressure
    status = nf90_inq_varid( ncid, 'p_srf_aver', varid) !mb
    status = nf90_get_var (ncid, varid, psOBS)

    !mixing ratio to specific humidity
    status = nf90_inq_varid( ncid, 'q', varid   ) !g/kg
    status = nf90_get_var (ncid, varid, tmp_2d)
    qOBS = tmp_2d(nlevOBS:1:-1, :) / 1000.0
    sqOBS = qOBS / (qOBS + 1)

    !temperature
    status = nf90_inq_varid( ncid, 'T', varid   ) !K
    status = nf90_get_var (ncid, varid, tmp_2d)
    tOBS = tmp_2d(nlevOBS:1:-1, :)

    !horizontal q adv
    status = nf90_inq_varid( ncid, 'q_adv_h', varid   ) !g/kg/hour
    status = nf90_get_var (ncid, varid, tmp_2d)
    divqHOBS = tmp_2d(nlevOBS:1:-1,:) / 3600.0 / 1000.0

    !horizontal t adv
    status = nf90_inq_varid( ncid, 'T_adv_h', varid   ) !K/hour
    status = nf90_get_var (ncid, varid, tmp_2d)
    divtHOBS = tmp_2d(nlevOBS:1:-1,:) / 3600.0

    !vertical q adv
    status = nf90_inq_varid( ncid, 'q_adv_v', varid   ) !g/kg/hour
    status = nf90_get_var (ncid, varid, tmp_2d)
    divqVOBS = tmp_2d(nlevOBS:1:-1,:) / 3600.0 / 1000.0

    !vertical t adv
    status = nf90_inq_varid( ncid, 'T_adv_v', varid   ) !K/hour
    status = nf90_get_var (ncid, varid, tmp_2d)
    divtVOBS = tmp_2d(nlevOBS:1:-1,:) / 3600.0

    !z
    do j=1, ntimeobs
    do i=1, nlevobs
       !zOBS(i,1) = 288/6.5 * (1 - (levOBS(i)/1013.25) ** (0.287 * 6.5/9.8)) * 1000.0
       zOBS(i,j) = tOBS(nlevobs-1,j)/6.5 * (1 - (levOBS(i)/psOBS(j)) ** (0.287 * 6.5/9.8)) * 1000.0
       !zOBS(i,1) = 287.05 * tOBS(i,1) / 9.8 * log(psOBS(1)/levobs(i))
    end do
    end do

    !pf
    do i=1, nlevobs
       pfOBS(i) = levOBS(i) - 12.5
    end do
    pfOBS(nlevobs+1) = levOBS(nlevobs) + 12.5


    !print debug
    write(*,*)"debug p_obs"
    write(*,*)levOBS
    write(*,*)"debug z_obs"
    write(*,*)zOBS(:,3)
    write(*,*)"debug t_obs"
    write(*,*)tOBS(:,3)
    write(*,*)"debug q_obs"
    write(*,*)qOBS(:,3)

    do i=1,ntimeobs
       call buoyan_dilute(1   ,1    , &
                    qOBS(1:nlevobs,i),tOBS(1:nlevobs, i),levOBS(1:nlevobs),zOBS(1:nlevobs,i),pfOBS(1:nlevobs+1)       , &
                    tp      ,qstp    ,tl      ,rl      ,cape     , &
                    pblt    ,lcl     ,lel     ,lon     ,maxi     , &
                    rgas    ,grav    ,cpres   ,msg     , &
                    tpert   )
        capeOBS(i) = cape(1)
        mxOBS(i) = maxi(1)
     end do

     open(1001, file="goamazon_cape.txt")
     do i=1, ntimeobs
        write(1001,"(F16.4)") capeOBS(i)
     end do
     close(1001)

    do i=1,ntimeobs
    do j=1,nlevobs
       qobs(j,i) = qobs(j,i) + (divqHobs(j,i) + divqVobs(j,i)) * 3600.0 
       tobs(j,i) = tobs(j,i) + (divtHobs(j,i) - divtVobs(j,i)) * 3600.0 
       if (qobs(j,i) < 0) then
           qobs(j,i) = 1.0e-12
       end if
    end do
    end do

    do i=1,ntimeobs
       maxi = mxobs(i)
       call buoyan_dilute_1(1   ,1    , &
                    qOBS(1:nlevobs,i),tOBS(1:nlevobs, i),levOBS(1:nlevobs),zOBS(1:nlevobs,i),pfOBS(1:nlevobs+1)       , &
                    tp      ,qstp    ,tl      ,rl      ,cape     , &
                    pblt    ,lcl     ,lel     ,lon     ,maxi     , &
                    rgas    ,grav    ,cpres   ,msg     , &
                    tpert   )
        capeOBS(i) = (cape(1) - capeOBS(i)) 
     end do
     open(1001, file="goamazon_dcape.txt")
     do i=1, ntimeobs
        write(1001,"(F16.4)") capeOBS(i)
     end do
     close(1001)
end program cape_calc
