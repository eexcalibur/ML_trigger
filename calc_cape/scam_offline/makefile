LDFLAGS =
LDFLAGS +=
FCFLAGS = -O0  -I/opt/cray/pe/netcdf/4.6.3.2/INTEL/19.0/include -L/opt/cray/pe/netcdf/4.6.3.2/INTEL/19.0/lib -lnetcdff -lnetcdf
FCFLAGS +=
FC = ifort

cape: cape_calc.F90 zm_conv.o shr_kind_mod.o physconst.o wv_sat_methods.o
	$(FC) $(FCFLAGS) $^ -o $@ $(LDFLAGS)

zm_conv.o: zm_conv.F90 shr_kind_mod.o physconst.o wv_sat_methods.o
        # `$<' is shorthand for the first dependency
        # in the list. This is used instead of `$^'
        # because only the `.f90' file needs to
        # be compiled.
	$(FC) $(FCFLAGS) -c $<

wv_sat_methods.o: wv_sat_methods.F90
	$(FC) $(FCFLAGS) -c $<

physconst.o: physconst.F90 shr_kind_mod.o shr_const_mod.o
	$(FC) $(FCFLAGS) -c $<

shr_const_mod.o: shr_const_mod.F90 shr_kind_mod.o
	$(FC) $(FCFLAGS) -c $<

shr_kind_mod.o: shr_kind_mod.F90
	$(FC) $(FCFLAGS) -c $<


.PHONEY: clean
clean:
	@rm -rf *.o *.mod
