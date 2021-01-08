import xarray as xr
import xesmf as xe
import pyspec.spectrum as spec
import sys, os
import numpy as np
    
def regrid(var, lons, lats, which):
    # rename coords for use with xESMF
    lonkey = [coord for coord in var.coords if "lon_" in coord][0]
    latkey = [coord for coord in var.coords if "lat_" in coord][0]
    var = var.rename({lonkey: "lon", latkey: "lat"})

    # whether inputs are
    if which == "pairs":
        locstream_out = True
    elif which == "grid":
        locstream_out = False

    # set up for output
    varint = xr.Dataset({"lat": (["y", "x"], lats), "lon": (["y", "x"], lons)})

    # Calculate weights.
    regridder = xe.Regridder(var, varint, "bilinear",  locstream_out=locstream_out)

    # Perform interpolation
    varint = regridder(var, keep_attrs=True)
    return varint, regridder

#def calculateKEspec(uvar, vvar, dx, dy):
#    nspec = 0
#    for yd in uvar.yearday.values:
#        print(yd)
#        for j in uvar.s_rho.values:
#            spec2d_u = spec.TWODimensional_spec(uvar.sel(s_rho=j, yearday=yd), dx, dy);
#            spec2d_v = spec.TWODimensional_spec(vvar.sel(s_rho=j, yearday=yd), dx, dy);
#            if nspec == 0:
#                spec_ke = 0.5*(spec2d_u.ispec + spec2d_v.ispec);
#            else:
#                spec_ke += 0.5*(spec2d_u.ispec + spec2d_v.ispec);
#            nspec += 1 # increment counter
#    
#    spec_ke = spec_ke/nspec
#    print(nspec)
#    return spec_ke, spec2d_u.ki


def calculateKEspec(uvar, vvar, dx, dy):
    def specfunc(uv, dx,dy):
        spec2d = spec.TWODimensional_spec(uv, dx,dy)
        return spec2d.ispec
    
    out_u = xr.apply_ufunc(specfunc, 
               uvar.load(),
               dx, 
               dy,
               input_core_dims=[['y', 'x'],[],[]],
               output_core_dims=[['ki']],
               #exclude_dims=set(['x'],),
               output_dtypes=[np.float64],
               vectorize=True)
    out_v = xr.apply_ufunc(specfunc, 
               vvar.load(),
               dx, 
               dy,
               input_core_dims=[['y', 'x'],[],[]],
               output_core_dims=[['ki']],
               #exclude_dims=set(['x'],),
               output_dtypes=[np.float64],
               vectorize=True)
    
    kespec_avg = (0.5*(out_u + out_v)).mean(['yearday', 's_rho'])
    # One hacky last call to spectrum to get the frequencies...
    spec2d = spec.TWODimensional_spec(uvar[0,0,:,:], dx,dy)
    return kespec_avg, spec2d.ki


#def calculateKEspec1(uvar,vvar, dx, dy):
#    nspec = 0
#    nd, nz, nx, ny = uvar.shape
#    for yd in range(0,nd):
#        print(yd)
#        for z in range(0, nz):
#            spec2d_u = spec.TWODimensional_spec(uvar[yd, z, :,:], dx, dy);
#            spec2d_v = spec.TWODimensional_spec(vvar[yd, z, :,:], dx, dy);
#            if nspec == 0:
#                spec_ke = 0.5*(spec2d_u.ispec + spec2d_v.ispec );
#            else:
#                spec_ke += 0.5*(spec2d_u.ispec + spec2d_v.ispec);
#            nspec += 1 # increment counter
    
#    spec_ke = spec_ke/nspec
#    print(nspec)
#    return spec_ke, spec2d_u.ki


