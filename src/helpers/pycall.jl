using PyCall

export xr, np
# Import OGGM sub-libraries in Julia
netCDF4 = pyimport("netCDF4")
cfg = pyimport("oggm.cfg")
utils = pyimport("oggm.utils")
workflow = pyimport("oggm.workflow")
tasks = pyimport("oggm.tasks")
graphics = pyimport("oggm.graphics")
bedtopo = pyimport("oggm.shop.bedtopo")
MBsandbox = pyimport("MBsandbox.mbmod_daily_oneflowline")

# Essential Python libraries
np = pyimport("numpy")
xr = pyimport("xarray")
# matplotlib = pyimport("matplotlib")
# matplotlib.use("Qt5Agg") 
