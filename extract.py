import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path



COMBINE_DIR = Path("/r003/nazhang/cmaq/tiqu/combine/ea/")
OUTPUT_DIR = Path("./data")
OUTPUT_DIR.mkdir(exist_ok=True)

MET_DIR=Path("/r002/kjgong/MCIP/ea")
GRIDCRO2D = next(MET_DIR.glob("GRIDCRO2D*"))

gridcro2d = xr.open_dataset(GRIDCRO2D, engine="scipy")

lon = gridcro2d['LON'][0, 0, :,:]
lat = gridcro2d['LAT'][0, 0, :,:]
np.savetxt(OUTPUT_DIR/"lon.txt",lon,fmt="%.4f")
np.savetxt(OUTPUT_DIR/"lat.txt",lat,fmt="%.4f")

NOX = ["1","0.9","0.8","0.65","0.5"]
VOC = ["1","0.9","0.8","0.65","0.5"]

def get_datetime(data,**kwargs):
    # 时间序列
    dt = data["TFLAG"][:, -1, :].to_pandas()
    dt["datetime"] = dt[0].astype(str)+dt[1].astype(str).str.zfill(6)
    dt.index = pd.to_datetime(dt["datetime"], format="%Y%j%H%M%S", utc=True,**kwargs)
    datetime = dt.index.tz_convert("Asia/Shanghai").tz_localize(None)
    return datetime

for n in NOX:
    for v in VOC:
        CASE = f"nox_{n}_voc_{v}"
        print(CASE)
        FILE = COMBINE_DIR/CASE/"ea_COMBINE_ACONC_v52_intel_saprc07tic_ae6i_aq_2020.0823-0925.nc"
        data = xr.open_dataset(FILE)
        dt = get_datetime(data)
        data.coords["TSTEP"] = ("TSTEP",dt)


        #for O3
        sel_o3  = data.O3.loc["2020-09-05":"2020-09-10"]
        mda8 = sel_o3.rolling(TSTEP=8,min_periods=6).mean().resample(TSTEP="1D").max()
        mat = mda8.mean(axis=(0,1))
        mat = mat*48/22.4
        np.savetxt(OUTPUT_DIR/f"{CASE}.csv",mat,fmt="%.4f")
