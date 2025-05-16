library(terra)
library(gstlearn)

Ngd=256
df=c(0.05,0.05)*Ngd
xmin=62.5
xmax=xmin+df[1]
ymin=-35
ymax=ymin+df[2]

# Load NetCDF as a raster (works well for LST, NDVI, etc.)
r <- rast("raw/ct5km_ssta_v3.1_20250101.nc")
window(r)=ext(c(xmin,xmax,ymin,ymax))

# Check available layers/variables
names(r)

# Plot the first layer
plot(r[[1]])
dim(r[[1]])

dat_anom=matrix(as.matrix(r[[1]]),Ngd)

#---------------------------


cm=read.csv("../cloud/cloud_mask_cd1.csv",sep=",")

plot(cm$Long,cm$Lat,cex=0.1)
abline(v=c(xmin,xmax),col="red")
abline(h=c(ymin,ymax),col="red")

#---------------------------------------

dbcm=fromTL(cm)
dbcm$setLocators(c("Long","Lat"),ELoc_X())

xgd=unique(crds(r[[1]])[,1])
ygd=unique(crds(r[[1]])[,2])
dbMigrate=DbGrid_create(nx=c(length(xgd),length(ygd)),
                         dx=abs(c(xgd[2]-xgd[1],ygd[2]-ygd[1])),
                         x0=c(min(xgd),min(ygd)),flagAddSampleRank = F)

migrate(dbcm,dbMigrate,name = "Mask",flag_fill = T)

mask_grid=matrix(dbMigrate["Migrate"],Ngd)
mask_grid=(mask_grid>0)+0
#---------------------------------------

image.plot(dat_anom)

uu=mask_grid+0
uu[uu==0]=NA
image.plot(dat_anom*uu)

#------------------

filename="temp_anom.h5"
h5createFile(filename)
h5createDataset(filename, "data", rev(c(1,Ngd,Ngd)), maxdims = rev(c(nex,Ngd,Ngd)),
                chunk=rev(c(1,Ngd,Ngd)),native=TRUE)
h5createDataset(filename, "mask", rev(c(1,Ngd,Ngd)), maxdims = rev(c(nex,Ngd,Ngd)),
                chunk=rev(c(1,Ngd,Ngd)),native=TRUE)
h5write(dat_anom, file=filename, name="data", index=list(1:Ngd,1:Ngd,1))
h5write(mask_grid, file=filename, name="mask", index=list(1:Ngd,1:Ngd,1))
h5closeAll()

uu=h5dump(filename)

#---------------------------------------
