library(terra)
library(rhdf5)
library(gstlearn)
library(minigst)
library(fields)

## Old file
old=h5dump("temp_anom.h5")
Ngd=256
df=c(0.05,0.05)*256


# Load NetCDF as a raster (works well for LST, NDVI, etc.)
r <- rast("raw/ct5km_ssta_v3.1_20250101.nc")

## Selected zones
tab_x=rbind(c(62.5,-35), ## From old file
  c(-140,8),
  c(-35,40)
)

## Plot
png(filename = "map_ssta.png",width=8,height = 5,res=600,units = "in")
plot(r[[1]])
rect(tab_x[,1],tab_x[,2],tab_x[,1]+df[1],tab_x[,2]+df[2])
text(tab_x[,1]+df[1]/2,tab_x[,2]+df[2]/2,-1+1:nrow(tab_x))
dev.off()


## Mask sizes
sizes=c(300,600,1000,2000)

## Name of the export file
filename="data_temp_anomb.h5"
if (file.exists(filename)) {
  file.remove(filename)
}
h5createFile(filename)
h5createDataset(filename, "data", rev(c(4,Ngd,Ngd)),
                chunk=rev(c(1,Ngd,Ngd)),native=TRUE)
h5createDataset(filename, "mask", rev(c(4,Ngd,Ngd)), 
                chunk=rev(c(1,Ngd,Ngd)),native=TRUE)
for(k in 1:length(sizes)){
  h5createDataset(filename, paste0("mask",sizes[k]), rev(c(4,Ngd,Ngd)), 
                  chunk=rev(c(1,Ngd,Ngd)),native=TRUE)
}

#---------------------------

## Fill data

h5write(old$data[,,1], file=filename, name="data", index=list(1:Ngd,1:Ngd,1))
for(i in 2:nrow(tab_x)){
  xmin=tab_x[i,1]
  xmax=xmin+df[1]
  ymin=tab_x[i,2]
  ymax=ymin+df[2]
  
  window(r)=ext(c(xmin,xmax,ymin,ymax))
  
  # Plot the first layer
  plot(r[[1]],main=paste0(xmin,",",ymax,",",xmax,",",ymin))
  
  dat_anom=matrix(as.matrix(r[[1]]),256)
  print(dim(dat_anom))
  
  h5write(dat_anom, file=filename, name="data", index=list(1:Ngd,1:Ngd,i+1))
  
}

#---------------------------

## Fill masks

mask_grid=old$mask[,,1]
h5write(mask_grid, file=filename, name="mask", index=list(1:Ngd,1:Ngd,1))
id_sel=which(mask_grid!=0,arr.ind = T)
i=1
for(k in 1:length(sizes)){
  set.seed(42*i+k)
  id_samp=sample(1:nrow(id_sel),sizes[k])
  mask_grid_sel=matrix(0,Ngd,Ngd)
  mask_grid_sel[id_sel[id_samp,]]=1
  h5write(mask_grid_sel, file=filename, name=paste0("mask",sizes[k]), index=list(1:Ngd,1:Ngd,i))
}
for(i in 2:length(sizes)){
  
  cm=read.csv(paste0("../cloud/cloud_mask_cd",i,".csv"),sep=",")
  
  #---------------------------------------
  
  dbcm=fromTL(cm)
  dbcm$setLocators(c("Long","Lat"),ELoc_X())
  xbar=c(mean(cm$Long),mean(cm$Lat))
  
  xgd=seq(from=xbar[1]-df[1]/2,to=xbar[1]+df[1]/2,by=0.05)[1:256]
  ygd=seq(from=xbar[2]-df[2]/2,to=xbar[2]+df[2]/2,by=0.05)[1:256]
  xgrid=expand.grid(x=xgd,
                    y=ygd)
  dbMigrate=DbGrid_create(nx=c(length(xgd),length(ygd)),
                          dx=abs(c(xgd[2]-xgd[1],ygd[2]-ygd[1])),
                          x0=c(min(xgd),min(ygd)),flagAddSampleRank = F)
  
  
  
  mask_grid=matrix(dbMigrate["Migrate"],256)
  mask_grid=(mask_grid>0)+0
  
  h5write(mask_grid, file=filename, name="mask", index=list(1:Ngd,1:Ngd,i))
  
  id_sel=which(mask_grid!=0,arr.ind = T)
  for(k in 1:length(sizes)){
    set.seed(42*i+k)
    id_samp=sample(1:nrow(id_sel),sizes[k])
    mask_grid_sel=matrix(0,Ngd,Ngd)
    mask_grid_sel[id_sel[id_samp,]]=1
    h5write(mask_grid_sel, file=filename, name=paste0("mask",sizes[k]), index=list(1:Ngd,1:Ngd,i))
  }
  
  
}

h5closeAll()
