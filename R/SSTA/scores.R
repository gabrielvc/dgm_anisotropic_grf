###########################################
# Prediction comparison
###########################################


library(NSGP)
library(Matrix)
library(rhdf5)
library(scoringRules)
library(fields)

# Package for viridis color palette
library(viridis)
viridis_colors <- viridis(128)

plot_sim=function(x,zlim=NULL,...){
  if(!is.null(zlim)){
    xplt=x
    xplt[x<zlim[1]]=zlim[1]
    xplt[x>zlim[2]]=zlim[2]
  }else{
    xplt=x
  }
  image.plot(matrix(xplt,nrow = Ngd)[,Ngd:1],zlim=zlim,...)
}

path_files="vecchia_temp/"

method_name=c("VMCMC","PriorVAE",
              # "VAE_finer",
              "MGDM")


#--------------------------------------------------------------

dat=h5dump(paste0(path_files,"data_temp_anomb.h5"))

Ngd=256

#------------------------------------

n_cases=3

crps_tab=data.frame(matrix(NA,nrow = 3,ncol=2*length(method_name)))
names(crps_tab)=c(paste0(method_name,"_mean"),paste0(method_name,"_sd"))
es_tab=crps_tab

#-------------------------------------

nsamples=78

id_case=1
n_cval=32

mObs_vec=NULL
sdObs_vec=NULL
for(id_case in 1:3){
  z_complete=c(dat$data[, ,id_case])
  sel_obs=which(c(dat$mask[, ,id_case])==1)
  sel_pred=which(c(dat$mask[, ,id_case])==0)
  
  set.seed(42)
  shulfled_indices=sample(1:length(sel_pred),length(sel_pred))
  cval_indices=list()
  for (id_val in 0:(n_cval-1)) {
    ipp=floor(seq(from=0,to=length(sel_pred),length.out=n_cval+1))
    cval_indices[[id_val+1]]=shulfled_indices[(ipp[id_val+1]+1):(ipp[id_val+2])]
  }
  
  z_mask=rep(NA,length(z_complete)); z_mask[sel_obs]=1
  mObs=mean(z_complete[sel_obs])
  sdObs=sd(z_complete[sel_obs])
  mObs_vec=c(mObs_vec,mObs)
  sdObs_vec=c(sdObs_vec,sdObs)
  
  z_complete=(z_complete-mObs)/sdObs
  z_obs=z_complete[sel_obs]
  z_unobs=z_complete[sel_pred]
  
  cat("Number of observations:",length(z_obs),"\n")

  filenames=c(paste0("post_",id_case,id_case,"_v16_s1000/data_new.h5"),
              # paste0("cloud_inpainting_",id_case-1,"/prior_vae/data.h5"),
              paste0("cloud_inpainting_",id_case-1,"/prior_vae_finer/data.h5"),
              paste0("cloud_inpainting_",id_case-1,"/mgdm_two_times_gibbs/data.h5"))
  dataname=rev(c("samples","samples",
                 # "samples",
                 "post"))

  ttl=paste0("~/Documents/Work/doc/diff_post_gauss_paper/images/map_crps_case_",id_case,".png")
  png(ttl,width = 10,height = 4,res=600,units = "in",bg = "transparent")
  par(mfrow=c(1,3))
  for(id_gen in 1:length(method_name)){

    tryCatch({
      cat("Method",method_name[id_gen],"/ Case",id_case,"\n")
      print(h5ls(paste0(path_files,filenames[id_gen])))
      
      dat_pred=h5dump(paste0(path_files,filenames[id_gen]))[[dataname[id_gen]]]
      y_complete=(dat_pred[, ,1,])
      y_complete=array(y_complete,dim=c(Ngd*Ngd,dim(y_complete)[length(dim(y_complete))]))

      id_na=apply(y_complete,2,function(x){any(is.na(x))})
      cat("Number of samples without NaNs: ",sum(!id_na),"\n")
      sel_samples=which(!id_na)[1:nsamples]
      y_complete=y_complete[,sel_samples]
      cat("---------------\n")
      
      # par(mfrow=c(1,3))
      # plot_sim(z_complete,main=paste("Img ",id_case-1,"Data"),zlim=c(-5,5))
      # plot_sim(z_mask*z_complete,main=paste("Img ",id_case-1,"Mask"),zlim=c(-5,5))
      # plot_sim(y_complete[,1],main=paste("Img ",id_case-1,"Method",method_name[id_gen]),zlim=c(-5,5))
      # par(mfrow=c(1,1))

      y_obs=y_complete[sel_obs,]
      y_unobs=y_complete[sel_pred,]

      sco=do.call(c,lapply(cval_indices, function(l){mean(crps_sample(z_unobs[l],y_unobs[l,]))}))
      # hist(sco,main = paste0("Img ",id_case," CRPS - ",method_name[id_gen], "  : Mean = ",round(mean(sco),3)))
      crps_tab[id_case,id_gen]=mean(sco)
      crps_tab[id_case,3+id_gen]=sd(sco)
      
      sco=do.call(c,lapply(cval_indices, function(l){(es_sample(z_unobs[l],y_unobs[l,]))}))
      # hist(sco,main = paste0("Img ",id_case," CRPS - ",method_name[id_gen], "  : Mean = ",round(mean(sco),3)))
      es_tab[id_case,id_gen]=mean(sco)
      es_tab[id_case,3+id_gen]=sd(sco)

      ## Create map
      all_crps=crps_sample(z_unobs,y_unobs)
      crps_map=rep(NA, Ngd*Ngd)
      crps_map[sel_pred]=all_crps
      plot_sim(crps_map,col=viridis_colors,main=method_name[id_gen],zlim=c(0,3.8))

      }
      ,
      error=function(cond){
        cat(method_name[id_gen],"not found\n")
        }
    )

  }
  par(mfrow=c(1,1))
  dev.off()
}

res=rbind(mObs_vec,sdObs_vec)
colnames(res)=paste0("Img_",0:(length(mObs_vec)-1))
## Mean and standard deviation of each case
write.csv(res,file = "vecchia_temp/param_norm.csv",row.names = c("mu","sigma"))

## Create CSV
ndig=3
col.names=c("","img_id","vecchia","priorvae","mgdm")
n_methods=length(method_name)

crsp_val=matrix(NA,nrow = n_cases,ncol=n_methods+2)
crsp_val[,2]=0:(n_cases-1)
crsp_val[,1]=0
for(i in 1:n_cases){
  crsp_val[i,(1:n_methods)+2]=paste0(round(crps_tab[i,1:n_cases],ndig)," (",round(crps_tab[i,n_cases+1:n_cases],ndig),")")
}
colnames(crsp_val)=col.names
write.csv(crsp_val,"crps_post_temp.csv",row.names = F,quote = F)


es_val=matrix(NA,nrow = n_cases,ncol=n_methods+2)
es_val[,2]=0:(n_cases-1)
es_val[,1]=0
for(i in 1:n_cases){
  es_val[i,(1:n_methods)+2]=paste0(round(es_tab[i,1:n_cases],ndig)," (",round(es_tab[i,n_cases+1:n_cases],ndig),")")
}
colnames(es_val)=col.names
write.csv(es_val,"es_post_temp.csv",row.names = F,quote = F)

#----------------------------------------------------



  
  