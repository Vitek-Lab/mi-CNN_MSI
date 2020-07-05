library("ggplot2")
library("e1071")
library("Cardinal")
library(tensorflow)
library(magrittr)
library("gtools")
########load data and pre-process
load("data/rccTissue.rdata")

rcc<-rcc.tissue

register(SerialParam())

rcc.resample<-rcc%>%
  normalize(method = "tic")%>%
  peakBin(c(seq(from=151, to = 1000, by = 1))) %>%
  process()


rcc.sim<-rcc.resample[,rcc.resample$diagnosis%in%c("normal")]

rcc.sim$sample <- run(rcc.sim)

rcc.sim$diagnosis <- as.character(rcc.sim$diagnosis)
###divide sample into two groups
cancer_ind<-c()
cancer_label<-c()
for (i in 1:length(unique(rcc.sim$sample)))
{
  rcc.sim.sub<-rcc.sim[,rcc.sim$sample==unique(rcc.sim$sample)[i]]
  
  xm<-(range(coord(rcc.sim.sub)$x)[2]+range(coord(rcc.sim.sub)$x)[1])/2
  ym<-(range(coord(rcc.sim.sub)$y)[2]+range(coord(rcc.sim.sub)$y)[1])/2
  
  cancer_ind<-c(cancer_ind,which(rcc.sim$sample==unique(rcc.sim$sample)[i]&(coord(rcc.sim)
                                                                            $x<xm)&(coord(rcc.sim)
                                                                                    $y<ym)))
  cancer_label<-c(cancer_label,which(rcc.sim$sample==unique(rcc.sim$sample)[i]&(coord(rcc.sim)
                                                                                $y<ym)))
}

rcc.sim$diagnosis[cancer_label]<-"cancer"
true_label<-rep("normal",dim(rcc.sim)[2])
true_label[cancer_ind]<-"cancer"
true_label<-as.factor(true_label)
image(rcc.sim, true_label~x*y,layout=c(2,2))
image(rcc.sim, diagnosis~x*y, layout=c(2,2))


##################
circle_ind<-c()

for (i in 1:length(unique(rcc.sim$sample)))
{
  rcc.sim.sub<-rcc.sim[,rcc.sim$sample==unique(rcc.sim$sample)[i]]
  
  xm<-(range(coord(rcc.sim.sub)$x)[2]+range(coord(rcc.sim.sub)$x)[1])/2
  ym<-(range(coord(rcc.sim.sub)$y)[2]+range(coord(rcc.sim.sub)$y)[1])/2
  
  circle_ind<-c(circle_ind,which(rcc.sim$sample==unique(rcc.sim$sample)[i]&(abs(coord(rcc.sim)
                                                                                $x-xm)<5)&(abs(coord(rcc.sim)
                                                                                                $y-ym)<5)))
  
}

rcc.sim$circle<-0
rcc.sim$circle[circle_ind]<-1


image(rcc.sim,circle~x*y)
#########randomly pick one molecule x, x+23, x+39, x-18
x<-sample(200:800,size=1)
#####case 1: sum(x,x+23,x+39, x-18) differ 
####for each pixel, intensity(sum(x))~N(group k, sigma), intensity(x)~p(x)*intensity(sum(x)), sum(p(x))=1
###divide sample into two groups
id<-which(mz(rcc.sim)==x)


id<-275
####mean and variance of x in different groups
c_mean<-50
n_mean<-150
c_sigma_bio<-c_mean*0.15
n_sigma_bio<-n_mean*0.15
c_sigma<-c_mean*0.1
n_sigma<-n_mean*0.1
####simulate intensity for cancer group
####cancer

set.seed(1256)
cancer_id<-which(true_label=="cancer")
n_cancer<-length(cancer_id)

normal_id<-which(true_label=="normal")
n_normal<-length(normal_id)

for (s in unique(rcc.sim$sample))
{
  
  s_id <- which (rcc.sim$sample == s)
  s_cancer_id <- cancer_id [which (cancer_id %in% s_id)]
  
  if ( length(s_cancer_id)!=0 )
  {
    c_mean_s <- rnorm(1,0,c_sigma_bio)
    in_s <- rnorm(1, 5, 5*0.1)
    out_s <- rnorm(1, -5, 5*0.1)
    
    for (cid in s_cancer_id)
    {
      if (cid %in% circle_ind)
      {
        sum_x<-c_mean + c_mean_s+rnorm(1,0,c_sigma)+ in_s
      }else
      {
        sum_x<-c_mean + c_mean_s+rnorm(1,0,c_sigma)+out_s
      }
      
      p_v <- rdirichlet(1,c(1,1,1,1))
      int_x<-sum_x*p_v[1]
      int_x23<-sum_x*p_v[2]
      int_x39<-sum_x*p_v[3]
      int_x18<-sum_x*p_v[4]
      spectra(rcc.sim)[id,cid]<-int_x
      
      spectra(rcc.sim)[id+22,cid]<-int_x23
      
      spectra(rcc.sim)[id+38,cid]<-int_x39
      
      spectra(rcc.sim)[id-18,cid]<-int_x18
    }
  }
  
  s_normal_id <- normal_id [which (normal_id %in% s_id)]
  
  n_mean_s <- rnorm(1,0,n_sigma_bio)
  
  for (nid in s_normal_id)
  {
    if (nid %in% circle_ind)
    {
      sum_x<-n_mean + n_mean_s + rnorm(1,0,n_sigma)+in_s
    }else{
      sum_x<-n_mean + n_mean_s + rnorm(1,0,n_sigma)+out_s
    }
    
    p_v <- rdirichlet(1,c(1,1,1,1))
    int_x<-sum_x*p_v[1]
    int_x23<-sum_x*p_v[2]
    int_x39<-sum_x*p_v[3]
    int_x18<-sum_x*p_v[4]
    spectra(rcc.sim)[id,nid]<-int_x
    
    spectra(rcc.sim)[id+22,nid]<-int_x23
    
    spectra(rcc.sim)[id+38,nid]<-int_x39
    
    spectra(rcc.sim)[id-18,nid]<-int_x18
  }
}

