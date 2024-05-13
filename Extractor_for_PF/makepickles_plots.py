import numpy as np
import uproot as up
#import ROOT as r
import awkward as ak
from time import time
import pickle
import os


def rescale(feature, minval, maxval):
    top = feature-minval
    bot = maxval-minval
    return top/bot
X_Min = -280
X_Max = 280

Y_Min = -280
Y_Max = 280

Z_Min = -550
Z_Max = 550

E_Min = 0
E_Max = 600

folder = "pf_pickles_plots_20M_v2"
split = 0.8
treeIn = "SinglePionGun_full.root:s"
os.makedirs(folder, exist_ok=True)


tree = up.open(treeIn)

#totE_true_pf_bc = r.TH2F("totE_true_pf_bc","Total Energy of Rechits in HCAL+ECAL before cuts;Gen level energy;Using PFElements",100,0,600,100,0,600)
#eta_phi_bc = r.TH2F("eta_phi_bc","True Eta-Phi before cuts;Eta;Phi",100,-3,3,100,-3.15,3.15)
#totE_true_pf_ac = r.TH2F("totE_true_pf_ac","Total Energy of Rechits in HCAL+ECAL after cuts;Gen level energy;Using PFElements",100,0,600,100,0,600)
#pf_true_ratio = r.TH2F("pf_true_ratio","PF/True vs true;True;PF/True",100,0,500,100,0,3)
t0=time()
maxEvents = 5000000
true = tree["true"].array(entry_stop=maxEvents)
emHitE = tree["emHitE"].array(entry_stop=maxEvents)
emHitF = tree["emHitF"].array(entry_stop=maxEvents)
emHitX = tree["emHitX"].array(entry_stop=maxEvents)
emHitY = tree["emHitY"].array(entry_stop=maxEvents)
emHitZ = tree["emHitZ"].array(entry_stop=maxEvents)
hadHitE = tree["hadHitE"].array(entry_stop=maxEvents)
hadHitF = tree["hadHitF"].array(entry_stop=maxEvents)
hadHitX = tree["hadHitX"].array(entry_stop=maxEvents)
hadHitY = tree["hadHitY"].array(entry_stop=maxEvents)
hadHitZ = tree["hadHitZ"].array(entry_stop=maxEvents)
eta = tree["eta"].array(entry_stop=maxEvents)
phi = tree["phi"].array(entry_stop=maxEvents)
ecal = tree["ecal"].array(entry_stop=maxEvents)
print("\tLoaded features in %0.3f seconds"%(time()-t0))

t0=time()
ecalHitsEn = emHitE*emHitF
hcalHitsEn = hadHitE*hadHitF

#for i in range(len(true)):
#    totE_true_pf_bc.Fill(true[i],totE[i])    
#    eta_phi_bc.Fill(eta[i],phi[i])

truep = true + 0.0001
filter = (true>0)

ecalHitsEn = ecalHitsEn[filter]
hcalHitsEn = hcalHitsEn[filter]
emHitX = emHitX[filter]
emHitY = emHitY[filter]
emHitZ = emHitZ[filter]
hadHitX = hadHitX[filter]
hadHitY = hadHitY[filter]
hadHitZ = hadHitZ[filter]
eta = eta[filter]
ecal = ecal[filter]

true = true[filter]

#for i in range(len(true)):
#    totE_true_pf_ac.Fill(true[i],totE[i])
print("\tApplied event selections in %0.3f seconds"%(time()-t0))

t0=time()
Hit_X = ak.concatenate((hadHitX,emHitX),axis=1)
Hit_Y = ak.concatenate((hadHitY,emHitY),axis=1)
Hit_Z = ak.concatenate((hadHitZ,emHitZ),axis=1)
Hit_E = ak.concatenate((hcalHitsEn,ecalHitsEn),axis=1)
remzeros = [(len(hadHitX[j])+len(emHitX[j]))>0 for j in range(len(hadHitX))]
Hit_X = Hit_X[remzeros]
Hit_Y = Hit_Y[remzeros]
Hit_Z = Hit_Z[remzeros]
Hit_E = Hit_E[remzeros]
true = true[remzeros]
eta = eta[remzeros]
ecal = ecal[remzeros]
#hcal = hcal[remzeros]
ecalHitsEn = ecalHitsEn[remzeros]
hcalHitsEn = hcalHitsEn[remzeros]
emHitX = emHitX[remzeros]
emHitY = emHitY[remzeros]
emHitZ = emHitZ[remzeros]
hadHitX = hadHitX[remzeros]
hadHitY = hadHitY[remzeros]
hadHitZ = hadHitZ[remzeros]
print("Building features")
#cf = cartfeat(Hit_X,Hit_Y,Hit_Z,Hit_E)
print("\tBuilt features in %0.3f seconds"%(time()-t0))

t0=time()
print("\tTorchify in %0.3f seconds"%(time()-t0))

t0=time()
with open("%s/eta.pickle"%folder, 'wb') as f:
    pickle.dump(eta, f, protocol = 4)
print("\tDumped eta in %0.3f seconds"%(time()-t0))

t0=time()
with open("%s/ecal.pickle"%folder, 'wb') as f:
    pickle.dump(ecal, f, protocol = 4)
print("\tDumped ecal in %0.3f seconds"%(time()-t0))

t0=time()
with open("%s/trueE_target.pickle"%folder, 'wb') as f:
    pickle.dump(true, f, protocol = 4)
print("\tDumped trueE target in %0.3f seconds"%(time()-t0))

with open("%s/hadHitX.pickle"%folder, 'wb') as f:
    pickle.dump(hadHitX, f, protocol = 4)
print("\tDumped hadHitX target in %0.3f seconds"%(time()-t0))

with open("%s/hadHitY.pickle"%folder, 'wb') as f:
    pickle.dump(hadHitY, f, protocol = 4)
print("\tDumped hadHitY target in %0.3f seconds"%(time()-t0))

with open("%s/hadHitZ.pickle"%folder, 'wb') as f:
    pickle.dump(hadHitZ, f, protocol = 4)
print("\tDumped hadHitZ target in %0.3f seconds"%(time()-t0))

with open("%s/Hit_X.pickle"%folder, 'wb') as f:
    pickle.dump(Hit_X, f, protocol = 4)
print("\tDumped HitX target in %0.3f seconds"%(time()-t0))

with open("%s/Hit_Y.pickle"%folder, 'wb') as f:
    pickle.dump(Hit_Y, f, protocol = 4)
print("\tDumped HitY target in %0.3f seconds"%(time()-t0))

with open("%s/Hit_Z.pickle"%folder, 'wb') as f:
    pickle.dump(Hit_Z, f, protocol = 4)
print("\tDumped HitZ target in %0.3f seconds"%(time()-t0))

with open("%s/Hit_E.pickle"%folder, 'wb') as f:
    pickle.dump(Hit_E, f, protocol = 4)
print("\tDumped HitE target in %0.3f seconds"%(time()-t0))

with open("%s/hcalHitsEn.pickle"%folder, 'wb') as f:
    pickle.dump(hcalHitsEn, f, protocol = 4)
print("\tDumped hcalHitsEn target in %0.3f seconds"%(time()-t0))

with open("%s/ecalHitsEn.pickle"%folder, 'wb') as f:
    pickle.dump(ecalHitsEn, f, protocol = 4)
print("\tDumped ecalHitsEn target in %0.3f seconds"%(time()-t0))
