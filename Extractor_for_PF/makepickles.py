import numpy as np
import uproot as up
#import ROOT as r
import awkward as ak
from torch_geometric.data import Data
import torch
from time import time
import pickle
import os


def rescale(feature, minval, maxval):
    top = feature-minval
    bot = maxval-minval
    return top/bot
def torchify(feat, graph_x = None):
    data = [Data(x = torch.from_numpy(ak.to_numpy(ele).astype(np.float32))) for ele in feat]
    if graph_x is not None:
        for d, gx in zip(data, graph_x):
            d.graph_x = gx
    return data
X_Min = -280
X_Max = 280

Y_Min = -280
Y_Max = 280

Z_Min = -550
Z_Max = 550

E_Min = 0
E_Max = 600

folder = "pf_pickles_plots"
split = 0.8
treeIn = "SinglePionGun_ken.root:s"
os.makedirs(folder, exist_ok=True)


tree = up.open(treeIn)

#totE_true_pf_bc = r.TH2F("totE_true_pf_bc","Total Energy of Rechits in HCAL+ECAL before cuts;Gen level energy;Using PFElements",100,0,600,100,0,600)
#eta_phi_bc = r.TH2F("eta_phi_bc","True Eta-Phi before cuts;Eta;Phi",100,-3,3,100,-3.15,3.15)
#totE_true_pf_ac = r.TH2F("totE_true_pf_ac","Total Energy of Rechits in HCAL+ECAL after cuts;Gen level energy;Using PFElements",100,0,600,100,0,600)
#pf_true_ratio = r.TH2F("pf_true_ratio","PF/True vs true;True;PF/True",100,0,500,100,0,3)
t0=time()
maxEvents = 10000
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
ecalSum = ak.sum(ecalHitsEn,axis=1)
hcalSum = ak.sum(hcalHitsEn,axis=1)
totE = (ecalSum+hcalSum)

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

ecalSum = ak.sum(ecalHitsEn,axis=1)
hcalSum = ak.sum(hcalHitsEn,axis=1)
totE = (ecalSum+hcalSum)
true = true[filter]

#for i in range(len(true)):
#    totE_true_pf_ac.Fill(true[i],totE[i])
print("\tApplied event selections in %0.3f seconds"%(time()-t0))

def cartfeat(x, y, z, En ):
    E = rescale(En, E_Min, E_Max)
    x = rescale(x, X_Min, X_Max)
    y = rescale(y, Y_Min, Y_Max)
    z = rescale(z, Z_Min, Z_Max)
    return ak.concatenate((x[:,:,None], y[:,:,None], z[:,:,None], E[:,:,None]), -1)

t0=time()
Hit_X = ak.concatenate((hadHitX,emHitX),axis=1)
Hit_Y = ak.concatenate((hadHitY,emHitY),axis=1)
Hit_Z = ak.concatenate((hadHitZ,emHitZ),axis=1)
Hit_E = ak.concatenate((hcalHitsEn,ecalHitsEn),axis=1)
remzeros = [len(j)>0 for j in Hit_X]
Hit_X = Hit_X[remzeros]
Hit_Y = Hit_Y[remzeros]
Hit_Z = Hit_Z[remzeros]
Hit_E = Hit_E[remzeros]
true = true[remzeros]
eta = eta[remzeros]
ecal = ecal[remzeros]
ecalHitsEn = ecalHitsEn[remzeros]
hcalHitsEn = hcalHitsEn[remzeros]
print("Building features")
cf = cartfeat(Hit_X,Hit_Y,Hit_Z,Hit_E)
print("\tBuilt features in %0.3f seconds"%(time()-t0))

t0=time()
cf = torchify(cf)
print("\tTorchify in %0.3f seconds"%(time()-t0))

t0=time()
with open("%s/trueE_target.pickle"%folder, 'wb') as f:
    pickle.dump(true, f, protocol = 4)
print("\tDumped trueE target in %0.3f seconds"%(time()-t0))

with open("%s/hcalHitsEn.pickle"%folder, 'wb') as f:
    pickle.dump(hcalHitsEn, f, protocol = 4)
print("\tDumped hcalHitsEn target in %0.3f seconds"%(time()-t0))

with open("%s/ecalHitsEn.pickle"%folder, 'wb') as f:
    pickle.dump(ecalHitsEn, f, protocol = 4)
print("\tDumped ecalHitsEn target in %0.3f seconds"%(time()-t0))

t0=time()
with open("%s/recHitEn.pickle"%folder, 'wb') as f:
    pickle.dump(Hit_E, f, protocol = 4)
print("\tDumped rechitEn in %0.3f seconds"%(time()-t0))

t0=time()
with open("%s/eta.pickle"%folder, 'wb') as f:
    pickle.dump(eta, f, protocol = 4)
print("\tDumped eta in %0.3f seconds"%(time()-t0))

t0=time()
with open("%s/ecal.pickle"%folder, 'wb') as f:
    pickle.dump(ecal, f, protocol = 4)
print("\tDumped ecal in %0.3f seconds"%(time()-t0))

t0=time()
with open("%s/cartfeat.pickle"%folder, 'wb') as f:
    torch.save(cf, f, pickle_protocol = 4)
print("\tDumped features in %0.3f seconds"%(time()-t0))

length=len(true)
train_idx = np.random.choice(length, int(split * length + 0.5), replace=False)

mask = np.ones(length, dtype=bool)
mask[train_idx] = False
valid_idx = mask.nonzero()[0]

with open("%s/all_valididx.pickle" % folder, "wb") as f:
    pickle.dump(valid_idx, f)

with open("%s/all_trainidx.pickle" % folder, "wb") as f:
    pickle.dump(train_idx, f)


"""
t0=time()
plotsToDraw=[totE_true_pf_bc,totE_true_pf_ac,eta_phi_bc]#,pf_true_ratio]
def drawPlot(plot,plotname="",log=False):
    if(plotname==""):
        plotname=plot.GetName()
    canvas = r.TCanvas("canvas", "2D Histogram Canvas", 800, 800)
    plot.Draw("colz")
    r.gStyle.SetOptStat(111111)
    canvas.SetLeftMargin(0.12)
    if(log):
        canvas.SetLogy()
    file = r.TFile("output.root", "RECREATE")
    file.cd()
    canvas.SetRightMargin(0.1)
    canvas.Update()
    canvas.Draw()
    canvas.Write()
    file.Close()
    canvas.SaveAs(folder+"/"+plotname+".png")
    canvas.SaveAs(folder+"/"+plotname+".pdf")

for i in plotsToDraw:
    print("Plotting ",i.GetName())
    drawPlot(i,i.GetName())
print("\tDumped plots in %0.3f seconds"%(time()-t0))
"""
