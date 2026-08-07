# Minimal 1D lattice simulation (loop extrusion only) using LEF_Dynamics.
# Same LEF dynamics and the same parameters as sim3D.py, with the polychrom / OpenMM /
# 3D polymer and the bond updater removed. LEFs only -- no polymerase, no CTCF unstalling,
# no LEF-LEF permeability, no loop shrinking.
#
# Example:
#   python sim1D.py ctcfpath=setup_files/ctcf.dat
#
# Output (in <outpath>/lef1d####_.../):
#   LEFpositions.npy  int array, shape (numSaved, Nlefs, 2) -- (left leg, right leg) per LEF
#   savedSteps.npy    LEF timestep index of each save
#   sites.npz         CTCF stall arrays used by the run
#   paramsDict.pkl    all parameters

import os, pickle, time
import numpy as np

import pyximport; pyximport.install(setup_args={"include_dirs":np.get_include()})
from LEF_Dynamics import LEFTranslocatorDirectional

import tools


#initialize paramsDict filled with params we can feed in via commandline
#(only the parameters that the 1D simulation actually uses; values identical to sim3D.py)
paramsDict={
            "numsave":10000, # must be a multiple of restartUpdaterEveryBlocks//saveevery, and numsave*saveevery > life
            "saveevery":100, # num LEF steps between saves; must divide restartUpdaterEveryBlocks
            "initskip":80, # num saved frames' worth of blocks to discard as equilibration
            "initsteps":1000000, # LEF steps run before the block loop starts
            "outpath":"outputs", # user defined
            "npoly":40080, # practical choice
            "life":3000, #Gabriele et al 2022, 25min at 0.5s/timestep
            "sep":480, # Gabriele et al 2022
            "vlef":0.05, #Gabriele et al 2022; p(step per timestep per leg) = vlef
            "stall":0.8, # cohesin stall prob per encounter with a Ctcf
            "stallall":False,
            "ctcfpath":"", #user defined
            "flag":"" #user defined
           }
#info about select variables
helpDict={
          "saveevery":"num LEF steps between saves; must divide 1000 - default 100",
          "numsave":"num saved frames; must be a multiple of 1000//saveevery - default 10000",
          "initskip":"num equilibraiton blocks that are not recorded - default 80",
          "initsteps":"num LEF steps run before the block loop starts - default 1000000",
          "flag":"label to add to end of output folder - default ''"
         }

#get command line values and options
paramsInput= tools.argsList(pdict=paramsDict.keys(), hdict=helpDict)
for p in paramsInput.arg_dict:
    print(p, paramsInput.arg_dict[p])

for pname in paramsDict.keys():
    if pname in paramsInput.arg_dict:
        paramsDict[pname] = paramsInput.arg_dict[pname]


####basic sim parameters
smcStepsPerBlock=1
saveEveryBlocks = int(paramsDict['saveevery'])
skipSavedBlocksBeginning = int(paramsDict['initskip'])  # how many blocks (saved) to skip at the start
totalSavedBlocks = int(paramsDict['numsave'])# how many blocks to save (number of blocks done is totalSavedBlocks * saveEveryBlocks)
restartUpdaterEveryBlocks = 1000 # kept so the schedule matches sim3D.py

####parameters for polymer
LENGTH=int(paramsDict['npoly'])

####extrusion params
LIFETIME=float(paramsDict['life'])
SEPARATION=float(paramsDict['sep'])
EXTR_SPEED=float(paramsDict['vlef'])
Nlefs = int(LENGTH // SEPARATION)
STALL_RATE= float(paramsDict['stall'])
STALL_ALL= int(paramsDict['stallall'])
stall_path_str=str(paramsDict['ctcfpath'])

#### where the data goes
folder_ind=1
folder = paramsDict['outpath']+"/lef1d0001"
for pname in paramsDict:
    if pname not in ['outpath','initskip','initsteps','numsave','saveevery',
                     'stall','stallall',"ctcfpath",'flag']:
        folder = folder+"_"+pname+str(paramsDict[pname])
if STALL_RATE>0.:
    if STALL_ALL:
        folder = folder+"_stallall"+str(STALL_RATE)
    else:
        folder = folder+"_stallsites"+str(STALL_RATE)
if len(paramsDict["flag"])>0:
    folder = folder+"_"+paramsDict["flag"]

while os.path.exists(folder):
    folder=folder.replace("lef1d{:04}".format(folder_ind), "lef1d{:04}".format(folder_ind+1))
    folder_ind+=1

print(folder)
os.makedirs(folder)
pickle.dump(paramsDict,open(folder+"/paramsDict.pkl","wb"))


# assertions for easy managing code below
assert restartUpdaterEveryBlocks % saveEveryBlocks == 0
assert (skipSavedBlocksBeginning * saveEveryBlocks) % restartUpdaterEveryBlocks == 0
assert (totalSavedBlocks * saveEveryBlocks) % restartUpdaterEveryBlocks == 0

#for equilibration
#guard: with skipSavedBlocksBeginning==0 the product stays 0 and this loop never terminates
if skipSavedBlocksBeginning > 0:
    while saveEveryBlocks * skipSavedBlocksBeginning * smcStepsPerBlock <= LIFETIME:
        skipSavedBlocksBeginning *= 2

savesPerUpdater = restartUpdaterEveryBlocks // saveEveryBlocks
updaterInitsSkip = saveEveryBlocks * skipSavedBlocksBeginning // restartUpdaterEveryBlocks
updaterInitsTotal = (totalSavedBlocks + skipSavedBlocksBeginning) * saveEveryBlocks // restartUpdaterEveryBlocks
print("Simulation will run {0} rounds of {1} LEF steps, first {2} will be skipped;"
      " saving every {3} step(s) for {4} frames".format(
    updaterInitsTotal, restartUpdaterEveryBlocks, updaterInitsSkip, saveEveryBlocks,
    (updaterInitsTotal - updaterInitsSkip) * savesPerUpdater))

# more assertions for equilibration
assert (totalSavedBlocks * saveEveryBlocks * smcStepsPerBlock) > LIFETIME


def buildStallArrays():
    """CTCF stall probabilities for the two LEF legs, read from ctcfpath (or a uniform/empty fallback)."""
    stallLeftArray = np.zeros(LENGTH, dtype=np.double)
    stallRightArray = np.zeros(LENGTH, dtype=np.double)

    if len(stall_path_str)>0:
        with open(stall_path_str,"r") as myfile:
            lines=myfile.readlines()

        plus=True
        basePatternLeft=[]
        basePatternRight=[]
        for line in lines[:-1]:
            if line.rstrip()=="-":
                plus=False
                continue
            if plus:
                basePatternLeft.append(int(line.rstrip().split()[0]))
            else:
                basePatternRight.append(int(line.rstrip().split()[0]))
        repeat_interval=lines[-1].rstrip().split()
        stallLeftList=list(basePatternLeft)
        while max(stallLeftList)<LENGTH:
            stallLeftList = stallLeftList + [a+max(stallLeftList)+int(repeat_interval[0]) for a in basePatternLeft]
        while stallLeftList[-1]>=LENGTH:
            stallLeftList.pop()
        stallRightList=list(basePatternRight)
        while max(stallRightList)<LENGTH:
            stallRightList = stallRightList + [b+max(stallRightList)+int(repeat_interval[1]) for b in basePatternRight]
        while stallRightList[-1]>=LENGTH:
            stallRightList.pop()
        for ii in stallLeftList:
            stallLeftArray[ii] = STALL_RATE
        for ii in stallRightList:
            stallRightArray[ii] = STALL_RATE
    else:
        if not STALL_ALL:
            stallList = []# put locations of CTCFs here
        else:
            stallList = np.arange(LENGTH)
        for i in stallList:
            #put in correct stall prob
            stallLeftArray[i] = STALL_RATE
            stallRightArray[i] = STALL_RATE
    return stallLeftArray, stallRightArray


def initModel():
    # this just inits the simulation model. Put your previous init code here
    birthArray = np.zeros(LENGTH, dtype=np.double) + 0.1
    deathArray = np.zeros(LENGTH, dtype=np.double) + 1.0 / LIFETIME
    stallDeathArray = np.zeros(LENGTH, dtype=np.double) + 1 / LIFETIME
    # this translocator tests `randnum() > pause` to step, so pause is a genuine pause
    # probability and p(step per leg per timestep) = vlef
    pauseArray = np.ones(LENGTH, dtype=np.double) * (1.-EXTR_SPEED)

    stallLeftArray, stallRightArray = buildStallArrays()

    SMCTran = LEFTranslocatorDirectional(
        birthArray,
        deathArray,
        stallLeftArray,
        stallRightArray,
        pauseArray,
        stallDeathArray,
        Nlefs,
    )
    return SMCTran, stallLeftArray, stallRightArray


SMCTran, stallLeftArray, stallRightArray = initModel()

np.savez(os.path.join(folder, "sites.npz"),
         stallLeft=stallLeftArray, stallRight=stallRightArray)

init_num_steps=int(paramsDict['initsteps'])
if init_num_steps > 0:
    print("equilibrating LEF dynamics for {0} steps".format(init_num_steps))
    tstart=time.time()
    SMCTran.steps(init_num_steps)  # first steps to "equilibrate" SMC dynamics. If desired of course.
    print("done in {0:.1f} s".format(time.time()-tstart))


numSaves = (updaterInitsTotal - updaterInitsSkip) * savesPerUpdater
LEFpositions = np.zeros((numSaves, Nlefs, 2), dtype=np.int64)
savedSteps = np.zeros(numSaves, dtype=np.int64)
saveCount = 0
step = 0

# now the 1D simulation code starts
tstart=time.time()
for updaterCount in range(updaterInitsTotal):
    doSave = updaterCount >= updaterInitsSkip
    print("round", updaterCount, "/", updaterInitsTotal)

    for i in range(restartUpdaterEveryBlocks):
        SMCTran.steps(smcStepsPerBlock)
        step += smcStepsPerBlock
        if doSave and (i % saveEveryBlocks == (saveEveryBlocks - 1)):
            left, right = SMCTran.getLEFs()
            LEFpositions[saveCount, :, 0] = left
            LEFpositions[saveCount, :, 1] = right
            savedSteps[saveCount] = step
            saveCount += 1

print("{0} LEF steps in {1:.1f} s".format(step, time.time()-tstart))

np.save(os.path.join(folder, "LEFpositions.npy"), LEFpositions)
np.save(os.path.join(folder, "savedSteps.npy"), savedSteps)
print("saved {0} frames to {1}".format(saveCount, folder))
