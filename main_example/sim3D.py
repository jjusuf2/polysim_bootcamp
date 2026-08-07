# Simulations for compartments and extrusion using polychrom.
# Mirrors 202608_jamie/compSim.py, but drives the new_version LEF_Dynamics translocator:
# LEFs only -- no polymerase, no CTCF unstalling, no LEF-LEF permeability, no loop shrinking.
# The stall coin is flipped once per arrival at a site, so `stall` is a per-encounter probability.
#
# Example:
#   python sim3D.py comppath=setup_files/comps.dat ctcfpath=setup_files/ctcf.dat outpath=outputs

import os, sys
import numpy as np
import time, pickle

import openmm

import polychrom
from polychrom import forcekits, forces, simulation, starting_conformations
from polychrom.hdf5_format import HDF5Reporter
from polychrom.starting_conformations import grow_cubic, create_random_walk

from smcBondUpdater import smcBondUpdater
import pyximport; pyximport.install(setup_args={"include_dirs":np.get_include()})
from LEF_Dynamics import LEFTranslocatorDirectional

import simUtils, tools


#initialize paramsDict filled with params we can feed in via commandline
paramsDict={
            "gpu":str(0),
            "dt":40, #Goel et al 2025
            "thermostat":0.01, #Goel et al 2025
            "thermostat0":0.01, #Goel et al 2025
            "numsave":10000, #Goel et al 2025
            "saveevery":100, #Goel et al 2025
            "initskip":80, #Goel et al 2025
            "pbc":False, #Goel et al 2025
            "outpath":"outputs", # user defined
            "npoly":40080, # practical choice
            "nchr":1, # practical choice
            "density":0.2, # Gabriele et al 2022
            "repel":3.0, #Goel et al 2025
            "epsA":0., #Goel et al 2025
            "epsB":0.,
            "epsC":5, #Goel et al 2025
            "epsBC":0., #Goel et al 2025
            "epsAll":0., #Goel et al 2025
            "alen":500, #Goel et al 2025
            "blen":500, #Goel et al 2025 (not ultimately used)
            "clen":0, #Goel et al 2025 (not ultimately used)
            "cspace":500, #Goel et al 2025 (not ultimately used)
            "life":3000, #Gabriele et al 2022, 25min at 0.5s/timestep
            "sep":480, # Gabriele et al 2022
            "vlef":0.05, #Gabriele et al 2022; p(step per timestep per leg) = vlef
            "stall":0.8, # cohesin stall prob per encounter with a Ctcf
            "stallall":False,
            "polysteps":450, #calibrated to 0.001111111s/polymer step, so 0.5s/loop extrusion step
            "ignore":True,
            "integrator":"langevin", #Goel et al 2025
            "restartpath":"restart", #user defined
            "restart":"", #user defined
            "comppath":"", #user defined
            "ctcfpath":"", #user defined
            "microcomppath":"", #user defined
            "flag":"" #user defined
           }
#info about select variables
helpDict={
          "thermstat0":"thermostat for equilibration - default 0.01",
          "initskip":"num polymer equilibraiton blocks - default 80",
          "polysteps":"num polymer timesteps between LEF timesteps - default 450",
          "ignore":"whether to ignore adjacent particles for nonbond particle potentials - default True",
          "restartpath":"name of directory where restart file is located - default 'restart'",
          "restart":"name of restart file - default ''",
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
gpu=str(paramsDict['gpu'])
dt=int(paramsDict['dt'])
THERMOSTAT=float(paramsDict['thermostat'])
THERMOSTAT0=float(paramsDict['thermostat0']) if float(paramsDict['thermostat0'])>THERMOSTAT else THERMOSTAT
INTEGRATOR=str(paramsDict['integrator'])

prev_step=0

polyBlockSteps=int(paramsDict['polysteps'])

smcStepsPerBlock=1
saveEveryBlocks = int(paramsDict['saveevery'])#100  # save every 100 blocks
skipSavedBlocksBeginning = int(paramsDict['initskip'])  # how many blocks (saved) to skip after you restart LEF positions
totalSavedBlocks = int(paramsDict['numsave'])# how many blocks to save (number of blocks done is totalSavedBlocks * saveEveryBlocks)
restartUpdaterEveryBlocks = 1000 # don't need to restart very often.


####parameters for polymer
LENGTH=int(paramsDict['npoly'])
numChr=int(paramsDict['nchr'])
chrSizes=[LENGTH//numChr]*numChr
density=float(paramsDict['density'])
PBC=int(paramsDict['pbc'])
Rsph= (LENGTH/density)**(1./3.)
PBCbox_len=(4./3.*np.pi*LENGTH/density)**(1./3.)
pbc_param = [PBCbox_len]*3 if PBC else False  # feed this into simulation initialization
REPEL=float(paramsDict['repel']) #soft repulsion -- gives some excl vol, but chain can still pass itself.

polyBondWiggleDist= 0.1

ignoreAdjacent=paramsDict['ignore']
RESTART=False
if len(str(paramsDict['restart'])) > 0:
    skipSavedBlocksBeginning=0 # no LEF equilibration for a sim restart
    saveEveryBlocks=10 # save much more frequently
    RESTART=True
restartfile=str(paramsDict['restartpath'])+"/"+str(paramsDict['restart'])

# parameters for smc bonds
smcBondWiggleDist = 0.1
smcBondDist = 0.5

#compartment interactions
EPSILON_ALL=float(paramsDict['epsAll'])
EPSILON_A=float(paramsDict['epsA'])
EPSILON_B=float(paramsDict['epsB'])
EPSILON_C=float(paramsDict['epsC'])
EPSILON_BC=float(paramsDict['epsBC'])

B_length=int(paramsDict['blen'])
A_length=int(paramsDict['alen'])
C_length=int(paramsDict['clen'])
C_spacing=int(paramsDict['cspace'])
comp_list_str=str(paramsDict['comppath'])
microcomp_list_str=str(paramsDict['microcomppath'])

if not LENGTH % (B_length+A_length) == 0:
    print("Warning: total poly length not divisible by (len B + len A)")
#periodic list of blocks
B_list=[ii*(A_length+B_length) + jj for ii in range(int(LENGTH // (A_length+B_length))+1) for jj in range(B_length)
        if ii*(A_length+B_length) + jj < LENGTH]
if len(comp_list_str)>0:
    #####
    #FILE FORMAT: for each line is start end for a B compartment, except last line, which dictates how many A monomers follow
    #####
    with open(comp_list_str,"r") as myfile:
        lines=myfile.readlines()
    basePattern=[]
    for comp in lines[:-1]:
        cc= comp.rstrip().split() # read in B compartment start and ends
        basePattern=basePattern+list(np.arange(int(cc[0]),int(cc[1])))
    lastA=int(lines[-1].rstrip())
    B_list=list(basePattern)
    while max(B_list) < LENGTH - 1:
        B_list = B_list + list(basePattern+max(B_list)+1+lastA)
    while B_list[-1] >= LENGTH:
        B_list.pop()

if (EPSILON_C>0.) and (C_length>0):
    C_list=[C_spacing*ii + jj for ii in range(int(LENGTH // C_spacing)+1) for jj in range(C_length)
            if C_spacing*ii + jj < LENGTH]
else:
    C_list=[]
if len(microcomp_list_str)>0:
    #####
    #FILE FORMAT: for each line is start end for a C compartment, except last line, which dictates how many A monomers follow
    #####
    basePattern=[]
    with open(microcomp_list_str,"r") as myfile:
        lines=myfile.readlines()
    for comp in lines[:-1]:
        comp= comp.rstrip().split() # read in micro-compartment start and ends
        basePattern=basePattern+list(np.arange(int(comp[0]),int(comp[1])))
    lastA=int(lines[-1].rstrip())
    C_list=list(basePattern)
    while max(C_list) < LENGTH - 1:
        C_list = C_list + list(basePattern+max(C_list)+1+lastA)
    while C_list[-1] >= LENGTH:
        C_list.pop()

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
folder = paramsDict['outpath']+"/trajectory0001"
for pname in paramsDict:
    if pname not in ['gpu','outpath','initskip','numsave','saveevery','pbc','ignore',
                     'epsAll','epsC','epsBC','clen','cspace','epsA',
                     'stall','stallall',
                     'restart','restartpath',"comppath","ctcfpath","microcomppath",
                     'integrator','thermostat0',
                     'flag','alen','blen','clen']:
        folder = folder+"_"+pname+str(paramsDict[pname])
if EPSILON_A>0.:
    folder=folder+"_epsA"+str(EPSILON_A)
if EPSILON_ALL>0.:
    folder = folder+"_epsAll"+str(EPSILON_ALL)
if (EPSILON_C>0.) or (C_length>0):
    folder = folder+"_clen"+str(C_length)+"_cspace"+str(C_spacing)+"_epsC"+str(EPSILON_C)+"_epsBC"+str(EPSILON_BC)
if STALL_RATE>0.:
    if STALL_ALL:
        folder = folder+"_stallall"+str(STALL_RATE)
    else:
        folder = folder+"_stallsites"+str(STALL_RATE)
if PBC:
    folder = folder+"_"+"PBC"
if not INTEGRATOR=="langevin":
    folder = folder+"_"+str(paramsDict["integrator"])
if RESTART:
    folder = folder+"_restart"
if len(paramsDict["flag"])>0:
    folder = folder+"_"+paramsDict["flag"]

while os.path.exists(folder):
    folder=folder.replace("trajectory{:04}".format(folder_ind), "trajectory{:04}".format(folder_ind+1))
    folder_ind+=1

print(folder)
reporter = HDF5Reporter(folder=folder,
                        max_data_length=100, overwrite=True)
pickle.dump(paramsDict,open(folder+"/paramsDict.pkl","wb"))
if RESTART:
    os.system("cp {0} {1}".format(restartfile, folder+"/"))



# assertions for easy managing code below
assert restartUpdaterEveryBlocks % saveEveryBlocks == 0
assert (skipSavedBlocksBeginning * saveEveryBlocks) % restartUpdaterEveryBlocks == 0
assert (totalSavedBlocks * saveEveryBlocks) % restartUpdaterEveryBlocks == 0

#for equilibration
if not RESTART:
    while saveEveryBlocks * skipSavedBlocksBeginning * smcStepsPerBlock <= LIFETIME:
        skipSavedBlocksBeginning *= 2

savesPerUpdater = restartUpdaterEveryBlocks // saveEveryBlocks
updaterInitsSkip = saveEveryBlocks * skipSavedBlocksBeginning // restartUpdaterEveryBlocks
updaterInitsTotal = (totalSavedBlocks + skipSavedBlocksBeginning) * saveEveryBlocks // restartUpdaterEveryBlocks
print("Bond updater will be initialized {0} times, first {1} will be skipped".format(updaterInitsTotal, updaterInitsSkip))

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
    return SMCTran




#polymer object
polymer = np.array([]).reshape(0,3)

if not RESTART:
    #initialize polymer position
    for ii in range(numChr):
        if PBC:
            Rpoly= Rsph*(chrSizes[ii]/numChr)**(1./3.)
        else:
            Rpoly= PBCbox_len*(chrSizes[ii]/numChr)**(1./3.)

        def rw_constraint_func(start_pos, new_pos):
            """Return True if new_pos < Rpoly away from start_pos and new_pos is within Rsph of origin"""
            if ((np.sum((np.array(start_pos)-np.array(new_pos))**2)<Rpoly**2)
                and (np.sum(np.array(new_pos)**2)<Rsph**2)):
                return True
            else:
                return False

        if (ii==0):
            starting_pt=(-0.99*Rsph,0,0) # start first chr at this point so locus will start near periphery
        else:
            while True:
                starting_pt = Rsph*np.random.uniform(-1,1,3)
                if np.sum(starting_pt**2)<Rsph**2:
                    break

        polymer=np.concatenate((polymer, simUtils.create_constrained_random_walk(chrSizes[ii],
                                                                                 rw_constraint_func,
                                                                                 starting_point=starting_pt)))
    #center the polymer
    polymer = polymer - np.mean(polymer,axis=0)

else:
    polymer=pickle.load(open(restartfile,"rb"))


# ------------feed smcTran to the bond updater---
SMCTran = initModel()
init_num_steps=1000000
SMCTran.steps(init_num_steps)  # first steps to "equilibrate" SMC dynamics. If desired of course.
bondUpdater = smcBondUpdater(SMCTran)  # now feed this thing to bond updater (do it once!)


totBondsAdded=0
LEFStepsTaken=0

# now polymer simulation code starts
for updaterCount in range(updaterInitsTotal):
    doSave = updaterCount >= updaterInitsSkip
    print("updater init", updaterCount, "/", updaterInitsTotal)

    if doSave:
        THERMOSTAT_TO_USE=THERMOSTAT
    else:
        THERMOSTAT_TO_USE=THERMOSTAT0

    # simulation parameters are set below

    if doSave:
        max_Ek=20.
    else:
        max_Ek=20.

    sim = simulation.Simulation(platform="CUDA",
            GPU=gpu,
            integrator=INTEGRATOR,
            collision_rate=THERMOSTAT_TO_USE,
            timestep=dt,#40 is ok here
            max_Ek=max_Ek,
            N=LENGTH,
            PBCbox=pbc_param,
            save_decimals=3,
            reporters=[reporter])# set up GPU here

    sim.set_data(polymer, center=False)
    if not PBC:
        sim.add_force(forces.spherical_confinement(sim, r=Rsph,
                                                   #density=density,
                                                   k=10))
    extraHardList=[]
    if not ignoreAdjacent:
        extraHardList=B_list
    if (EPSILON_C > 0.) or (EPSILON_A > 0.):
        if not ignoreAdjacent:
            extraHardList= extraHardList + C_list
        nonbondedForce=forces.heteropolymer_SSW
        monoTypeList=np.zeros(LENGTH,dtype=int)
        monoTypeList[B_list] = 1
        monoTypeList[C_list] = 2
        extraHardList=np.unique(extraHardList)
        nonbondedKeys={
                       "interactionMatrix":np.array([[EPSILON_A,0.,0.],[0.,EPSILON_B,EPSILON_BC],[0.,EPSILON_BC,EPSILON_C]]),
                       "monomerTypes":monoTypeList,
                       "extraHardParticlesIdxs":extraHardList,
                       "repulsionEnergy": REPEL,
                       "attractionEnergy": EPSILON_ALL,
                      }
    else:
        nonbondedForce=forces.selective_SSW
        extraHardList=np.unique(extraHardList)
        nonbondedKeys={
                "repulsionEnergy": REPEL,
                "stickyParticlesIdxs": B_list,
                "extraHardParticlesIdxs": extraHardList, # empty unless ignoreAdjacent is False
                "attractionEnergy": EPSILON_ALL,
                "selectiveAttractionEnergy": EPSILON_B
                }

    sim.add_force(
        forcekits.polymer_chains(
            sim,
            chains=[(int(np.sum(chrSizes[:ii])), int(np.sum(chrSizes[:ii])+chrSizes[ii]), False) for ii in range(numChr)],
            bond_force_func=forces.harmonic_bonds,
            bond_force_kwargs={
                "bondLength": 1.0,
                "bondWiggleDistance": polyBondWiggleDist,  # Bond distance will fluctuate +-  on average
            },
            angle_force_func=forces.angle_force,
            angle_force_kwargs={
                "k": 1.5,
            },
            nonbonded_force_func=nonbondedForce, # force & keys chosen above
            nonbonded_force_kwargs=nonbondedKeys,
            except_bonds= ignoreAdjacent, #if True, don't calculate nonbonded forces for adjacent monos
        ))


    # ------------ initializing bondUpdater; adding bonds ---------
    # copied from addBond
    kbond = sim.kbondScalingFactor / (smcBondWiggleDist**2)
    bondDist = smcBondDist * sim.length_scale

    activeParams = {"length": bondDist, "k": kbond}
    inactiveParams = {"length": bondDist, "k": 0}
    bondUpdater.setParams(activeParams, inactiveParams)

    # this step actually puts all bonds in and sets first bonds to be what they should be
    bondUpdater.setup(
        bondForce=sim.force_dict['harmonic_bonds'],
        blocks=restartUpdaterEveryBlocks,
        smcStepsPerBlock=smcStepsPerBlock
    )  # now only one step of SMC per step
    print("Restarting bondUpdater")


    # If your simulation does not start, consider using energy minimization below
    if (updaterCount==0) or (updaterCount==updaterInitsSkip):
        sim.local_energy_minimization()
    else:
        sim._apply_forces()

    sim.step=prev_step
    for i in range(restartUpdaterEveryBlocks):
        if i % saveEveryBlocks == (saveEveryBlocks - 1):
            sim.do_block(steps=polyBlockSteps)
            if (not ("calibration" in paramsDict["flag"])) and (i%(10*saveEveryBlocks)== (10*saveEveryBlocks - 1)) and (SEPARATION<LENGTH):
                pickle.dump(
                        curBonds,
                        open(os.path.join(folder, "SMC{0}.dat".format(sim.step)), "wb"),
                        )
                with open(os.path.join(folder, "bondsAdded.txt"), "a") as bondCountFile:
                    bondCountFile.write(str(LEFStepsTaken)+" "+str(totBondsAdded)+"\n")
                totBondsAdded=0
                LEFStepsTaken=0
        else:
            sim.integrator.step(polyBlockSteps)  # do polyBlockSteps without getting the positions from the GPU (faster)
            sim.step+=polyBlockSteps # integrator doesn't increment steps
        if i < restartUpdaterEveryBlocks - 1:
            curBonds, pastBonds, numBondsAdded = bondUpdater.step(sim.context, countSteps=True)  # this updates bonds. You can do something with bonds here
            totBondsAdded += numBondsAdded
            LEFStepsTaken += 1
    polymer = sim.get_data()  # save data and step, and delete the simulation
    prev_step = sim.step
    del sim

    reporter.blocks_only = True  # Write output hdf5-files only for blocks

    time.sleep(0.2)


reporter.dump_data()
