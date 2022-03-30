#!/usr/bin/env python
####### Author: Liliana C. Gallegos
####### Email: lilianac.gallegos@colostate.edu

import numpy as np
import pandas as pd
import os
import os.path
import glob
import time
import subprocess
import pickle

import json

from rdkit.Chem import AllChem as Chem
from rdkit.Chem import rdDistGeom, Draw, Descriptors, rdMolAlign, SDMolSupplier, rdMolTransforms, rdmolfiles # TorsionFingerprints, rdMolTransforms, PropertyMol,
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.drawOptions.addAtomIndices = False
IPythonConsole.drawOptions.addStereoAnnotation = True
from rdkit.Chem.Draw import rdDepictor
rdDepictor.SetPreferCoordGen(True)
from rdkit.Geometry import Point3D
from rdkit.Chem import PandasTools

def RE_intermediate(ligand, aryl, amine):
    '''
    Generates reductive elimination intermediate from substrate smiles.
    Input smiles for ligand, aryl, and amine. Returns the mol object for intermediate.
    '''

    # Convert to mol objects from given substrate smiles
    Pd2_mol  = Chem.MolFromSmiles('Cl[Pd]Cl')
    lig_mol  = Chem.MolFromSmiles(ligand)
    ar_mol = Chem.MolFromSmiles(aryl)
    am_mol = Chem.MolFromSmiles(amine)

    # Build intermediate with SMARTS
    rxn2 = Chem.ReactionFromSmarts('[P:1].[Pd:2][Cl]>>[Pd:2]([P+:1])')
    rxn3 = Chem.ReactionFromSmarts('([P:1].[Pd:2]([P+:3])[Cl:4])>>[P+:1][Pd:2]([P+:3])')
    rxn4 = Chem.ReactionFromSmarts('[Pd:1].[#6:2]([Cl,Br,I:3]).[#7:4]>>[Pd:1]([#6:2])([#7:4]).[Cl,Br,I:3].[H]')

    inter1 = rxn2.RunReactants((lig_mol, Pd2_mol))[0][0]
    inter2 = rxn3.RunReactants((inter1,))[0][0]
    intermediate = rxn4.RunReactants((inter2, ar_mol, am_mol))[0][0]
    Chem.SanitizeMol(intermediate)

    return intermediate

def get_atoms(mol):
    atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
    symbol = [a.GetSymbol() for a in mol.GetAtoms()]
    return atoms, symbol

def core_atoms(mol, center_atom_num):
    ''' Finds index of the metal and its neighboring atoms given center atomic number. '''
    # intermediate mol
    center_idx, nneighbours, neighbours = [],[],[]
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == center_atom_num:  # Pd == 46
            center_idx = atom.GetIdx()
            nneighbours = len(atom.GetNeighbors())
            neighbours = atom.GetNeighbors()
    return center_idx, nneighbours, neighbours

## embed_sqplanar_template() source: pyconfort
def embed_sqplanar_template(mol):
    ''' Embeds the square-planar template [snips of code from pyconfort]. '''
    sqtemp = os.path.basename('./template-4-and-5.sdf')
    sqtemp = [ mol_1 for mol_1 in Chem.SDMolSupplier(sqtemp) ]
    sqtemp = sqtemp[0]

    center_idx, nneighbours, neighbours = core_atoms(mol, 46)

    for name in range(1):
        #assigning order of replacement
        if name == 0:
            j = [1,2,3]

        for atom in sqtemp.GetAtoms():
            if atom.GetSymbol() == 'F':
                mol_1 = Chem.RWMol(sqtemp)
                idx = atom.GetIdx()
                mol_1.RemoveAtom(idx)
                mol_1 = mol_1.GetMol()

        for atom in mol_1.GetAtoms():
    #         if atom.GetIdx() == 4:
    #             atom.SetAtomicNum(14)
            if atom.GetIdx() == 0:
                atom.SetAtomicNum(neighbours[0].GetAtomicNum())
            if atom.GetIdx() == 3:
                atom.SetAtomicNum(neighbours[j[0]].GetAtomicNum())
            if atom.GetIdx() == 2:
                atom.SetAtomicNum(neighbours[j[1]].GetAtomicNum())
            if atom.GetIdx() == 1:
                atom.SetAtomicNum(neighbours[j[2]].GetAtomicNum())

        # assigning and embedding onto the core
        num_atom_match = mol.GetSubstructMatch(mol_1)
        mol_embed = Chem.AddHs(mol)

        # definition of coordmap, the coreconfID(the firstone =-1)
        coordMap = {}
        coreConfId=-1
        randomseed=-1
        force_constant=10000

        # This part selects which atoms from molecule are the atoms of the core
        try:
            coreConf = mol_1.GetConformer(coreConfId)
        except:
            pass
        for k, idxI in enumerate(num_atom_match):
            core_mol_1 = coreConf.GetAtomPosition(k)
            coordMap[idxI] = core_mol_1

        ci = rdDistGeom.EmbedMolecule(mol_embed, coordMap=coordMap, randomSeed=randomseed)

        if ci >= 0:
            GetFF = Chem.UFFGetMoleculeForceField(mol_embed,confId=-1)

            #algin molecule to the core
            algMap = [(k, l) for l, k in enumerate(num_atom_match)]

            for k, idxI in enumerate(num_atom_match):
                for l in range(k + 1, len(num_atom_match)):
                    idxJ = num_atom_match[l]
                    d = coordMap[idxI].Distance(coordMap[idxJ])
                    GetFF.AddDistanceConstraint(idxI, idxJ, d, d, force_constant)
            GetFF.Initialize()
            GetFF.Minimize(maxIts=100)
            # rotate the embedded conformation onto the core_mol:
            rdMolAlign.AlignMol(mol_embed, mol_1, atomMap=algMap)
    return mol_embed

def build_TS(mol, new_angle, new_dist1, new_dist2):
    ''' Transforms the intermediate of mol object given into the TS structure with the angle, dihedral angle, and distances. '''
    center_idx, nneighbours, neighbours = core_atoms(mol, 46)
    Patom, NCatom = [], []
    Xhalides = [17, 35, 53] # Cl, Br, I
    for i, nb in enumerate(neighbours):
        if nb.GetAtomicNum() == 6: Catom = nb.GetIdx()
        if nb.GetAtomicNum() in set(Xhalides): XNatom = nb.GetIdx()
        if nb.GetAtomicNum() == 7:
            XNatom = nb.GetIdx()
            Nnb = nb.GetNeighbors()
            Nidx = [N.GetIdx() for N in Nnb if N.GetAtomicNum() != 46]
            NCatom.append(Nidx)
        if nb.GetAtomicNum() == 15: Patom.append(nb.GetIdx())

    ## modify by bond distance: Pd-C
    current_dist1 = Chem.GetBondLength(mol.GetConformer(), center_idx, Catom); #print(current_dist1)
    Chem.SetBondLength(mol.GetConformer(), center_idx, Catom, new_dist1)
    ## modify by bond distance: Pd-XN
    current_dist2 = Chem.GetBondLength(mol.GetConformer(), center_idx, XNatom); #print(current_dist2)
    Chem.SetBondLength(mol.GetConformer(), center_idx, XNatom, new_dist2)
    ## modify by angle: C-Pd-XN
    for p in Patom:
        ppdc_angle = Chem.GetAngleDeg(mol.GetConformer(), p, center_idx, Catom)
        ppdn_angle = Chem.GetAngleDeg(mol.GetConformer(), p, center_idx, XNatom)
        if ppdc_angle < 91:
            ppdc_chng = round(float((90.0 - new_angle)*0.2),2)
            Chem.SetAngleDeg(mol.GetConformer(), p, center_idx, Catom, ppdc_angle+ppdc_chng)
        elif ppdn_angle < 91:
            ppdn_chng = round(float((90.0 - new_angle)*0.8),2)
            Chem.SetAngleDeg(mol.GetConformer(), p, center_idx, XNatom, ppdn_angle+ppdn_chng)
    current_angle = Chem.GetAngleDeg(mol.GetConformer(), Catom, center_idx, XNatom); #print(current_angle)
    max_angle = new_angle+5.0
    if current_angle > max_angle:
        Chem.SetAngleDeg(mol.GetConformer(), Catom, center_idx, XNatom, max_angle)
    ## modify by dihedral angle on amines 2 neighboring atoms: (e.g., not for imines): P-Pd-N-C or P-Pd-N-H
    if len(NCatom) == 2:
        for p in Patom:
            pn_angle = Chem.GetAngleDeg(mol.GetConformer(), p, center_idx, XNatom);
            if pn_angle < 130:
                cdih1 = Chem.GetDihedralDeg(mol.GetConformer(), p, center_idx, XNatom, NCatom[0][0])
                cdih2 = Chem.GetDihedralDeg(mol.GetConformer(), p, center_idx, XNatom, NCatom[0][1])
                if cdih1 < -71:
                    adj = cdih1 + 71.0
                    new_cdih2 = round(float(cdih2 - adj), 2); #print(new_cdih2)
                    if new_cdih2 < 80: Chem.SetDihedralDeg(mol.GetConformer(), p, center_idx, XNatom, NCatom[0][0], -71.0)
                    else: Chem.SetDihedralDeg(mol.GetConformer(), p, center_idx, XNatom, NCatom[0][0], 71.0)
                elif cdih1 > 71:
                    adj = cdih1 - 71.0
                    new_cdih2 = round(float(cdih2 - adj), 2); #print(new_cdih2)
                    if new_cdih2 > -80: Chem.SetDihedralDeg(mol.GetConformer(), p, center_idx, XNatom, NCatom[0][0], 71.0)
                    else: Chem.SetDihedralDeg(mol.GetConformer(), p, center_idx, XNatom, NCatom[0][0], -71.0)

                if cdih2 < -71:
                    adj = cdih2 + 71.0
                    new_cdih1 = round(float(cdih1 - adj), 2); #print(new_cdih1)
                    if new_cdih1 < 80: Chem.SetDihedralDeg(mol.GetConformer(), p, center_idx, XNatom, NCatom[0][1], -71.0)
                    else: Chem.SetDihedralDeg(mol.GetConformer(), p, center_idx, XNatom, NCatom[0][1], 71.0)
                elif cdih2 > 71:
                    adj = cdih2 - 71.0
                    new_cdih1 = round(float(cdih1 - adj), 2); #print(new_cdih1)
                    if new_cdih1 > -80: Chem.SetDihedralDeg(mol.GetConformer(), p, center_idx, XNatom, NCatom[0][1], 71.0)
                    else: Chem.SetDihedralDeg(mol.GetConformer(), p, center_idx, XNatom, NCatom[0][1], -71.0)
    return mol

# get_minstructure() source - Project: GB-GA   Author: jensengroup   File: scoring_functions.py
def get_minstructure(mol, n_confs):
    '''From a mol object, gives the minimized conformer from multiple conformers with coordinates.'''

    mol = Chem.AddHs(mol)
    new_mol = Chem.Mol(mol)

    Chem.EmbedMultipleConfs(mol,numConfs=n_confs,useExpTorsionAnglePrefs=True,useBasicKnowledge=True)
#     energies = Chem.UFFOptimizeMoleculeConfs(mol,maxIters=2000, nonBondedThresh=100.0)
    energies = Chem.MMFFOptimizeMoleculeConfs(mol,maxIters=2000, nonBondedThresh=100.0)

    energies_list = [e[1] for e in energies]
    min_e_index = energies_list.index(min(energies_list))
    new_molE = min(energies_list)
    new_mol.AddConformer(mol.GetConformer(min_e_index))
#     new_mol = Chem.RemoveHs(new_mol)

    conformer = new_mol.GetConformer()
    coordinates = conformer.GetPositions()
    coordinates = np.array(coordinates)

    atoms, symbol = get_atoms(new_mol)
    return new_mol, atoms, symbol, coordinates

def create_constraints(name, mol, path):
    '''
    From mol object, creates a constraint input file of TS active bonds (e.g., Pd-C, Pd-N/X, N-C, Pd-P).
    Input the mol object; returns either a True or False if constraint.inp file created or not.
    '''
    mol_dir = path+'/'+name
    if not os.path.exists(mol_dir): print('Directory/filename does not exist.')
    os.chdir(mol_dir)

    center_idx, nneighbours, neighbours = core_atoms(mol, 46)

    # Since labels begin at 1, NEED to change index by +1.
    center_label = center_idx+1
    Xhalides = [17, 35, 53] # Cl, Br, I
    for i, nb in enumerate(neighbours):
        if nb.GetAtomicNum() == 6: Clabel = nb.GetIdx()+1
        if nb.GetAtomicNum() in set(Xhalides): XNlabel = nb.GetIdx()+1
        if nb.GetAtomicNum() == 7: XNlabel = nb.GetIdx()+1
        if nb.GetAtomicNum() == 15: Plabel = nb.GetIdx()+1

    # creates constrain script to collect constrain atoms (deleted at end)
    with open(f'./crest_{name}_constraint.sh','w+') as f:
        f.write('#!/bin/bash \n')
        f.write(f'crest {name}_coord --constrain {Clabel},{center_label},{XNlabel} > {name}_constraint.txt')

    if os.path.exists(f'./crest_{name}_constraint.sh') == True:
        subprocess.call(['chmod', '777', f'crest_{name}_constraint.sh'])
        subprocess.call([f'./crest_{name}_constraint.sh'])

    file = f'{name}_constraint.txt'
    if os.path.exists(file) == True:
        atoms_line = []
        with open(file, 'r') as f:
            data = f.read()
            lines = data.split('\n')
            for i, line in enumerate(lines):
                if line.find('$metadyn') > -1:
                    atoms_line.append(lines[i+1].split(':')[1])
                    break
        # creates contrain input for crest
        with open(f'{name}_constraint.inp','w+') as f:
            f.write('$constrain \n')
            f.write(f'  distance: {center_label},{Clabel},auto \n')
            f.write(f'  distance: {center_label},{XNlabel},auto \n')
            f.write(f'  distance: {Clabel},{XNlabel},auto \n')
            # f.write(f'  distance: {center_label},{Plabel},auto \n')
            f.write(f'  angle: {Clabel},{center_label},{XNlabel},auto \n')
            f.write('  force constant=1.5 \n')
            f.write('  reference=coord.ref \n')
            f.write('$metadyn \n')
            f.write(f'  atoms:{atoms_line[0]} \n')
            f.write('$end \n')
            os.remove(f'{name}_constraint.txt')
            os.remove(f'crest_{name}_constraint.sh')
            status = True
    else:
        status = False
        print('o Error! constraint.txt file missing so inp file not created.')
    return status

def mol2xtb(name, mol, charge, solvent, path, TS=False, nproc=1):
    ''' Creates script and files to run XTB. Script created runs XTB optimization and hessian calculation. '''

    mol_dir = path+'/'+name
    if not os.path.exists(mol_dir): os.makedirs(mol_dir)
    os.chdir(mol_dir)

    # Create smile, xyz, sdf, turbomole, mol files in directory
    smile = Chem.MolToSmiles(Chem.RemoveHs(mol))
    with open(f'./{name}_smile.smi','w+') as f: f.write(f'{smile}\n')

    Chem.AddHs(mol)
    Chem.MolToXYZFile(mol, f'{name}.xyz')
    Chem.MolToMolFile(mol, f'{name}.mol')
    writer = rdmolfiles.SDWriter(f'{name}.sdf')
    writer.write(mol)
    subprocess.call([ 'obabel', '-imol', f'{name}.mol', '-otmol', '-O', f'{name}_coord'])

    if not os.path.exists(f'optE.thermo'):
        xtb_script = f'./runXTB_{name}.sh'
        with open(xtb_script,'w+') as f:
            f.write('#!/bin/bash \n')
            if TS != False: ## TS geometry
                create_constraints(name, mol, path)  ### Needs tailored to reaction of interest
                f.write(f'xtb {name}.xyz --opt --chrg {charge} --input {name}_constraint.inp --alpb toluene --namespace {name} -P {nproc} > optE.out \n')
                f.write(f'xtb {name}.xtbopt.xyz --chrg {charge} --hess --alpb {solvent} --namespace {name} -P 1 > optE.thermo \n')
            else: ## ground state geometry
                f.write(f'xtb {name}.xyz --chrg {charge} --ohess --alpb toluene --namespace {name} -P {nproc} > optE.thermo \n')
            f.write('\n')

        subprocess.call(['chmod', '+x', xtb_script])
    else: status = False

    if not os.path.exists(f'./runXTB_{name}.sh'): status = False
    else: status = True

    return status

def mol2orca(name, mol, charge, path, xtb=False, TS=False, nproc=4):
    ''' Creates script and files to run XTB. Script created runs XTB optimization and hessian calculation. '''

    mol_dir = path+'/'+name
    if not os.path.exists(mol_dir): os.makedirs(mol_dir)
    os.chdir(mol_dir)

    # Create smile, xyz, sdf, mol, and orca input files in directory
    smile = Chem.MolToSmiles(Chem.RemoveHs(mol))
    with open(f'./{name}_smile.smi','w+') as f: f.write(f'{smile}\n')

    # input xyz coordinates from prior XTB constrained optimization
    if xtb == False:
        Chem.AddHs(mol)
        Chem.MolToXYZFile(mol, f'{name}.xyz')
        Chem.MolToMolFile(mol, f'{name}.mol')
        writer = rdmolfiles.SDWriter(f'{name}.sdf')
        writer.write(mol)
        xyz_text = Chem.MolToXYZBlock(mol).split('\n\n')[1]
        status = True
    else:
        sdfs = glob.glob(f'{name}.xtbtopo.mol')
        if len(sdfs) == 0:
            status = False
            pass
        else:
            mol = SDMolSupplier(sdfs[0], sanitize=False, removeHs=False)
            xyz_text = Chem.MolToXYZBlock(mol[0]).split('\n\n')[1]
            status = True


    # orca calculations
    hess_input = f'{name}.inp'
    opt_input = f'{name}_opt.inp'
    if status != False:
        # orca hess calculation
        with open(hess_input,'w+') as f:
            f.write('# Input for hess \n')
            f.write('! XTB2 \n')
            f.write('! NumFreq \n')
            f.write(f'%pal nprocs {nproc} \n')
            f.write('        end \n')
            f.write('%MaxCore 200 \n')
            f.write(f'*xyz {charge} 1 \n')
            f.write(xyz_text)
            f.write('* \n')

        # orca opt calculation
        with open(opt_input,'w+') as f:
            f.write('# Input for opt \n')
            f.write('! XTB2 \n')
            if TS != False: f.write('! OptTS NumFreq \n')
            else: f.write('! Opt NumFreq \n')
            f.write(f'%pal nprocs {nproc} \n')
            f.write('        end \n')
            f.write('%MaxCore 200 \n')
            f.write('%FREQ TEMP 298.15 \n')
            f.write('CutOffFreq 100 \n')
            f.write('END \n')
            f.write('%geom \n')
            f.write('        inhess Read \n')
            f.write(f'        InHessName "{name}.hess" \n')
            f.write('end \n')
            f.write(f'*xyz {charge} 1 \n')
            f.write(xyz_text)
            f.write('* \n')

    if not os.path.exists(hess_input) and not os.path.exists(opt_input): status = False
    else:
        orca_dir = '/usr/local/ORCA502/orca'
        orca_script1 = f'./runORCA_hess_{name}.sh'
        with open(orca_script1,'w+') as f:
            f.write('#!/bin/bash \n')
            f.write(f'nohup {orca_dir} {hess_input} > {name}.out & \n')
            f.write('\n')\

        orca_script2 = f'./runORCA_opt_{name}.sh'
        with open(orca_script2,'w+') as f:
            f.write('#!/bin/bash \n')
            f.write(f'nohup {orca_dir} {opt_input} > {name}_opt.out & \n')
            f.write('\n')

        subprocess.call(['chmod', '+x', f'{orca_script1}'])
        subprocess.call(['chmod', '+x', f'{orca_script2}'])
        status = True

    return status

def read_xtb_out(name):
    '''
    Reads the output thermo data file after an XTB optimization and frequency calculation.
    Input the name of the molecule (or directory); outputs the Free energies at varying temps, ZPE, number of
    imag frequencies. If any, then returns imag freq value otherwise returns None.
    '''
    file = glob.glob(name)
    os.chdir(file[0])
    free_energies, ZPEs, n_imags, imag_freqs = [], [], [], []

    thermo_file = glob.glob('optE.thermo')[0]
    if not os.path.exists(thermo_file):
        print('No thermo data in directory.');
        free_energies, ZPEs, n_imags, imag_freqs = [], [], [], []
    else:
        with open(thermo_file, 'r') as f:
            data = f.readlines()
            for i in range(0,len(data)):
                if data[i].find('Frequency Printout') > -1:
                    freqs = data[i+4].split()[2:]
                    imag_freq = [i for i in freqs if float(i) < -20]; #print(imag_freq)
                    if len(imag_freq) > 0: imag_freqs.append(imag_freq)
                if data[i].find('# imaginary freq') > -1:
                    n_imag = int(data[i].split()[4]);
                    n_imags.append(n_imag)
                    if n_imag == 0: imag_freqs.append(None)
                if data[i].find(':: zero point energy') > -1:
                    ZPE = float(data[i].split()[4])
                    ZPEs.append(ZPE)
                if data[i].find(':: total free energy') > -1:
                    freeG = float(data[i].split()[4])
                    free_energies.append(freeG)
            if len(free_energies) == 0 and len(ZPEs) == 0:
                for i in range(0,len(data)):
                    if data[i].find(':: total energy') > -1:
                        energy = float(data[i].split()[3])
                        free_energies = [energy]
                    if data[i].find('vibrational frequencies') > -1:
                        freqs = data[i+1].split()[2:]
                        imag_freq = [i for i in freqs if float(i) < -1]
                        if len(imag_freq) == 0:
                            n_imags.append(0)
                            imag_freqs.append(None)
                if len(n_imags) == 0: n_imags.append(0); imag_freqs.append(None)

    return free_energies, ZPEs, n_imags, imag_freqs

def read_orca_out(name):
    '''
    Reads the output file after an ORCA optimization.
    Input the name of the molecule (or directory); returns free energy, imag frequencies (if any), and number of imag. freqs.
    '''
    file = glob.glob(name)
    os.chdir(file[0])
    free_energies, imag_freqs = [], []

    orca_out = glob.glob(f'{name}_opt.out')[0]

    if not os.path.exists(orca_out):
        print('No orca output found in directory.');
        free_energies, n_imags, imag_freqs = [], [], []
    else:
        with open(orca_out, 'r') as f:
            data = f.readlines()
            for i in range(0,len(data)):
                if data[i].find('aborting the run') > -1:
                    print(f'o Error: {orca_out} file aborted run ')
                    free_energies, n_imags, imag_freqs = [], [], []
                    break
                else:
                    if data[i].find('Final Gibbs free energy') > -1:
                        freeG = float(data[i].split()[-2])
                        free_energies.append(freeG)

                    if data[i].find('***imaginary mode***') > -1:
                        imag_freq = float(data[i].split()[1])
                        imag_freqs.append(imag_freq)

            n_imag = len(imag_freqs)

    return free_energies, n_imag, imag_freqs

def create_batch(paths, script='xtb', chunk_size=3):
    '''
    Creates a batch script to run multiple either XTB/CREST/CREGEN/ORCA bash scripts in parallel in its appropriate directories.
    Run created batch script in terminal using commands:
        nohup ./batch_calcs.sh > batch_output.txt &
    '''
    chunked_list = list()
    directories = []
    #### Default chunk_size=3; using 6 nprocs for CREST calcs --> running 18 nprocs parallel
    for i in range(0, len(paths), chunk_size):
        path_chunk = paths[i:i+chunk_size]
        shpaths =[]
        for path in path_chunk:
            name = path.split('/')[-1]
            if script == 'xtb': shpath = path+'/runXTB_'+name+'.sh &'
            elif script == 'crest': shpath = path+'/runCREST_'+name+'.sh &'
            elif script == 'cregen': shpath = path+'/run_cregen.sh &'
            elif script == 'orca1': shpath = path+'/runORCA_hess_'+name+'.sh &'
            elif script == 'orca2': shpath = path+'/runORCA_opt_'+name+'.sh &'
            shpaths.append(shpath)
        chunked_list.append(path_chunk); directories.append(shpaths)

    shfile = f'./batch_calcs_{script}.sh'
    with open(shfile,'w+') as f:
        f.write('#!/bin/bash \n\n')
        f.write(f'echo "Batch Running - {len(paths)} total calculations!!!"\n')
        f.write('STARTTIME=$(date +%s)\n')
        for i, sets in enumerate(chunked_list):
            f.write(f'echo "Running batch {i+1} out of {len(chunked_list)} "\n')
            for j, calc_dir in enumerate(sets):
                calc_name = calc_dir.split('/')[-1]
                calc = directories[i][j]
                if j == len(sets)-1: calc = calc+'& '
                f.write(f'echo "Running: {calc_name}" \n')
                f.write(f'cd {calc_dir} \n')
                f.write(f'bash {calc} \n')
            f.write(f'echo "Done batch {i+1}!" \n')
            f.write('\n')
        f.write('ENDTIME=$(date +%s)\n')
        f.write(f'echo "Calculation process time for {len(paths)} total calcs: $(($ENDTIME - $STARTTIME)) seconds" \n')
    subprocess.call(['chmod', '+x', f'{shfile}'])
    #### subprocess.call([shfile]) #### Dont run on jupyter notebook
    status = True
    return status
