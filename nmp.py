#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import random
from scipy import constants as con
from scipy import stats  
from optparse import OptionParser 
import numpy as np
import re
import h5py
import glob
import os


class Gaussian(object):
    def __init__(self):
        self.force_constants = []
        self.normal_mode_coord = []
        self.all_atoms=0
        self.all_forces=0
        self.energy=0
        self.atom_label=[]
        self.displacement_coeff=0
        self.count=1
        self.temp=0
        self.element = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', \
                   'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', \
                   'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', \
                   'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', \
                   'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', \
                   'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', \
                   'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', \
                   'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', \
                   'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
    
    def read_gaussian_for_normal_mode(self, gau_file):
        """
        read Atomic coordinate, all normal mode coordinates and
            force constant from Gaussian freq output.
        Returns:
            None
        """
        coord_line_num = []
        freq_line_num = []
        with open(gau_file, 'r') as reader:
            coord_key_word = "Coordinates (Angstroms)"
            freq_key_word = "Frequencies"
            all_content = reader.readlines()
            for index, line in enumerate(all_content):
                if coord_key_word in line:
                    coord_line_num.append(index)
                elif freq_key_word in line:
                    freq_line_num.append(index)
            atoms = []
            for i in all_content[coord_line_num[-1] + 3:]:
                if "---" not in i:
                    self.atom_label.append(self.element[int(i.split()[1]) - 1])
                    atoms.append(list(map(float, i.split()[3:6])))
                else:
                    break
            atom_nums = len(atoms)
            self.all_atoms = np.array(atoms)
            count1 = 0
            count2 = 0
            for i in all_content[freq_line_num[0]:]:
                if "Atom" in i:
                    break
                elif "Frc consts" in i:
                    count2 = count1
                else:
                    count1 += 1
            #print(count1,count2)
            for i in freq_line_num:
                freq_num_line = len(all_content[i].split()) - 2
                self.force_constants.extend(list(map(float, all_content[i + count2].split()[-1 * freq_num_line:])))
                tmp = []
                for j in all_content[i + count1 + 2:i + count1 + 2 + atom_nums]:
                    tmp.append((list(map(float, j.split()[-1 * freq_num_line * 3:]))))
                tmp2 = np.array(tmp)
                split_point = tmp2.shape[-1] // 3
                if split_point <= 1:
                    self.normal_mode_coord.append(tmp2)
                else:
                    split_point_at = np.arange(1, split_point) * 3
                    for k in np.split(tmp2, split_point_at, 1):
                        self.normal_mode_coord.append(k)

    def read_gaussian_for_energy_and_force(self, gau_file):
        """
        read Atomic coordinate, all normal mode coordinates and
            force constant from Gaussian freq output.
        Returns:
            None
        """
        coord_line_num = []
        force_line_num = []
        energy_line_num = []
        self.atom_label=[]
        Normal_termination = False
        with open(gau_file, 'r') as reader:
            coord_key_word = "Coordinates (Angstroms)"
            force_key_word = "Forces (Hartrees/Bohr)"
            energy_key_word =  "SCF Done:"
            Normal_termination_key_word = "Normal termination"
            all_content = reader.readlines()
            for index, line in enumerate(all_content):
                if coord_key_word in line:
                    coord_line_num.append(index)
                elif force_key_word in line:
                    force_line_num.append(index)
                elif energy_key_word in line:
                    energy_line_num.append(index)
                elif Normal_termination_key_word in line:
                    Normal_termination = True
            if Normal_termination == False:
                return False
            atoms = []
            for i in all_content[coord_line_num[-1] + 3:]:
                if "---" not in i:
                    self.atom_label.append(self.element[int(i.split()[1]) - 1])
                    atoms.append(list(map(float, i.split()[3:6])))
                else:
                    break
            #print(self.atom_label)
            atom_nums = len(atoms)
            self.all_atoms = np.around(atoms, decimals=3)   
            pattern=r"SCF Done:.*?E.*?=(.*?)A\.U\."
            self.energy = float(re.findall(pattern,all_content[energy_line_num[-1]])[0])
            #print(self.all_forces)
            #print(self.energy)            
            if force_line_num==[]:
                return (True,False)
            #print(force_line_num)
            forces = []
            for i in all_content[force_line_num[-1] + 3:]:
                if "---" not in i:
                    #self.atom_label.append(self.element[int(i.split()[1]) - 1])
                    forces.append(list(map(float, i.split()[2:5])))
                else:
                    break
            #atom_nums = len(atoms)
            self.all_forces = np.array(forces) 
            return (True,True)


    def write_gaussian_input(self,T,charge,spin_multiplicity,self_determine,scaling_factor,freq):
        """
        Write new gaussian input files.
        Returns:
            None
        """        
        self.temp=T
        if self_determine:
            self.displacement_=np.random.uniform(-1.0*scaling_factor,scaling_factor,len(self.normal_mode_coord))
        else:
            self.displacement_coeff=self.gen_displacement(np.array(self.force_constants))
        for i in range(len(self.normal_mode_coord)):
            if self_determine:
                displaced_atoms = self.all_atoms + self.displacement_[i]  * self.normal_mode_coord[i]
            else:
                #self.displacement_coeff=R 
                #normalize normal_mode_coord to unitless vector
                coeff=1.0*self.normal_mode_coord[i]/np.sqrt(np.sum(np.square(self.normal_mode_coord[i]),axis=1))\
                                                   .reshape(self.normal_mode_coord[i].shape[0],-1)  
                displaced_atoms = self.all_atoms + self.displacement_coeff[i] * coeff              
            freqs='freq' if freq else ''
            for spin in spin_multiplicity:
                to_be_writen=[]
                with open(str(self.count)+'-'+str(spin)+'.gjf','w') as writer:
                    to_be_writen.append("%mem=100GB")
                    to_be_writen.append("%nprocshared=24")
                    to_be_writen.append("# m062x/def2tzvp empiricaldispersion=gd3 nosymm guess=mix %s\n" %(freqs))
                    to_be_writen.append("By NXU\n")
                    to_be_writen.append("%d %d" %(charge,spin))
                    for k in range(len(displaced_atoms)):
                        to_be_writen.append(' %s %10.4f %10.4f %10.4f' %(self.atom_label[k],displaced_atoms[k][0],\
                            displaced_atoms[k][1],displaced_atoms[k][2]))
                    to_be_writen.append("\n\n")
                    writer.writelines([i+'\n' for i in to_be_writen])

            self.count=self.count+1

    def gen_displacement(self,read_in_frc):
        """
        q=Rq.
        Returns:
            array(R ret) Q = R*Q
        """         
        len_Q = len(read_in_frc)
        k_b = con.k  # 1.38064852E-23 # m²*kg*s⁻²*K⁻¹ Boltzman Constant
        Temperature = self.temp  # Temperature
        beta = 1 / (k_b * Temperature)
        #read_in_frc = 1.2485  # mdyne/A
        # 1 mdyne/A = 1e-8/1e-10 N/m = 100 N/m
        random_num=np.random.uniform(0.0,1.0/len_Q,len_Q)
        force_constant = read_in_frc * 100.0
        random_sign = [-1.0 if i == 0 else 1.0 for i in stats.bernoulli.rvs(0.5, size=len_Q)]
        Ret = random_sign*np.sqrt(3.0 * random_num * len(self.all_atoms) * k_b * Temperature / force_constant)*1e10
        #unit Angstrom
        return Ret

def write_training_data(instance,spin_multiplicity,gau_log_suffix):
    energy_list=[]
    coordinates_list=[]
    force_list=[]
    species=''
    gau_out_files=glob.glob('*-*.%s'%(gau_log_suffix))
    if gau_out_files == []:
        print("No batch gaussian outputs found!")
        sys.exit(1)
    gau_out_files_seq=list(map(lambda x:int(x.split("-")[0]),gau_out_files))
    spin_multiplicity_=set(list(map(lambda x:int(x.split("-")[1].split('.')[0]),gau_out_files))) 
    #print(sorted(gau_out_files_seq))
    for i in sorted(set(gau_out_files_seq)):
        tmp_energy=0
        for k in spin_multiplicity_:
            if not os.path.exists("%d-%d.log" %(i,k)):
                continue
            ret=instance.read_gaussian_for_energy_and_force("%d-%d.log" %(i,k))
            if ret and instance.energy<tmp_energy:
                tmp_energy=instance.energy
                tmp_coordinate=instance.all_atoms
                species=instance.atom_label
                if ret[1]:
                    tmp_force=instance.all_forces
        if tmp_energy==0:
            continue
        energy_list.append(tmp_energy)
        coordinates_list.append(tmp_coordinate)
        if ret and ret[1]:
            force_list.append(tmp_force)

    if sys.version[0]=='2':
        vlen=unicode
    else:
        vlen=str

    dt = h5py.special_dtype(vlen=vlen)
    f=h5py.File("my.h5","w")
    g1=f.create_group("first")
    # print(np.array(coordinates_list).shape)
    # print(np.array(energy_list).shape)
    # print(np.array(species))
    g1.create_dataset("coordinates",data=np.array(coordinates_list))
    g1.create_dataset("energies",data=np.array(energy_list))
    if len(force_list)==len(energy_list):
        g1.create_dataset('forces',data=np.array(force_list))
    ds =g1.create_dataset("species",np.array(species).shape , dtype=dt)
    ds[:] = np.array(species)

def opt_parsers():
    parser = OptionParser()  

    parser.add_option("-n", "--nmp",  
                      action="store_true", dest="nmp", default=False,  
                      help="Using normal mode sampling to sampling configurations!") 

    parser.add_option("-s", "--scale",  
                      dest="scale", default='false',  
                      help="Scaling factor for nmp.") 

    parser.add_option("-T", "--tem",  
                      dest="temperature", default=298.15,type="float",  
                      help="Temperature for generating training data.")     

    parser.add_option("-b", "--batch",  
                      dest="batch", default=1, type="int", 
                      help="Will produce batch*num_normal_mode_coodinate samples.") 

    parser.add_option("-l", "--gau",  
                      dest="gau_log", default='',  
                      help="Specify gaussian output for nmp.") 

    parser.add_option("-c", "--charge",  
                      dest="charge", default=0,type="int",  
                      help="Charge of the molecule.") 

    parser.add_option("-m", "--spin",  
                      dest="spin_multiplicity", default='1',  
                      help="Spin multiplicity of the molecule. eg. 1.3.5 for multi-multiplicities.") 

    parser.add_option("-q", "--freq",  
                      action="store_true", dest="freq", default=False,  
                      help="Whether to calculate forces in samplings!") 

    parser.add_option("-r", "--read",  
                      action="store_true", dest="read", default=False,  
                      help="Read energies,coordinate (and forces if exist) from Gaussian outputs!") 

    parser.add_option("--merge",  
                      action="store_true", dest="merge", default=False,  
                      help="Merge hdf5 files.") 

    return parser.parse_args()


if __name__ == "__main__":

    if len(sys.argv)<=1:
        print("HELP: python nmp.py -h for help!")
        sys.exit(1)
    #script,logfile=sys.argv
    (options,args) = opt_parsers()
    gauss=Gaussian()
    self_determine=False
    charge=options.charge
    batch = options.batch
    scaling_factor=0
    try:
        if options.scale != 'false':
            self_determine=True
            scaling_factor=float(options.scale)
        if options.spin_multiplicity != '1':
            spin_multiplicity=list(map(int,options.spin_multiplicity.split('.'))) 
        else:
            spin_multiplicity=[1]
    except:
        raise ValueError('Wrong arguments!')
    if options.nmp and options.read:
        print("Can not use -n and -r simultaneously!")
        sys.exit(1)
    elif not options.nmp and not options.read and not options.merge:
        print("please use either -n or -r!")
        sys.exit(1)
    
    if options.nmp:
        print("Sampling....")
        if not os.path.exists(options.gau_log):
            print('Gaussian out does not exists')
            sys.exit(1)
        gauss.read_gaussian_for_normal_mode(options.gau_log)    

        for j in range(batch):
            gauss.write_gaussian_input(options.temperature,charge,spin_multiplicity,self_determine,scaling_factor,options.freq)            
        print("Sampling finished.")

    if options.read:
        print("Extract energies from batch gaussian output.")
        gau_log_suffix='log'
        if not os.path.exists(options.gau_log):
            print('Read gaussian output with .log suffix by default.')
        else:
            try:
                gau_log_suffix=options.gau_log.split('.')[-1]
            except:
                print('Specify a gaussian output with -l')
                sys.exit(0)
        write_training_data(gauss,spin_multiplicity,gau_log_suffix)
        print("Training dataset my.h5 has been generated. Bye.")

    if options.merge:
        if os.path.exists('new.h5'):
            import shutil
            shutil.move('new.h5','new.h5.bak')
        h5file=glob.glob('*.h5')+glob.glob('*.hdf5')
        if sys.version[0]=='2':
            vlen=unicode
        else:
            vlen=str
        coords_all=[]
        energys_all=[]
        species_all=[]
        forces_all=[]    
        for i in h5file:
            f=h5py.File(i,"r")
            keys=list(f.keys())
            for key in keys:
                try:
                    coord=np.array(f[key]['coordinates'])
                    energy=np.array(f[key]['energies'])
                    species=list(f[key]['species'])
                    if 'forces' in f[key].keys():   
                        forces=np.array(f[key]['forces'])
                        forces_all.append(forces)
                    #species=[i.decode() for i in species]
                    species_data = np.array(species)
                    coords_all.append(coord)
                    energys_all.append(energy)
                    species_all.append(species_data)
                except:
                    continue
            f.close() 
        dt = h5py.special_dtype(vlen=vlen)
        f=h5py.File("new.h5","w")
        count=1
        for i in range(len(species_all)):
            g1=f.create_group(str(count))
            g1.create_dataset("coordinates",data=coords_all[i])
            g1.create_dataset("energies",data=energys_all[i])
            ds =g1.create_dataset("species",species_all[i].shape , dtype=dt)
            ds[:] = species_all[i] 
            count+=1
        f.close()
        print("Successfully merged hdf5 files.")
            


