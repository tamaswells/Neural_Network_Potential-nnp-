# -*- coding: utf-8 -*-
"""Tools for interfacing with `ASE`_.

.. _ASE:
    https://wiki.fysik.dtu.dk/ase
"""

from __future__ import absolute_import
import torch
import ase.neighborlist
from torchani import utils
import ase.calculators.calculator
import ase.units
import copy
import numpy as np
import torchani

def return_global():
    return new_energy
def fill_missing_energies(energy,species,_all_species):
    #species_list=species.numpy().flatten()
    global new_energy 
    #set_species_list=sorted(set(species_list[species_list>-1]))
    new_energy=[0.0]*(max(_all_species)+1)
    for i in range(len(_all_species)):
        new_energy[_all_species[i]]=energy[i]
    return torch.tensor(new_energy, dtype=torch.float64)

# @torch.jit.script
def present_species(species):
    """Given a vector of species of atoms, compute the unique species present.

    Arguments:
        species (:class:`torch.Tensor`): 1D vector of shape ``(atoms,)``

    Returns:
        :class:`torch.Tensor`: 1D vector storing present atom types sorted.
    """
    # present_species, _ = species.flatten()._unique(sorted=True)
    present_species = species.flatten().unique(sorted=True)
    if present_species[0].item() == -1:
        present_species = present_species[1:]
    return present_species

class EnergyShifter2(torchani.utils.EnergyShifter):
    def __init__(self, self_energies, fit_intercept=False):
        super(EnergyShifter2, self).__init__(self_energies, fit_intercept)

    def sae_from_dataset(self, atomic_properties, properties):
        """Compute atomic self energies from dataset.

        Least-squares solution to a linear equation is calculated to output
        ``self_energies`` when ``self_energies = None`` is passed to
        :class:`torchani.EnergyShifter`
        """
        species = atomic_properties['species']
        global all_species
        species_list = species.numpy().flatten()
        all_species = sorted(set(species_list[species_list>-1]))
        energies = properties['energies']
        present_species_ = present_species(species)
        X = (species.unsqueeze(-1) == present_species_).sum(dim=1).to(torch.double)
        
        # Concatenate a vector of ones to find fit intercept
        if self.fit_intercept:
            X = torch.cat((X, torch.ones(X.shape[0], 1).to(torch.double)), dim=-1)
        y = energies.unsqueeze(dim=-1)
        coeff_, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        return coeff_.squeeze()

    def sae(self, species2):
        intercept = 0.0
        if self.fit_intercept:
            intercept = self.self_energies[-1]
        #species2=species2-species2.min()
        
        if self.self_energies.dim()==0:
            self.self_energies=self.self_energies.unsqueeze(0)  
        s=self.self_energies
        
        self.self_energies=fill_missing_energies(self.self_energies,species2,all_species)

        #self.self_energies=torch.tensor([-37.8693,0,0, -37.8693], dtype=torch.float64)
        self_energies = self.self_energies[species2]
        self_energies[species2 == -1] = 0
        self.self_energies=s
        
        return self_energies.sum(dim=1) + intercept    
    def forward(self, species_energies):
        """(species, molecular energies)->(species, molecular energies + sae)
        """
        species, energies = species_energies
        sae = self.sae(species).to(energies.dtype).to(energies.device)
        return species, energies + sae

class Calculator2(ase.calculators.calculator.Calculator):
    """TorchANI calculator for ASE

    Arguments:
        species (:class:`collections.abc.Sequence` of :class:`str`):
            sequence of all supported species, in order.
        aev_computer (:class:`torchani.AEVComputer`): AEV computer.
        model (:class:`torchani.ANIModel` or :class:`torchani.Ensemble`):
            neural network potential models.
        energy_shifter (:class:`torchani.EnergyShifter`): Energy shifter.
        dtype (:class:`torchani.EnergyShifter`): data type to use,
            by dafault ``torch.float64``.
        overwrite (bool): After wrapping atoms into central box, whether
            to replace the original positions stored in :class:`ase.Atoms`
            object with the wrapped positions.
    """

    implemented_properties = ['energy', 'forces', 'stress', 'free_energy']

    def __init__(self, species, aev_computer, model, energy_shifter,shift,converter,dtype=torch.float64, overwrite=False):
        super(Calculator2, self).__init__()
        self.species_to_tensor = utils.ChemicalSymbolsToInts(species)
        # aev_computer.neighborlist will be changed later, so we need a copy to
        # make sure we do not change the original object
        self.shift_=shift
        self.converter=converter
        self.aev_computer = copy.deepcopy(aev_computer)
        self.model = copy.deepcopy(model)
        self.energy_shifter = copy.deepcopy(energy_shifter)
        self.overwrite = overwrite

        self.device = self.aev_computer.EtaR.device
        self.dtype = dtype

        self.whole = torch.nn.Sequential(
            self.aev_computer,
            self.model,
            self.energy_shifter
        ).to(dtype)

    @staticmethod
    def strain(tensor, displacement, surface_normal_axis):
        displacement_of_tensor = torch.zeros_like(tensor)
        for axis in range(3):
            displacement_of_tensor[..., axis] = tensor[..., surface_normal_axis] * displacement[axis]
        return displacement_of_tensor

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=ase.calculators.calculator.all_changes):
        super(Calculator2, self).calculate(atoms, properties, system_changes)
        cell = torch.tensor(self.atoms.get_cell(complete=True),
                            dtype=self.dtype, device=self.device)
        pbc = torch.tensor(self.atoms.get_pbc(), dtype=torch.bool,
                           device=self.device)
        pbc_enabled = pbc.any().item()
        species = self.species_to_tensor(self.atoms.get_chemical_symbols()).to(self.device)
        
        species = species.unsqueeze(0)
        coordinates = torch.tensor(self.atoms.get_positions())
        coordinates = coordinates.unsqueeze(0).to(self.device).to(self.dtype) \
                                 .requires_grad_('forces' in properties)

        if pbc_enabled:
            coordinates = utils.map2central(cell, coordinates, pbc)
            if self.overwrite and atoms is not None:
                atoms.set_positions(coordinates.detach().cpu().reshape(-1, 3).numpy())

        if 'stress' in properties:
            displacements = torch.zeros(3, 3, requires_grad=True,
                                        dtype=self.dtype, device=self.device)
            displacement_x, displacement_y, displacement_z = displacements
            strain_x = self.strain(coordinates, displacement_x, 0)
            strain_y = self.strain(coordinates, displacement_y, 1)
            strain_z = self.strain(coordinates, displacement_z, 2)
            coordinates = coordinates + strain_x + strain_y + strain_z

        if pbc_enabled:
            if 'stress' in properties:
                strain_x = self.strain(cell, displacement_x, 0)
                strain_y = self.strain(cell, displacement_y, 1)
                strain_z = self.strain(cell, displacement_z, 2)
                cell = cell + strain_x + strain_y + strain_z
            _, energy = self.whole((species, coordinates, cell, pbc))
        else:
            _, energy = self.whole((species, coordinates))
        
        self.results['energy'] = (energy.item())*self.converter

        self.results['free_energy'] = (energy.item())*self.converter

        if 'forces' in properties:
            forces = -torch.autograd.grad(energy.squeeze(), coordinates)[0]
            self.results['forces'] = forces.squeeze().to('cpu').numpy()

        if 'stress' in properties:
            volume = self.atoms.get_volume()
            stress = torch.autograd.grad(energy.squeeze(), displacements)[0] / volume
            self.results['stress'] = stress.cpu().numpy()
