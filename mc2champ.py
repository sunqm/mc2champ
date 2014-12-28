#
# There are two ways to call this script with the pyscf MCSCF output.
#
# 1. Add the current directory to PYTHONPATH, to directly load this module
#       import mc2champ
#
# 2. Copy this script to /path/to/pyscf, then load this script by
#       from pyscf impor mc2champ
#
# After the normal pyscf calculation, call make_champ_input to generate CHAMP
# input, e.g
#
#       from pyscf import gto
#       from pyscf import scf
#       from pyscf import mcscf
#       mol = gto.Mole()
#       mol.build(
#           verbose = 5
#           atom = [
#               ["O", (0., 0.,  0.7)],
#               ["O", (0., 0., -0.7)],]
#
#           basis = 'cc-pvdz',
#           spin = 2,
#       )
#
#       mf = scf.UHF(mol)
#       mf.scf()
#
#       mc = mcscf.CASSCF(mol, m, 4, (4,2))
#       mc.mc1step()
#
#       import mc2champ
#       mc2champ.make_champ_input('inputname', mc)
#

import numpy
import pyscf.fci

def make_head(finp, mol, ncsf, ndet, norb, csf_sum,
              cutoff_g2q=0.0025, cutoff_d2c=0.025):
    finp.write("'ncsf=%d ndet= %d norb= %d" % (ncsf, ndet, norb))
    finp.write("csf_sum=%12.9f cutoff_g2q=%f cutoff_d2c=0.025'  title\n" %
               (csf_sum, cutoff_g2q, cutoff_d2c))
    finp.write('''1837465927472523                         irn
0 1                                      iperiodic,ibasis
0.5   -75.9   '  Hartrees'               hb,etrial,eunit
10   100   1   100   0                   nstep,nblk,nblkeq,nconf,nconf_new
0    0    1    -2                        idump,irstar,isite,ipr
6  1.  5.  1.  1.                        imetro delta,deltar,deltat fbias
2 1 1 1 1 0 0 0 0                        idmc,ipq,itau_eff,iacc_rej,icross,icuspg,idiv_v,icut_br,icut_e
50  .1                                   nfprod,tau
0  -1   1  0                             nloc,numr,nforce,nefp
12 6 	 	 	 	 	 nelec,nup\n''')
    finp.write("'* Geometry section'\n")
    finp.write('3               ndim\n')
    finp.write('%d %d           nctype,ncent\n' % (mol.natm,mol.natm))
    finp.write('%s (iwctype(i),i=1,ncent)\n' %
               ' '.join(map(str, range(1,mol.natm+1)))) #TODO: add symm-eq centers
    finp.write('%s (znuc(i),i=1,nctype)\n' %
               ' '.join(map(str, [mol.charge_of_atm(i) for i in range(mol.natm)])))
    for i in range(mol.natm):
        coord = mol.coord_of_atm(i)
        finp.write('%f %f %f' % tuple(coord))
        finp.write('%d          ((cent(k,i),k=1,3),i=1,ncent)\n' % i)

def make_det(finp, mol, mo, ci, norb, nelec, tol=.1):
    if isinstance(nelec, int):
        nelecb = (nelec - mol.spin) / 2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    cidx = numpy.where(abs(ci)>tol)
    ncsf = ci.size
    ndet = ci.size

    finp.write("'* Determinantal section'\n")
    finp.write('0 0 tm          inum_orb,iorb_used,iorb_format\n')
    finp.write('%d  %d  %d      ndet,nbasis,norb\n' %
               (ndet, mol.nao_nr(), norb))
#FIXME ns np nd
    finp.write('2   2  4 4 4   1  0 0 0  1 1 1 1 1   0  0 0 0  0 0 0 0 0  0 0 0 0 0 0 0  0  0 0 0  0 0 0 0 0 n1s...4pz,sa,pa,da\n')
    nmo = mo.shape[1]
    for i in range(nmo):
        finp.write('%s ((coef(j,i),j=1,nbasis),i=1,norb)\n' %
                   ' '.join(map(str, mo[:,i])))
#FIXME exp of STO
    finp.write('??? exponential of STO,???  (zex(i),i=1,nbasis)\n')
    def str2orbidx(string):
        bstring = bin(string)
        occlst = []
        for i in range(norb):
            if bstring[i] == '1':
                occlst.append(i+1)
        return occlst
    finp.write('%s (csf_coef(icsf),icsf=1,ncsf)\n' % ' '.join(map(str, ci[cidx])))
    for k in range(ndet):
#FIXME label_det?
        # (iworbd(iel,idet),iel=1,nelec), label_det(idet)
        s = pyscf.fci.cistring.addr2str(norb, neleca, cidx[0][k])
        finp.write('%s  ' % ' '.join(map(str, str2orbidx(s))))
        s = pyscf.fci.cistring.addr2str(norb, nelecb, cidx[1][k])
        finp.write('%s  ' % ' '.join(map(str, str2orbidx(s))))
        finp.write('%d\n' % i)
    finp.write('%d      ncsf\n' % ncsf)
#1.00000000 -0.75690096 -0.56812240 -0.37934694 0.24317056 0.40655818 -0.06949017 -0.00390204 -0.22953891 0.18572625 0.09740735 -0.35067648 0.00031633 -0.00008407 -0.00483767 -0.00351984 (csf_coef(icsf),icsf=1,ncsf)
    finp.write('%s      (ndet_in_csf(icsf),icsf=1,ncsf)\n' % ('1 '*ncsf))
    for i in range(ncsf):
        finp.write('%d\n' i)    # index
        finp.write('1\n')       # coeff

def make_champ_input(inputname, casscf):
    make_head(finp, mol, ncsf, ndet, norb, csf_sum,
              cutoff_g2q=0.0025, cutoff_d2c=0.025)
    make_det(finp, mol, mo, casscf.ci, ncsf, ndet, casscf.ncas)

if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import mcscf
    mol = gto.Mole()
    mol.build(
        verbose = 5
        atom = [
            ["O", (0., 0.,  0.7)],
            ["O", (0., 0., -0.7)],]

        basis = 'cc-pvdz',
        spin = 2,
    )

    mf = scf.UHF(mol)
    mf.scf()

    mc = mcscf.CASSCF(mol, m, 4, (4,2))
    mc.mc1step()

    make_champ_input('example.inp', mc)
