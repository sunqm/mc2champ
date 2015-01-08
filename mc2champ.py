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

def make_head(finp, mc, cidx, csf_sum,
              cutoff_g2q=0.0025, cutoff_d2c=0.025):
    mol = mc.mol
    if isinstance(mc.mo_coeff, numpy.ndarray) and mc.mo_coeff.ndim == 2:
        nmo = mc.mo_coeff.shape[1]
    else:
        nmo = mc.mo_coeff[0].shape[1]
    ncsf = ndet = len(cidx[0])
# should I multiply 2 on nmo, due to the alpha and beta spin?
    finp.write("'ncsf=%d ndet= %d norb= %d" % (ncsf, ndet, nmo*2))
    finp.write("csf_sum=%12.9f cutoff_g2q=%f cutoff_d2c=%f'  title\n" %
               (csf_sum, cutoff_g2q, cutoff_d2c))
    finp.write('''1837465927472523                         irn
0 1                                      iperiodic,ibasis
0.5   -75.9   '  Hartrees'               hb,etrial,eunit
10   100   1   100   0                   nstep,nblk,nblkeq,nconf,nconf_new
0    0    1    -2                        idump,irstar,isite,ipr
6  1.  5.  1.  1.                        imetro delta,deltar,deltat fbias
2 1 1 1 1 0 0 0 0                        idmc,ipq,itau_eff,iacc_rej,icross,icuspg,idiv_v,icut_br,icut_e
50  .1                                   nfprod,tau
0  -1   1  0                             nloc,numr,nforce,nefp\n''')
    finp.write('%d %d           nelec,nup\n' %
               (mol.nelectron, (mol.nelectron+mol.spin)/2))
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

def make_det(finp, mc, cidx):
    mol = mc.mol
    if isinstance(mc.ncore, int):
        ncore = (mc.ncore,mc.ncore)
    else:
        ncore = mc.ncore
    if isinstance(mc.nelecas, int):
        nelecb = (mc.nelecas - mol.spin) / 2
        neleca = mc.nelecas - nelecb
    else:
        neleca, nelecb = mc.nelecas
    if isinstance(mc.mo_coeff, numpy.ndarray) and mc.mo_coeff.ndim == 2:
        nao,nmo = mc.mo_coeff.shape
        mo = (mc.mo_coeff, mc.mo_coeff)
    else:
        nao,nmo = mc.mo_coeff[0].shape
        mo = mc.mo_coeff
    assert(mc.ci.ndim == 2)
    ndet = ncsf = len(cidx[0])

    finp.write("'* Determinantal section'\n")
    finp.write('0 0 tm          inum_orb,iorb_used,iorb_format\n')
    finp.write('%d  %d  %d      ndet,nbasis,nmo\n' %
               (ndet, mol.nao_nr(), nmo))
    for ia in range(mol.natm):
# 2 1s , 2 2s, 4 2px 4 2py 4 2pz 1 3s 0 3px 3py 3pz
# 2   2  4 4 4   1  0 0 0  1 1 1 1 1   0  0 0 0  0 0 0 0 0  0 0 0 0 0 0 0  0  0 0 0  0 0 0 0 0 n1s...4pz,sa,pa,da
        lcounts = [0 for i in range(5)] # up to g function
        for ib in range(mol.nbas):
            if mol.atom_of_bas(ib) == ia:
                l = mol.angular_of_bas(ib)
                lcounts[l] += 1
        for l, li in enumerate(lcounts):
            finp.write('%s   ' % (' '.join([str(li)]*(l*2+1))))
        finp.write(' n1s...4pz,sa,pa,da\n')  # (*)

    label = []
    for ib in range(mol.nbas):
        ia = mol.atom_of_bas(ib)
        l = mol.angular_of_bas(ib)
        nc = mol.nctr_of_bas(ib)
        for n in range(nc):
            for m in range(-l, l+1):
                label.append((ia, l, n, m))
    d_score = {0: 0, 2: 1, -2: 2, 1: 3, -1: 4}
    def ordering(bf1, bf2):
        if label[bf1][0] != label[bf2][0]:
            return label[bf1][0] - label[bf2][0]
        if label[bf1][1] != label[bf2][1]:
            return label[bf1][1] - label[bf2][1]
        if label[bf1][3] != label[bf2][3]:
            m1 = label[bf1][3]
            m2 = label[bf2][3]
            if l < 2:
                return m1 - m2
            elif l == 2:
                return d_score[m1] - d_score[m2]
            else:
# order as l ,-l, (l-1), -(l-1), ...
                if abs(m1) == abs(m2):
                    return -m1 - -m2
                else:
                    return -abs(m1) - -abs(m2)
        return bf1 - bf2
    idx = sorted(range(nao), cmp=ordering)
# the order depends on another flag numr
# if numr =0 or 1, using the order of (*), otherwise, group the AOs
# reordering the MOs, to s s s s s..., px px px px px px, ... py py py py py
    for i in range(nmo):
        finp.write('%s ((coef(j,i),j=1,nbasis),i=1,norb)\n' %
                   ' '.join(map(str, mo[0][idx,i])))
        finp.write('%s ((coef(j,i),j=1,nbasis),i=1,norb)\n' %
                   ' '.join(map(str, mo[1][idx,i])))

#FIXME exp of STO, for alpha beta, using diff orbital indices
    finp.write('??? STO exps???  (zex(i),i=1,nbasis)\n')

    def str2orbidx(string, ncore):
        bstring = bin(string)
        occlst = []
        for i,s in enumerate(bstring):
            if s == '1':
                occlst.append(ncore+i+1)
        return range(1,ncore+1) + occlst
    for k in range(ncsf):
        s = pyscf.fci.cistring.addr2str(mc.ncas, neleca, cidx[0][k])
        finp.write('%s    ' % ' '.join(map(lambda x:str(x*2),
                                           str2orbidx(s,ncore[0]))))
        s = pyscf.fci.cistring.addr2str(mc.ncas, nelecb, cidx[1][k])
        finp.write('%s    ' % ' '.join(map(lambda x:str(x*2+1),
                                           str2orbidx(s,ncore[1]))))
        if k == ncsf-1:
            finp.write('%d (iworbd(iel,idet),iel=1,nelec), label_det(idet)\n' % 0)
        else:
#label_det is never used, set to 0
            finp.write('%d\n' % 0)
    finp.write('%d      ncsf\n' % ncsf)
    finp.write('%s      (csf_coef(icsf),icsf=1,ncsf)\n' %
               ' '.join(map(str, mc.ci[cidx])))
#1.00000000 -0.75690096 -0.56812240 -0.37934694 0.24317056 0.40655818 -0.06949017 -0.00390204 -0.22953891 0.18572625 0.09740735 -0.35067648 0.00031633 -0.00008407 -0.00483767 -0.00351984 (csf_coef(icsf),icsf=1,ncsf)
    finp.write('%s      (ndet_in_csf(icsf),icsf=1,ncsf)\n' % ('1 '*ncsf))
    for i in range(ncsf):
        finp.write('%d\n' % (i+1)) # index
        finp.write('1.\n')         # coeff for symm-adapted csf basis

# put 1 1 1 1,..., here
#orbitals
# symmetry
#1 2 1 3 4 2 1 6 5 2 3 4 1 2 6 5 1 2 3 4 7 8 9 10 1 6 5 2 1 1 1 1 2 2 2 2 6 6 5 5 3 3 4 4
# end

def make_champ_input(inputname, casscf, tol=.01):
    with open(inputname, 'w') as finp:
        cidx = numpy.where(abs(casscf.ci)>tol)
        csf_sum = 0.987654321
        make_head(finp, casscf, cidx, csf_sum,
                  cutoff_g2q=0.0025, cutoff_d2c=0.025)
        make_det(finp, casscf, cidx)

if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import mcscf
    mol = gto.Mole()
    mol.build(
        verbose = 0,
        atom = [
            ["O", (0., 0.,  0.7)],
            ["O", (0., 0., -0.7)],],

        basis = 'cc-pvdz',
        spin = 2,
    )

    mf = scf.UHF(mol)
    mf.scf()

    mc = mcscf.CASSCF(mol, mf, 4, (4,2))
    mc.mc1step()

    make_champ_input('example.inp', mc)
