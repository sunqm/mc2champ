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
#       mc2champ.make_champ_input('inputname', mc, blabel=[...])
#

import numpy
import pyscf.fci
import pyscf.lib.parameters as param

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
0 1                                      iperiodic,ibasis\n''')
    finp.write('''0.5   %.15g   '  Hartrees'    hb,etrial,eunit\n''' % mc.e_tot)
    finp.write('''10   100   1   100   0                   nstep,nblk,nblkeq,nconf,nconf_new
0    0    1    -2                        idump,irstar,isite,ipr
6  1.  5.  1.  1.                        imetro delta,deltar,deltat fbias
2 1 1 1 1 0 0 0 0                        idmc,ipq,itau_eff,iacc_rej,icross,icuspg,idiv_v,icut_br,icut_e
50  .1                                   nfprod,tau
0  -1   1  0                             nloc,numr,nforce,nefp\n''')
    finp.write('%d %d           nelec,nup\n' %
               (mol.nelectron, (mol.nelectron+mol.spin)/2))
    finp.write("'* Geometry section'\n")
    finp.write('3               ndim\n')
    nctype = len(mol._basis)
    cpair = zip(sorted(mol._basis.keys()), range(1,nctype+1))
    ctypemap = dict(cpair)
    finp.write('%d %d           nctype,ncent\n' % (nctype,mol.natm))
    finp.write('%s (iwctype(i),i=1,ncent)\n' %
               ' '.join(['%d'%ctypemap[mol.atom_symbol(i)] for i in range(mol.natm)]))
    finp.write('%s (znuc(i),i=1,nctype)\n' %
               ' '.join(['%f'%param.NUC[i[0]] for i in cpair]))
    for i in range(mol.natm):
        coord = mol.atom_coord(i)
        finp.write('%f %f %f ' % tuple(coord))
        finp.write(' %d          ((cent(k,i),k=1,3),i=1,ncent)\n' %
                   ctypemap[mol.atom_symbol(i)])

def label_sto(finp, mol, shell_ids):
    if shell_ids:
        for symb in sorted(mol._basis.keys()):
# 2 1s , 2 2s, 4 2px 4 2py 4 2pz 1 3s 0 3px 3py 3pz
# 2   2  4 4 4   1  0 0 0  1 1 1 1 1   0  0 0 0  0 0 0 0 0  0 0 0 0 0 0 0  0  0 0 0  0 0 0 0 0 n1s...4pz,sa,pa,da
            sh = shell_ids[symb]
            lcounts = [[0 for l in range(6)] for shell in range(7)] # up to g function
            for n, bi in enumerate(mol._basis[symb]):
                l = bi[0]
                lcounts[sh[n]][l] += len(bi[1]) - 1
            for lsh, shcount in enumerate(lcounts):
                for l, li in enumerate(shcount):
                    if l+1 <= lsh:
                        finp.write('%s   ' % (' '.join([str(li)]*(l*2+1))))
            finp.write(' n1s...4pz,sa,pa,da\n')  # (*)
    else:
        finp.write(' ?? n1s...4pz,sa,pa,da\n')

def forder(label):
    d_score = {0: 0, 2: 1, -2: 2, 1: 3, -1: 4}
    def ordering(bf1, bf2):
        if label[bf1][0] != label[bf2][0]:
            return label[bf1][0] - label[bf2][0]
        if label[bf1][1] != label[bf2][1]:
            return label[bf1][1] - label[bf2][1]

        l = label[bf1][1]
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
    return ordering

def make_det(finp, mc, cidx, basis_label):
    mol = mc.mol
    shell_ids = dict([(k,v[0]) for k, v in basis_label.items()])
    zexps = dict([(k,v[1]) for k, v in basis_label.items()])
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
    label_sto(finp, mol, shell_ids)

    label = []
    shell_count = 0
    for ib in range(mol.nbas):
        ia = mol.bas_atom(ib)
        l = mol.bas_angular(ib)
        nc = mol.bas_nctr(ib)
        for n in range(nc):
            for m in range(-l, l+1):
                label.append((ia, l, n, m))
            shell_count += 1
    idx = sorted(range(nao), cmp=forder(label))
    for symb, bs in mol._basis.items():
        label = []
        shell_count = 0
        nbf = 0
        for ib in bs:
            l = ib[0]
            nc = len(ib[1]) - 1
            for n in range(nc):
                for m in range(-l, l+1):
                    label.append((1, l, n, m, shell_count))
                shell_count += 1
            nbf += (l*2+1)*nc
        idx = sorted(range(nbf), cmp=forder(label))
        iwrwf = [str(label[i][4]+1) for i in idx]
        finp.write('%s   (iwrwf(ib),ib=1,nbastyp)\n' % ' '.join(iwrwf))
# the order depends on another flag numr
# if numr =0 or 1, using the order of (*), otherwise, group the AOs
# reordering the MOs, to s s s s s..., px px px px px px, ... py py py py py
    for i in range(nmo):
        if i == 0:
            finp.write('%s     ((coef(j,i),j=1,nbasis),i=1,norb)\n' %
                       ' '.join(map(str, mo[0][idx,i])))
        else:
            finp.write('%s\n' %
                       ' '.join(map(str, mo[0][idx,i])))
        finp.write('%s\n' % ' '.join(map(str, mo[1][idx,i])))

    if zexps:
        for symb in sorted(mol._basis.keys()):
            finp.write('%s  (zex(i),i=1,nbasis)\n' %
                       ' '.join(map(str, zexps[symb])))
    else:
        finp.write('??? STO exps???  (zex(i),i=1,nbasis)\n')

    def str2orbidx(string, ncore):
        bstring = bin(string)
        occlst = []
        for i,s in enumerate(bstring):
            if s == '1':
                occlst.append(ncore+i)
        return range(0,ncore) + occlst
    for k in range(ncsf):
        s = pyscf.fci.cistring.addr2str(mc.ncas, neleca, cidx[0][k])
        finp.write('%s    ' % ' '.join(map(lambda x:str(x*2+1),
                                           str2orbidx(s,ncore[0]))))
        s = pyscf.fci.cistring.addr2str(mc.ncas, nelecb, cidx[1][k])
        finp.write('%s    ' % ' '.join(map(lambda x:str(x*2+2),
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

def make_champ_input(inputname, casscf, blabel, tol=.001):
    with open(inputname, 'w') as finp:
        cidx = numpy.where(abs(casscf.ci)>tol)
        csf_sum = numpy.linalg.norm(casscf.ci[cidx])**2
        make_head(finp, casscf, cidx, csf_sum,
                  cutoff_g2q=0.0025, cutoff_d2c=0.025)
        make_det(finp, casscf, cidx, blabel)

def parse_basis(bastr):
    LMAP = {'S':0, 'P':1, 'D':2, 'F':3, 'G':4}
    bs = []
    bnow = None
    shell_ids = []
    zexps = []
    for bline in bastr.split('\n'):
        if bline.strip():
            bdat = bline.split()
            if len(bdat) == 6: # start a new STO-nG
                bs.append(bnow)
                l = LMAP[bdat[1][-1]]
                zexp = float(bdat[3])
                gexp = float(bdat[4])
                gc   = float(bdat[5])
                shell_ids.append(int(bdat[1][:-1]))
                zexps.append(zexp)
                bnow = [l, (gexp, gc)]
            else:
                gexp = float(bdat[3])
                gc   = float(bdat[4])
                bnow.append((gexp, gc))
    bs.append(bnow)
    return bs[1:], (shell_ids, zexps)

if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import mcscf
    from pyscf.tools import dump_mat
    mol = gto.Mole()
    mol.build(
        verbose = 5,
        output = 'o2.out',
        atom = '''
            O  0 0  0.7
            O  0 0 -0.7''',
        basis = 'cc-pvdz',
        spin = 2,
    )

    mf = scf.UHF(mol)
    mf.scf()

    mc = mcscf.CASSCF(mol, mf, 4, (4,2))
    mc.natorb = True
    mc.kernel()
    mc.analyze()

    make_champ_input('example.inp', mc, {})

