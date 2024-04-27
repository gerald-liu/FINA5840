from functools import partial
def pv_annuity(pmt, n, r):
    return pmt*(1-(1+r)**-n)/r

def fv_annuity(pmt, n, r):
    return pmt*((1+r)**n-1)/r

def bond_price(c, mat, freq, r):
    n = mat * freq
    r = r/freq
    c = c/freq
    return pv_annuity(c, n, r) + (1+r)**-n

def pv(cf, r): #cf is a list
    n = len(cf)
    sum_pv = 0.0
    for i in range(n):
        sum_pv += cf[i]/(1+r)**(i+1)
    return sum_pv

def pv1(cf, tl, r):
    n = len(cf)
    if len(cf)!=len(tl):
        print('length does not align')
    else:
        sum_pv = 0.0
        for i in range(n):
            sum_pv += cf[i] / (1 + r) ** tl[i]
        return sum_pv

def IRR(c, mat, freq, p):
    tol = 1e-10
    r0 = 1e-4
    incr = 0.1
    r1 = 0.1
    i = 0
    i_max = 10000
    bpp = partial(bond_price, c, mat, freq)
    while bpp(r1)-p >0:
        r1 += incr
    found = False
    while not found and i<i_max:
        r_temp = (r0 + r1)/2
        if bpp(r_temp)-p < 0:
            r1 = r_temp
        else:
            r0 = r_temp
        if abs(bpp(r_temp)-p) < tol:
            found = True
        i +=1
    return r_temp

def IRR_cf(cf, p):
    tol = 1e-10
    r0 = 1e-4
    incr = 0.1
    r1 = 0.1
    i = 0
    i_max = 10000
    bpp = partial(pv, cf)
    while bpp(r1)-p >0:
        r1 += incr
    found = False
    while not found and i<i_max:
        r_temp = (r0 + r1)/2
        if bpp(r_temp)-p < 0:
            r1 = r_temp
        else:
            r0 = r_temp
        if abs(bpp(r_temp)-p) < tol:
            found = True
        i +=1
    return r_temp

def IRR_cf1(cf, tl, p):
    tol = 1e-10
    r0 = 1e-4
    incr = 0.1
    r1 = 0.1
    i = 0
    i_max = 10000
    bpp = partial(pv1, cf, tl)
    while bpp(r1)-p >0:
        r1 += incr
    found = False
    while not found and i<i_max:
        r_temp = (r0 + r1)/2
        if bpp(r_temp)-p < 0:
            r1 = r_temp
        else:
            r0 = r_temp
        if abs(bpp(r_temp)-p) < tol:
            found = True
        i +=1
    return r_temp



