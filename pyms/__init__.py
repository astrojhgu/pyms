from . import native
import numpy as np
from scipy.sparse.coo import coo_matrix
import matplotlib.pylab as plt
import itertools
import functools
def sparse2csmat(mat):
    col=mat.col
    row=mat.row
    data=mat.data
    h,w=mat.shape
    return native.get_csmat(h, w, data.astype('float64'), row.astype(np.uint), col.astype(np.uint))

def csmat2sparse(mat):
    (i, j, v)=mat.internal_data()
    rows=mat.nrows()
    cols=mat.ncols()
    return coo_matrix((v, (i,j)), shape=(rows, cols))
    

def brute_solver_mo(ptr_mats, tods, noises, tol=1e-10, m_max=50):
    solver=native.empty_brute_solver_mo(tol, m_max);
    for p, t, n in zip(ptr_mats, tods, noises):
        h,w=p.shape
        assert(h==t.shape[0])
        assert(t.shape==n.shape)
        p1=sparse2csmat(p)
        native.add_obs(solver, p1, t.astype('float64'), n.astype('float64'))
    return solver

def solve(s):
    return native.brute_solve(s)

def define_pixels(ra_deg, dec_deg, pixel_size, fov_center_deg=None):
    fov_center_ra, fov_center_dec=fov_center_deg if fov_center_deg is not None else native.auto_fov_center(ra_deg, dec_deg)
    (ptr_mat, pixels)=native.define_pixels(ra_deg.astype('float64'), dec_deg.astype('float64'), fov_center_ra, fov_center_dec, pixel_size)
    return (csmat2sparse(ptr_mat), pixels)

def define_pixels_mo(ra_deg_list, dec_deg_list, pixel_size, fov_center_deg=None):
    ra_deg_flatten=np.array(list(itertools.chain(*ra_deg_list)))
    dec_deg_flatten=np.array(list(itertools.chain(*dec_deg_list)))
    print(ra_deg_flatten)
    print(dec_deg_flatten)
    fov_center_ra, fov_center_dec=fov_center_deg if fov_center_deg is not None else native.auto_fov_center(ra_deg_flatten, dec_deg_flatten)

    ra_deg_list=[i.astype('float64') for i in ra_deg_list]
    dec_deg_list=[i.astype('float64') for i in dec_deg_list]

    (ptr_mat, pixels)=native.define_pixels_mo(ra_deg_list, dec_deg_list, fov_center_ra, fov_center_dec, pixel_size)
    return ([csmat2sparse(i) for i in ptr_mat], pixels)

def filter_ptr_mat(mat_list, pix_list, th):
    h=functools.reduce(lambda x,y:x+y, map(lambda x: np.sum(x, axis=0),mat_list))
    h=np.array(h).squeeze()
    cols=h>th
    new_pmat=[i for i in map(lambda x: x.tocsr()[:, cols].tocoo(), mat_list)]
    print(len(new_pmat))
    for i in new_pmat:
        print(i.shape)
    return (new_pmat, np.array(pix_list)[cols].tolist())

    
def define_pixels_mo_gaussian(ra_deg_list, dec_deg_list, pixel_size, rcut_deg, sigma_deg, fov_center_deg=None, hit_th=None):
    ra_deg_flatten=np.array(list(itertools.chain(*ra_deg_list)))
    dec_deg_flatten=np.array(list(itertools.chain(*dec_deg_list)))
    print(ra_deg_flatten)
    print(dec_deg_flatten)
    fov_center_ra, fov_center_dec=fov_center_deg if fov_center_deg is not None else native.auto_fov_center(ra_deg_flatten, dec_deg_flatten)

    ra_deg_list=[i.astype('float64') for i in ra_deg_list]
    dec_deg_list=[i.astype('float64') for i in dec_deg_list]

    (ptr_mat, pixels)=native.define_pixels_mo_gaussian(ra_deg_list, dec_deg_list, fov_center_ra, fov_center_dec, pixel_size, rcut_deg, sigma_deg)
    return ([csmat2sparse(i) for i in ptr_mat], pixels)

def plot_hit_map(ptr_mat, pixel_list):
    h=np.array(np.sum(ptr_mat, axis=0)).squeeze()
    #print(np.max(np.sum(ptr_mat, axis=0)))
    i_min=int(np.min(pixel_list[:,0]))
    i_max=int(np.max(pixel_list[:,0]))
    j_min=int(np.min(pixel_list[:,1]))
    j_max=int(np.max(pixel_list[:,1]))

    image=np.zeros([i_max-i_min+1, j_max-j_min+1])

    for n in range(0, pixel_list.shape[0]):
        i,j=pixel_list[n,:]
        image[i-i_min, j-j_min]=h[n]

    plt.imshow(image, aspect='auto')
    plt.colorbar()

def plot_hit_map_mo(ptr_mat_list, pixel_list):        
    h=functools.reduce(lambda x,y:x+y, map(lambda x: np.sum(x, axis=0),ptr_mat_list))
    h=np.array(h).squeeze()
    #print(np.max(np.sum(ptr_mat, axis=0)))
    i_min=int(np.min(pixel_list[:,0]))
    i_max=int(np.max(pixel_list[:,0]))
    j_min=int(np.min(pixel_list[:,1]))
    j_max=int(np.max(pixel_list[:,1]))

    image=np.zeros([i_max-i_min+1, j_max-j_min+1])

    for n in range(0, pixel_list.shape[0]):
        i,j=pixel_list[n,:]
        image[i-i_min, j-j_min]=h[n]

    plt.imshow(image, aspect='auto')
    plt.colorbar()

def fill_map(values, pixel_list, pm_list=None, th=None):
    i_min=int(np.min(pixel_list[:,0]))
    i_max=int(np.max(pixel_list[:,0]))
    j_min=int(np.min(pixel_list[:,1]))
    j_max=int(np.max(pixel_list[:,1]))

    image=np.zeros([i_max-i_min+1, j_max-j_min+1])*np.NaN

    if pm_list is None:
        for (i,j,v) in zip(pixel_list[:,0], pixel_list[:,1], values):
            image[i-i_min,j-j_min]=v
    else:
        h=functools.reduce(lambda x,y:x+y, map(lambda x: np.sum(x, axis=0),pm_list))
        h=np.array(h).squeeze()
        for (i, j, v, c) in zip(pixel_list[:,0], pixel_list[:, 1], values, h):
            if c > th:
                image[i-i_min, j-j_min]=v


    return image