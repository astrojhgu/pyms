#![allow(unused_variables)]
#![allow(unused_imports)]

use num_traits::float::FloatConst;
use pyo3::prelude::*;
use pyo3::types::{PyList};
use map_solver::brute_mo::MappingProblem as BruteSkySolverMo;
use map_solver::naive_mo::MappingProblem as NaiveSkySolverMo;
use numpy::{ToPyArray, IntoPyArray, PyArrayDyn, PyArray1, PyArray2};
use ndarray::{ArrayView1, Array1};
use linear_solver::io::{RawEntry, RawMM};
use scorus::coordinates::sphcoord::SphCoord;
use simobs::gridder::Gridder;
use std::iter::FromIterator;

#[pyclass]
pub struct PyCsMatF64{
    data: sprs::CsMat<f64>
}

#[pymethods]
impl PyCsMatF64{
    fn nrows(&self, py: Python)-> PyResult<usize>{
        Ok(self.data.rows())
    }

    fn ncols(&self, py: Python)->PyResult<usize>{
        Ok(self.data.cols())
    }

    fn internal_data(&self, py: Python)->(Py<PyArray1<usize>>,Py<PyArray1<usize>>, Py<PyArray1<f64>>){
        let rm=RawMM::from_sparse(&self.data);
        let values=Array1::from_iter(rm.entries.iter().map(|x|{
            x.value
        })).into_pyarray(py).to_owned();
        let rows=Array1::from_iter(rm.entries.iter().map(|x|{
            x.i
        })).into_pyarray(py).to_owned();
        let cols=Array1::from_iter(rm.entries.iter().map(|x|{
            x.j
        })).into_pyarray(py).to_owned();
        (rows, cols, values)
    }
}

impl std::convert::From<sprs::CsMat<f64>> for PyCsMatF64{
    fn from(x: sprs::CsMat<f64>)->PyCsMatF64{
        PyCsMatF64{
            data: x
        }
    }
}

#[pyclass]
pub struct PyBruteMo{
    solver: BruteSkySolverMo
}

pub fn get_csmat(h: usize, w: usize, data: ArrayView1<f64>, row: ArrayView1<usize>, col: ArrayView1<usize>)->PyCsMatF64{
        let entries:Vec<_>=data.iter().zip(row.iter().zip(col.iter())).map(|(&x, (&i, &j))|{
            RawEntry::new(i, j, x)
        }).collect();


        RawMM{
            height: h, 
            width: w,
            storage: linear_solver::io::Storage::Sparse,
            qual: linear_solver::io::Qualifier::General,
            entries
        }.to_sparse().into()
    }

#[pymodule]
fn native(_py: Python, m: &PyModule)->PyResult<()>{
    #[pyfn(m, "get_csmat")]
    fn get_csmat_py(py: Python, h: usize, w: usize, data: &PyArray1<f64>, row: &PyArray1<usize>, col: &PyArray1<usize>)->PyCsMatF64{
        get_csmat(h, w, data.as_array(), row.as_array(), col.as_array())
    }

    #[pyfn(m, "empty_brute_solver_mo")]
    fn compose_solver_mo(py: Python, tol: f64, m_max: usize)->PyBruteMo
    {
        PyBruteMo{
            solver: BruteSkySolverMo::empty().with_tol(tol).with_m_max(m_max)
        }
    }

    #[pyfn(m, "add_obs")]
    fn add_obs(py: Python, solver: &mut PyBruteMo, 
    ptr: &PyCsMatF64,tod: &PyArray1<f64>, noise: &PyArray1<f64>){
        solver.solver.add_obs(ptr.data.clone(), tod.as_array().to_owned(), noise.as_array().to_owned())
    }

    #[pyfn(m, "brute_solve")]
    fn brute_solve(py: Python, solver: &mut PyBruteMo)->Py<PyArray1<f64>>{
        match solver.solver.x{
            None=>{
                println!("No initial guess set, use naive solve to initial it");
                let mp = NaiveSkySolverMo::new(solver.solver.ptr_mat.clone(), solver.solver.tod.clone())
                    .with_tol(1e-10)
                    .with_m_max(50);
                    solver.solver.set_init_value(mp.solve_sky())
            }
            _ => {}
        }
        solver.solver.solve_sky(100, None).into_pyarray(py).to_owned()
    }

    #[pyfn(m, "auto_fov_center")]
    fn auto_fov_center(py: Python, ra: &PyArray1<f64>, dec: &PyArray1<f64>)->(f64, f64){
        let ra=ra.as_array().map(|&x|{x.to_radians()});
        let dec=dec.as_array().map(|&x|{x.to_radians()});   
        let (ra, dec)=simobs::utils::auto_fov_center(ra.as_slice().unwrap(), dec.as_slice().unwrap());
        (ra.to_degrees(), dec.to_degrees())
    }


    #[pyfn(m, "define_pixels")]
    fn define_pixels(py: Python, ra_list: &PyArray1<f64>, dec_list: &PyArray1<f64>, fov_center_ra: f64, fov_center_dec: f64, pix_size_deg: f64)->(PyCsMatF64, Py<PyArray2<i64>>){
        let ra=ra_list.as_array().map(|&x| x.to_radians());
        let dec=dec_list.as_array().map(|&x| x.to_radians());

        let sph_list: Vec<_> = ra
        .iter()
        .zip(dec.iter())
        .map(|(&r, &d)| SphCoord::new(f64::PI() / 2.0 - d, r))
        .collect();

        let fov_center=SphCoord::new(f64::PI() / 2.0 - fov_center_dec.to_radians(), fov_center_ra.to_radians());

        let step=pix_size_deg.to_radians();
        let gridder = Gridder::new(fov_center.pol, fov_center.az, step, step);
        let (m, pix_idx) = gridder.get_ptr_matrix(&sph_list);
        let pix_idx=pix_idx.map(|&x| x as i64);
        (PyCsMatF64{data: m}, pix_idx.into_pyarray(py).to_owned())
    }

    #[pyfn(m, "define_pixels_mo")]
    fn define_pixels_mo(py: Python, ra_list: &PyList, dec_list: &PyList, fov_center_ra: f64, fov_center_dec: f64, pix_size_deg: f64)->(Vec<PyCsMatF64>, Py<PyArray2<i64>>)
    {
        let fov_center=SphCoord::new(f64::PI() / 2.0 - fov_center_dec.to_radians(), fov_center_ra.to_radians());

        let step=pix_size_deg.to_radians();
        
       let mut sph_lists:Vec<_>=Vec::new();

        for (ra, dec) in ra_list.iter().zip(dec_list.iter()){
            let ra1=<&PyArray1::<f64> as FromPyObject>::extract(ra).unwrap().as_array().map(|&x| x.to_radians());
            let dec1=<&PyArray1::<f64> as FromPyObject>::extract(dec).unwrap().as_array().map(|&x| x.to_radians());

            let sph_list:Vec<_> = ra1
                .iter()
                .zip(dec1.iter())
                .map(|(&r, &d)| SphCoord::new(f64::PI() / 2.0 - d, r))
                .collect();

            sph_lists.push(sph_list);
        }
        
        let sph_list_ref:Vec<_>=sph_lists.iter().map(|x|{&x[..]}).collect();
        let gridder = Gridder::new(fov_center.pol, fov_center.az, step, step);
        let (ptr_mat_vec, pix_idx)=gridder.get_ptr_matrix_mo(&sph_list_ref[..]);
        //let ff=PyList::new(py, ptr_mat_vec.iter());
        let pix_idx=pix_idx.map(|&x| x as i64);
        let ptr_mat:Vec<_>=ptr_mat_vec.into_iter().map(|m| PyCsMatF64{data: m}).collect();

        (ptr_mat, pix_idx.into_pyarray(py).to_owned())
    }

    Ok(())
}
