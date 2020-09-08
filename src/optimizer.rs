use optimization_engine;
use optimization_engine::core::Optimizer;
use std::sync::{Arc, Mutex};
use nalgebra::VectorN;
use std::thread;
use std::time::Duration;

pub trait CostFunction {
    fn call(&self, x : &[f64]) -> f64;
    fn grad(&self, x : &[f64], grad : &mut [f64]) -> f64;
}

impl<F> CostFunction for F 
    where F: Fn(&[f64])->f64 {

    fn call(&self, x : &[f64]) -> f64 {
        self(x)
    }

    fn grad(&self, x : &[f64], grad : &mut [f64]) -> f64{
        let f_0 = CostFunction::call(self, x);
        let h = 1e-8;
        for i in 0..x.len() {
            let mut x_h = Vec::from(x);
            x_h[i] += h;
            grad[i] = (CostFunction::call(self, &x_h) - f_0) / h;
        };
        f_0
    }
}

pub struct OnlineOptimizer {
    problem_size : usize, tolerance : f64, lbfgs_memory_size : usize,
    cost_function : Mutex<Arc<dyn CostFunction + Send + Sync>>,
    running : Mutex<bool>,
    current_value : Mutex<Vec<f64>>
}

impl<'a> OnlineOptimizer {
    pub fn new(problem_size : usize, tolerance : f64, lbfgs_memory_size : usize) -> Self {
        OnlineOptimizer {
            problem_size, 
            tolerance, 
            lbfgs_memory_size, 
            cost_function : Mutex::new(Arc::new(|x : &[f64]|->f64 {VectorN::<f64, nalgebra::dimension::Dynamic>::from_column_slice(x).norm_squared()})),
            running : Mutex::new(false),
            current_value : Mutex::new(Vec::new())
        }
    }

    pub fn set_cost_function(&self, cost : Arc<dyn CostFunction + Send + Sync>) {
        *self.cost_function.lock().unwrap() = cost;
    }

    pub fn get_current_value(&self) -> Vec<f64> {
        self.current_value.lock().unwrap().clone()
    }

    pub fn run(&self, x_start : Vec::<f64>) {
        let mut cache = optimization_engine::panoc::PANOCCache::new(self.problem_size, self.tolerance, self.lbfgs_memory_size);
        *self.current_value.lock().unwrap() = x_start.clone();
        self.set_cost_function(Arc::new(
            move |x: &[f64]| -> f64 {
                (VectorN::<f64, nalgebra::dimension::Dynamic>::from_column_slice(x) 
                - 
                VectorN::<f64, nalgebra::dimension::Dynamic>::from_column_slice(&x_start)).norm_squared()
            }
        ));
        *self.running.lock().unwrap() = true;
        while *self.running.lock().unwrap() {
            let cost_function = (*self.cost_function.lock().unwrap()).clone();

            let df = |x: &[f64], grad: &mut [f64]| -> Result<(), optimization_engine::SolverError> {
                cost_function.grad(x, grad);
                Ok(())
            };
    
            let f = |x: &[f64], c: &mut f64| -> Result<(), optimization_engine::SolverError> {
                *c = cost_function.call(x);
                Ok(())
            };

            let constraints = optimization_engine::constraints::NoConstraints::new();
            let problem = optimization_engine::Problem::new(
                &constraints, 
                df,
                f
            );
            let mut op = optimization_engine::panoc::PANOCOptimizer::new(problem, &mut cache).with_max_iter(10);
            let mut x = (*self.current_value.lock().unwrap()).clone();

            op.solve(x.as_mut_slice()).unwrap();
            *self.current_value.lock().unwrap() = x;

        }
    }

    pub fn stop(&self) {
        *self.running.lock().unwrap() = false;
    }
}
