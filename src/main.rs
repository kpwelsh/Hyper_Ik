mod robot;
mod optimizer;
use optimizer::CostFunction;
use nalgebra::{base::allocator::Allocator, DefaultAllocator, Rotation3, Vector3, Transform3, Translation, Isometry3, Matrix4, Matrix3, UnitQuaternion, Quaternion, MatrixMN, VectorN, U1, U3, U7};
use std::time::{Duration, Instant};
use std::sync::{Arc, Mutex};
use std::thread;
use pyo3::prelude::*;
extern crate openvr;

fn lst_sqr(A : MatrixMN<f64, U7, U7>, b : MatrixMN<f64, U7, U1>) -> MatrixMN<f64, U7, U1> {
    (A.transpose() * A + 1.0 * MatrixMN::<f64, U7, U7>::identity()).try_inverse().unwrap() * A.transpose() * b
}

fn update_cost_function(r : robot::Robot, op : &Arc<optimizer::OnlineOptimizer>, trans : Isometry3<f64>) {
    //let R : Matrix3<f64> = //Matrix3::from(trans.fixed_slice::<U3, U3>(0,0));
    let target_q = trans.rotation;// UnitQuaternion::from_matrix(&R);
    let target_x : Vector3<f64> = trans.translation.vector;
    let mut grad = [0.0; 7];
    let current_x = op.get_current_value();
    
    let cost = |x: &[f64]| -> f64 {
        let iso = r.transform(x, "panda_hand", None);
        let (q, p) = (iso.rotation, iso.translation);
        let (_, a, _) = (q.inverse() * target_q).polar_decomposition();
        (2.0 * a).powi(2) + (target_x - p.vector).norm_squared()
    };
    CostFunction::grad(
        &cost,
        &current_x,
        &mut grad
    );

    let J = VectorN::<f64, U7>::from_column_slice(&grad);
    let target_joint_angles = VectorN::<f64, U7>::from_column_slice(&current_x) + lst_sqr(J * J.transpose(), - J * cost(&current_x));

    op.set_cost_function(Arc::new(
        move |x: &[f64]| -> f64 {
            let iso = r.transform(x, "panda_hand", None);
            let (q, p) = (iso.rotation, iso.translation);
            let (_, a, _) = (q.inverse() * target_q).polar_decomposition();
            (2.0 * a).powi(2) + (target_x - p.vector).norm_squared()
        }
        // |x: &[f64]| -> f64 {
        //     (VectorN::<f64, U7>::from_column_slice(x) - target_joint_angles).norm_squared()
        // }
    ));
}

fn get_controller_input(system : &openvr::System, controller_index : u32) -> Option<(u64, Isometry3<f64>)> {
    match system.controller_state_with_pose(openvr::TrackingUniverseOrigin::RawAndUncalibrated, controller_index) {
        None => {
            println!("No controller input");
            None
        },
        Some((state, pose)) => {
            let m : &[[f32; 4]; 3] = pose.device_to_absolute_tracking();
            let transform = Isometry3::from_parts(
                Translation::<f64, U3>::new((m[0][3]).into(), m[1][3].into(), (m[2][3]).into()),
                UnitQuaternion::from_matrix(&Matrix3::new(
                    (m[0][0]).into(), (m[0][1]).into(), (m[0][2]).into(),
                    m[1][0].into(), m[1][1].into(), m[1][2].into(),
                    (m[2][0]).into(), (m[2][1]).into(), (m[2][2]).into()
                ))
            );
            let button_pressed = state.button_pressed;
            Some((button_pressed, transform))
        }
    }
}

fn main() -> Result<(), ()> {
    let gil = Python::acquire_gil();
    let py = gil.python();
    main_(py).map_err(|e| {
        e.print_and_set_sys_last_vars(py);
    })
}

fn main_(py : Python<'_>) -> PyResult<()> {
    let r = robot::Robot::new("../panda.urdf");

    let op = optimizer::OnlineOptimizer::new(
        7,
        1e-12,
        100
    );

    let op = Arc::new(op);
    let op_1 = Arc::clone(&op);

    thread::spawn(move || {
        op_1.run(vec![0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0]);
    });

    let context = unsafe { openvr::init(openvr::ApplicationType::Scene) }.unwrap();
    let system = context.system().unwrap();

    let controller_index = loop {
        match system.tracked_device_index_for_controller_role(openvr::TrackedControllerRole::RightHand) {
            None => {
                println!("Waiting for controller to connect");
                thread::sleep(Duration::from_millis(1000));
            },
            Some(v) => break v
        }
    };
    println!("{}", controller_index);

    let sim = PyModule::import(py, "simInterface")?;
    sim.call0("connect")?;
    println!("homing");
    sim.call1("commandJoints", (op.get_current_value(),))?;
    thread::sleep(Duration::from_millis(1000));
    println!("homed");

    let init_trans = r.transform(&op.get_current_value(), "panda_hand", None);
    let (_, trans) = get_controller_input(&system, controller_index).unwrap();
    let mut offset_transform = trans.inverse();
    loop {
        match get_controller_input(&system, controller_index) {
            None => {},
            Some((button_pressed, transform)) => {
                if (button_pressed & (1 << 2)) > 0 {
                    offset_transform = transform.inverse();
                }
                if (button_pressed & (1 << 33)) > 0 { 
                    sim.call0("grip")?;
                }

                println!("raw: {}", transform.translation);
                let transform = init_trans
                    * offset_transform 
                    * transform;
                println!("{}", transform.translation);
                
                let qs = op.get_current_value();
                sim.call1("commandJoints", (qs,))?;
                update_cost_function(r.clone(), &op, transform);
            }
        }
    }
}
