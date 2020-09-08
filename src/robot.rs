use nalgebra::{Quaternion, UnitQuaternion, Vector3, Isometry3, Translation3, Unit};
use std::collections::HashMap;
extern crate urdf_rs;

#[derive(Clone)]
struct Joint {
    parent_frame : String,
    joint_type : urdf_rs::JointType,
    translation : Translation3<f64>,
    fixed_rotation : UnitQuaternion<f64>,
    rotation_axis : Unit::<Vector3<f64>>
}

#[derive(Clone)]
pub struct Robot {
    joints : HashMap<String, Joint>
}

impl Robot {
    pub fn new(urdf : &str) -> Self {
        let mut robot = Robot {joints : HashMap::<String, Joint>::new()};
        let urdf_robot : urdf_rs::Robot = urdf_rs::read_file(urdf).unwrap();
        for joint in urdf_robot.joints {
            robot.joints.insert(joint.child.link,
                Joint {
                    parent_frame : joint.parent.link.clone(), 
                    joint_type : joint.joint_type, 
                    translation : Translation3::from(Vector3::from(joint.origin.xyz)),
                    fixed_rotation : UnitQuaternion::from_euler_angles(
                        joint.origin.rpy[0],
                        joint.origin.rpy[1],
                        joint.origin.rpy[2]
                    ),
                    rotation_axis : Unit::new_normalize(Vector3::from(joint.axis.xyz))
                }
            );
        }
        robot
    }

    pub fn transform(&self, joint_values: &[f64], end_link : &str, start_link : Option<&str>) -> Isometry3<f64> {
        let mut current_link = end_link;
        let mut isometry = Isometry3::identity();

        let mut i = (joint_values.len() - 1) as i32;
        loop {
            if let Some(s) = start_link {
                if s == current_link {
                    break;
                }
            }
            let joint = self.joints.get(current_link);
            match joint {
                None => {
                    match start_link {
                        None => break,
                        Some(_) => panic!("Could not find joint with child link: {}", current_link)
                    }
                },
                Some(j) => {
                    current_link = &j.parent_frame;

                    match j.joint_type {
                        urdf_rs::JointType::Fixed => {

                        },
                        urdf_rs::JointType::Revolute => {
                            if i < 0 {
                                panic!("Not enough joint values provided");
                            }
                            isometry = Isometry3::from_parts(
                                Translation3::new(0.0,0.0,0.0),
                                UnitQuaternion::from_axis_angle(&j.rotation_axis, joint_values[i as usize])
                            ) * isometry;
                            i -= 1;
                        },
                        _ => panic!("Unsupported joint type")
                    }
                    isometry = Isometry3::from_parts(j.translation, j.fixed_rotation) * isometry;
                }
            }
        };

        if i > 0 {
            panic!("Too many joint values provided: {}", i)
        }

        isometry
    }
}