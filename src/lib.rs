//! # ivory_kinematics
//! `ivory_kinematics` is a small crate used for performing forward/ and inverse kinematics in the
//! ivory robotic dog project. The target of this crate is to quickly perform these computations,
//! which actually, on modern AVX/SIMD supporting CPUs take less than 300 micro seconds.

pub mod model;
pub mod solver;

pub use self::{
    model::{
        leg::{Leg, LegBuilder},
        torso::{Torso, TorsoBuilder},
        arm::{Arm, ArmBuilder}
    },
    solver::{Solver, SolverBuilder},
};

#[derive(Debug)]
pub enum Error {
    LegNumberOutOfBounds(u8),
    PseudoInverse(&'static str),
    UnreachableTargetPosition(f64, nalgebra::Vector3<f64>),
}
