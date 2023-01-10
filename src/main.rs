use std::time::Instant;

use ivory_kinematics::{
    model::{arm::Arm, leg::Leg, torso::Torso},
    solver::Solver,
};
use nalgebra::Vector3;

fn main() {
    let torso: Torso = Torso::builder().build();
    let arm: Arm = Arm::builder().build();
    let legs: [Leg; 4] = [
        Leg::builder(0)
            .thetas(Vector3::<f64>::new(
                0.0,
                f64::to_radians(45.0),
                f64::to_radians(45.0),
            ))
            .lengths(Vector3::<f64>::new(2.0, 10.0, 10.0))
            .build(),
        Leg::builder(1)
            .thetas(Vector3::<f64>::new(
                0.0,
                f64::to_radians(45.0),
                f64::to_radians(45.0),
            ))
            .lengths(Vector3::<f64>::new(2.0, 10.0, 10.0))
            .build(),
        Leg::builder(2)
            .thetas(Vector3::<f64>::new(
                0.0,
                f64::to_radians(45.0),
                f64::to_radians(45.0),
            ))
            .lengths(Vector3::<f64>::new(2.0, 10.0, 10.0))
            .build(),
        Leg::builder(3)
            .thetas(Vector3::<f64>::new(
                0.0,
                f64::to_radians(45.0),
                f64::to_radians(45.0),
            ))
            .lengths(Vector3::<f64>::new(2.0, 10.0, 10.0))
            .build(),
    ];
    let mut solver: Solver = Solver::builder(torso, legs, arm).build();

    // {
    //     let a = Instant::now();
    //     let c = solver.fk_vertices_for_leg(0);
    //     let b = a.elapsed();
    //     println!("{:?}, {:#?}", b, c);
    // }

    // {
    //     println!("{:#?}", solver.fk_paw_ef_position_for_leg(0).unwrap());
    //     let a = Instant::now();
    //     let c = solver.move_paw_relative(0, &Vector3::<f64>::new(1.1, 5.1, 1.1), Some(0.01));
    //     let b = a.elapsed();
    //     println!(
    //         "{:?}, {:#?}, {:#?}",
    //         b,
    //         c,
    //         solver.fk_paw_ef_position_for_leg(0).unwrap()
    //     );
    // }

    {
        let end_effector_position = solver.fk_arm_ef_position().unwrap();
        let target_end_effector_position =
            end_effector_position + nalgebra::Vector3::<f64>::new(-3.0, -6.0, 1.0);
        let a = Instant::now();
        let c = solver.ik_arm_ef_position_with_eps(&target_end_effector_position, Some(0.0001));
        let b = a.elapsed();
        println!("{:?}, {:?}", b, c,);
    }
}
