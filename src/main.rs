use std::time::Instant;

use ivory_kinematics::{
    model::{leg::Leg, torso::Torso},
    solver::Solver,
};
use nalgebra::Vector3;

fn main() {
    let torso: Torso = Torso::builder()
        .legs([
            Leg::builder(0)
                .thetas(Vector3::<f64>::new(
                    0.0,
                    f64::to_radians(0.0),
                    f64::to_radians(0.0),
                ))
                .lengths(Vector3::<f64>::new(10.0, 10.0, 10.0))
                .build(),
            Leg::builder(1)
                .thetas(Vector3::<f64>::new(
                    0.0,
                    f64::to_radians(0.0),
                    f64::to_radians(0.0),
                ))
                .lengths(Vector3::<f64>::new(10.0, 10.0, 10.0))
                .build(),
            Leg::builder(2)
                .thetas(Vector3::<f64>::new(
                    0.0,
                    f64::to_radians(0.0),
                    f64::to_radians(0.0),
                ))
                .lengths(Vector3::<f64>::new(10.0, 10.0, 10.0))
                .build(),
            Leg::builder(3)
                .thetas(Vector3::<f64>::new(
                    0.0,
                    f64::to_radians(0.0),
                    f64::to_radians(0.0),
                ))
                .lengths(Vector3::<f64>::new(10.0, 10.0, 10.0))
                .build(),
        ])
        .build();
    let mut solver: Solver = Solver::builder(torso).build();

    {
        let a = Instant::now();
        let c = solver.fk_vertices_for_leg(0);
        let b = a.elapsed();
        println!("{:?}, {:#?}", b, c);
    }

    {
        let a = Instant::now();
        let c = solver.ik_paw_ef_position_for_leg(0, &Vector3::<f64>::new(0.01, 0.01, 0.01), 0.001);
        let b = a.elapsed();
        println!("{:?}, {:#?}", b, c);
    }

    {
        println!("{:#?}", solver.fk_paw_ef_position_for_leg(0).unwrap());
        let a = Instant::now();
        let c = solver.move_paw_relative(0, &Vector3::<f64>::new(1.1, 5.1, 1.1), Some(0.01));
        let b = a.elapsed();
        println!("{:?}, {:#?}, {:#?}", b, c, solver.fk_paw_ef_position_for_leg(0).unwrap());
    }
}
