use std::time::Instant;

use ivory_kinematics::model::{self, Torso};
use nalgebra::Vector3;

fn main() {
    let torso: Torso = Torso::builder().build();

    {
        let a = Instant::now();
        let c = torso.fk_paw_ef_position(0);
        let b = a.elapsed();
        println!("{:?}, {:#?}", b, c);
    }

    {
        let a = Instant::now();
        let c = torso.ik_paw_ef_position(0, Vector3::<f64>::new(0.01, 0.01, 0.01));
        let b = a.elapsed();
        println!("{:?}, {:#?}", b, c);
    }
}
