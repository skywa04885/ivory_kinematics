pub mod leg;
pub mod torso;

#[derive(Debug)]
pub enum Error {
    LegNumberOutOfBounds(u8),
    PseudoInverse(&'static str),
    UnreachableTargetPosition(f64, nalgebra::Vector3<f64>)
}

