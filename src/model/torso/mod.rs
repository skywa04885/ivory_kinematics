use super::leg::Leg;

#[derive(Debug, Clone)]
pub struct Torso {
    orientation: nalgebra::Vector3<f64>,
    position: nalgebra::Vector3<f64>,
    dimensions: nalgebra::Vector2<f64>,
    legs: [super::leg::Leg; 4],
}

impl Torso {
    #[inline(always)]
    pub fn new(
        orientation: nalgebra::Vector3<f64>,
        position: nalgebra::Vector3<f64>,
        dimensions: nalgebra::Vector2<f64>,
        legs: [super::leg::Leg; 4],
    ) -> Self {
        Self {
            orientation,
            position,
            dimensions,
            legs,
        }
    }

    #[inline(always)]
    pub fn builder() -> TorsoBuilder {
        TorsoBuilder::new()
    }

    #[inline(always)]
    pub fn orientation_mut(&mut self) -> &mut nalgebra::Vector3<f64> {
        &mut self.orientation
    }

    #[inline(always)]
    pub fn orientation(&self) -> &nalgebra::Vector3<f64> {
        &self.orientation
    }

    #[inline(always)]
    pub fn position_mut(&mut self) -> &mut nalgebra::Vector3<f64> {
        &mut self.position
    }

    #[inline(always)]
    pub fn position(&self) -> &nalgebra::Vector3<f64> {
        &self.position
    }

    #[inline(always)]
    pub fn dimensions_mut(&mut self) -> &mut nalgebra::Vector2<f64> {
        &mut self.dimensions
    }

    #[inline(always)]
    pub fn dimensions(&self) -> &nalgebra::Vector2<f64> {
        &self.dimensions
    }

    pub fn leg(&self, l: u8) -> Result<&super::leg::Leg, super::Error> {
        if l > 3 {
            return Err(super::Error::LegNumberOutOfBounds(l));
        }

        Ok(&self.legs[l as usize])
    }

    pub fn leg_mut(&mut self, l: u8) -> Result<&mut super::leg::Leg, super::Error> {
        if l > 3 {
            return Err(super::Error::LegNumberOutOfBounds(l));
        }

        Ok(&mut self.legs[l as usize])
    }
}

pub struct TorsoBuilder {
    orientation: nalgebra::Vector3<f64>,
    position: nalgebra::Vector3<f64>,
    dimensions: nalgebra::Vector2<f64>,
    legs: [super::leg::Leg; 4],
}

impl TorsoBuilder {
    pub fn new() -> Self {
        Self {
            orientation: nalgebra::Vector3::<f64>::zeros(),
            position: nalgebra::Vector3::<f64>::zeros(),
            dimensions: nalgebra::Vector2::<f64>::new(50.0, 90.0),
            legs: [
                super::leg::Leg::builder(0).build(),
                super::leg::Leg::builder(1).build(),
                super::leg::Leg::builder(2).build(),
                super::leg::Leg::builder(3).build(),
            ],
        }
    }
   
    #[inline(always)]
    pub fn legs(mut self, legs: [Leg; 4]) -> Self {
        self.legs = legs;

        self
    }

    #[inline(always)]
    pub fn build(self) -> Torso {
        Torso::new(self.orientation, self.position, self.dimensions, self.legs)
    }
}
