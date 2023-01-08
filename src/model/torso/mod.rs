#[derive(Debug, Clone)]
pub struct Torso {
    orientation: nalgebra::Vector3<f64>,
    position: nalgebra::Vector3<f64>,
    dimensions: nalgebra::Vector2<f64>,
}

impl Torso {
    #[inline(always)]
    pub fn new(
        orientation: nalgebra::Vector3<f64>,
        position: nalgebra::Vector3<f64>,
        dimensions: nalgebra::Vector2<f64>,
    ) -> Self {
        Self {
            orientation,
            position,
            dimensions,
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
}

pub struct TorsoBuilder {
    orientation: nalgebra::Vector3<f64>,
    position: nalgebra::Vector3<f64>,
    dimensions: nalgebra::Vector2<f64>,
}

impl TorsoBuilder {
    pub fn new() -> Self {
        Self {
            orientation: nalgebra::Vector3::<f64>::zeros(),
            position: nalgebra::Vector3::<f64>::zeros(),
            dimensions: nalgebra::Vector2::<f64>::new(50.0, 90.0),
        }
    }

    #[inline(always)]
    pub fn orientation(mut self, orientation: nalgebra::Vector3<f64>) -> Self {
        self.orientation = orientation;

        self
    }

    #[inline(always)]
    pub fn position(mut self, position: nalgebra::Vector3<f64>) -> Self {
        self.position = position;
        
        self
    }

    #[inline(always)]
    pub fn dimensions(mut self, dimensions: nalgebra::Vector2<f64>) -> Self {
        self.dimensions = dimensions;

        self
    }

    #[inline(always)]
    pub fn build(self) -> Torso {
        Torso::new(self.orientation, self.position, self.dimensions)
    }
}
