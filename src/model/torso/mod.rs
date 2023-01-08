#[derive(Debug, Clone)]
pub struct Torso {
    orientation: nalgebra::Vector3<f64>,
    position: nalgebra::Vector3<f64>,
    dimensions: nalgebra::Vector2<f64>,
}

impl Torso {
    /// Constructs a new torso.
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

    /// Constructs a new torso builder.
    #[inline(always)]
    pub fn builder() -> TorsoBuilder {
        TorsoBuilder::new()
    }

    /// Gets a mutable reference to the orientation of the torso.
    #[inline(always)]
    pub fn orientation_mut(&mut self) -> &mut nalgebra::Vector3<f64> {
        &mut self.orientation
    }

    /// Gets a reference to the orientation of the torso.
    #[inline(always)]
    pub fn orientation(&self) -> &nalgebra::Vector3<f64> {
        &self.orientation
    }

    /// Gets a mutable reference to the position of the torso.
    #[inline(always)]
    pub fn position_mut(&mut self) -> &mut nalgebra::Vector3<f64> {
        &mut self.position
    }

    /// Gets a reference to the position of the torso.
    #[inline(always)]
    pub fn position(&self) -> &nalgebra::Vector3<f64> {
        &self.position
    }

    /// Gets a mutable reference to the dimensions of the torso.
    #[inline(always)]
    pub fn dimensions_mut(&mut self) -> &mut nalgebra::Vector2<f64> {
        &mut self.dimensions
    }

    /// Gets a reference to the dimensions of the torso.
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
    /// Constructs a new torso builder.
    pub fn new() -> Self {
        Self {
            orientation: nalgebra::Vector3::<f64>::zeros(),
            position: nalgebra::Vector3::<f64>::zeros(),
            dimensions: nalgebra::Vector2::<f64>::new(50.0, 90.0),
        }
    }

    /// Sets the orientation for the torso to be built.
    #[inline(always)]
    pub fn orientation(mut self, orientation: nalgebra::Vector3<f64>) -> Self {
        self.orientation = orientation;

        self
    }

    /// Sets the position for the torso to be built.
    #[inline(always)]
    pub fn position(mut self, position: nalgebra::Vector3<f64>) -> Self {
        self.position = position;

        self
    }

    /// Sets the dimensions for the torso to be built.
    #[inline(always)]
    pub fn dimensions(mut self, dimensions: nalgebra::Vector2<f64>) -> Self {
        self.dimensions = dimensions;

        self
    }

    /// Builds a torso with the current parameters.
    #[inline(always)]
    pub fn build(self) -> Torso {
        Torso::new(self.orientation, self.position, self.dimensions)
    }
}
