#[derive(Debug, Clone)]
pub struct Arm {
    thetas: nalgebra::Vector5<f64>,
    lengths: nalgebra::Vector6<f64>,
    offsets: nalgebra::Vector1<f64>,
}

impl Arm {
    #[inline(always)]
    pub fn new(
        thetas: nalgebra::Vector5<f64>,
        lengths: nalgebra::Vector6<f64>,
        offsets: nalgebra::Vector1<f64>,
    ) -> Self {
        Self {
            thetas,
            lengths,
            offsets,
        }
    }

    #[inline(always)]
    pub fn thetas(&self) -> &nalgebra::Vector5<f64> {
        &self.thetas
    }

    #[inline(always)]
    pub fn thetas_mut(&mut self) -> &mut nalgebra::Vector5<f64> {
        &mut self.thetas
    }

    #[inline(always)]
    pub fn lengths(&self) -> &nalgebra::Vector6<f64> {
        &self.lengths
    }

    #[inline(always)]
    pub fn lengths_mut(&mut self) -> &mut nalgebra::Vector6<f64> {
        &mut self.lengths
    }

    #[inline(always)]
    pub fn offsets(&self) -> &nalgebra::Vector1<f64> {
        &self.offsets
    }

    #[inline(always)]
    pub fn offsets_mut(&mut self) -> &mut nalgebra::Vector1<f64> {
        &mut self.offsets
    }

    #[inline(always)]
    pub fn builder() -> ArmBuilder {
        ArmBuilder::new()
    }
}

pub struct ArmBuilder {
    thetas: nalgebra::Vector5<f64>,
    lengths: nalgebra::Vector6<f64>,
    offsets: nalgebra::Vector1<f64>,
}

impl ArmBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            thetas: nalgebra::Vector5::<f64>::new(
                f64::to_radians(0.0),
                f64::to_radians(0.0),
                f64::to_radians(0.0),
                f64::to_radians(0.0),
                f64::to_radians(0.0),
            ),
            lengths: nalgebra::Vector6::<f64>::new(10.0, 4.0, 20.0, 20.0, 0.0, 6.0),
            offsets: nalgebra::Vector1::<f64>::new(1.0),
        }
    }

    #[inline(always)]
    pub fn thetas(mut self, thetas: nalgebra::Vector5<f64>) -> Self {
        self.thetas = thetas;

        self
    }

    #[inline(always)]
    pub fn lengths(mut self, lengths: nalgebra::Vector6<f64>) -> Self {
        self.lengths = lengths;

        self
    }

    #[inline(always)]
    pub fn offsets(mut self, offsets: nalgebra::Vector1<f64>) -> Self {
        self.offsets = offsets;

        self
    }

    #[inline(always)]
    pub fn build(self) -> Arm {
        Arm::new(self.thetas, self.lengths, self.offsets)
    }
}
