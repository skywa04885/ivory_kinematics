#[derive(Debug, Clone)]
pub struct Leg {
    signs: nalgebra::Vector2<f64>,
    thetas: nalgebra::Vector3<f64>,
    lengths: nalgebra::Vector3<f64>,
}

impl Leg {
    #[inline(always)]
    pub fn new(
        signs: nalgebra::Vector2<f64>,
        thetas: nalgebra::Vector3<f64>,
        lengths: nalgebra::Vector3<f64>,
    ) -> Self {
        Self {
            signs,
            thetas,
            lengths,
        }
    }

    #[inline(always)]
    pub fn thetas(&self) -> &nalgebra::Vector3<f64> {
        &self.thetas
    }

    #[inline(always)]
    pub fn thetas_mut(&mut self) -> &mut nalgebra::Vector3<f64> {
        &mut self.thetas
    }

    #[inline(always)]
    pub fn lengths(&self) -> &nalgebra::Vector3<f64> {
        &self.lengths
    }

    #[inline(always)]
    pub fn signs(&self) -> &nalgebra::Vector2<f64> {
        &self.signs
    }

    #[inline(always)]
    pub fn builder(leg_no: u8) -> LegBuilder {
        LegBuilder::new(leg_no)
    }
}

pub struct LegBuilder {
    signs: nalgebra::Vector2<f64>,
    thetas: nalgebra::Vector3<f64>,
    lengths: nalgebra::Vector3<f64>,
}

impl LegBuilder {
    #[inline(always)]
    pub fn new(leg_no: u8) -> Self {
        Self {
            signs: match leg_no {
                0 => nalgebra::Vector2::<f64>::new(-1.0, -1.0),
                1 => nalgebra::Vector2::<f64>::new(-1.0, 1.0),
                2 => nalgebra::Vector2::<f64>::new(1.0, -1.0),
                3 => nalgebra::Vector2::<f64>::new(1.0, 1.0),
                _ => panic!("Invalid leg number"),
            },
            thetas: nalgebra::Vector3::<f64>::new(0.0, f64::to_radians(-45.0), f64::to_radians(90.0)),
            lengths: nalgebra::Vector3::<f64>::new(0.0, 25.0, 25.0),
        }
    }

    #[inline(always)]
    pub fn thetas(mut self, thetas: nalgebra::Vector3<f64>) -> Self {
        self.thetas = thetas;

        self
    }

    #[inline(always)]
    pub fn lengths(mut self, lengths: nalgebra::Vector3<f64>) -> Self {
        self.lengths = lengths;

        self
    }

    #[inline(always)]
    pub fn signs(mut self, signs: nalgebra::Vector2<f64>) -> Self {
        self.signs = signs;

        self
    }

    #[inline(always)]
    pub fn build(self) -> Leg {
        Leg::new(self.signs, self.thetas, self.lengths)
    }
}
