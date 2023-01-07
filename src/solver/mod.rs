use crate::model::{leg::Leg, torso::Torso, Error};

#[derive(Debug, Clone)]
pub struct Solver {
    torso: Torso,
    pseudo_inverse_epsilon: f64,
}

impl Solver {
    #[inline(always)]
    pub fn new(torso: Torso, pseudo_inverse_epsilon: f64) -> Self {
        Self {
            torso,
            pseudo_inverse_epsilon,
        }
    }

    #[inline(always)]
    pub fn builder(torso: Torso) -> SolverBuilder {
        SolverBuilder::new(torso)
    }

    #[inline(always)]
    pub fn torso(&self) -> &Torso {
        &self.torso
    }

    fn m_t(leg: &Leg, torso: &Torso) -> Result<nalgebra::Matrix4<f64>, Error> {
        let (s_x, s_z) = (leg.signs().x, leg.signs().y);

        let (alpha_t, beta_t, gamma_t) = (
            torso.orientation().x,
            torso.orientation().y,
            torso.orientation().z,
        );
        let (x_t, y_t, z_t) = (torso.position().x, torso.position().y, torso.position().z);
        let (w_t, h_t) = (torso.dimensions().x, torso.dimensions().y);

        Ok(nalgebra::Matrix4::<f64>::new(
            alpha_t.cos() * beta_t.cos(),
            -alpha_t.sin() * gamma_t.cos() + beta_t.sin() * gamma_t.sin() * alpha_t.cos(),
            alpha_t.sin() * gamma_t.sin() + beta_t.sin() * alpha_t.cos() * gamma_t.cos(),
            y_t * (-alpha_t.sin() * gamma_t.cos() + beta_t.sin() * gamma_t.sin() * alpha_t.cos())
                + ((1_f64 / 2.0) * h_t * s_x + x_t) * alpha_t.cos() * beta_t.cos()
                + ((1_f64 / 2.0) * s_z * w_t + z_t)
                    * (alpha_t.sin() * gamma_t.sin()
                        + beta_t.sin() * alpha_t.cos() * gamma_t.cos()),
            alpha_t.sin() * beta_t.cos(),
            alpha_t.sin() * beta_t.sin() * gamma_t.sin() + alpha_t.cos() * gamma_t.cos(),
            alpha_t.sin() * beta_t.sin() * gamma_t.cos() - gamma_t.sin() * alpha_t.cos(),
            y_t * (alpha_t.sin() * beta_t.sin() * gamma_t.sin() + alpha_t.cos() * gamma_t.cos())
                + ((1_f64 / 2.0) * h_t * s_x + x_t) * alpha_t.sin() * beta_t.cos()
                + ((1_f64 / 2.0) * s_z * w_t + z_t)
                    * (alpha_t.sin() * beta_t.sin() * gamma_t.cos()
                        - gamma_t.sin() * alpha_t.cos()),
            -beta_t.sin(),
            gamma_t.sin() * beta_t.cos(),
            beta_t.cos() * gamma_t.cos(),
            y_t * gamma_t.sin() * beta_t.cos() - ((1_f64 / 2.0) * h_t * s_x + x_t) * beta_t.sin()
                + ((1_f64 / 2.0) * s_z * w_t + z_t) * beta_t.cos() * gamma_t.cos(),
            0.0,
            0.0,
            0.0,
            1.0,
        ))
    }

    #[inline(always)]
    pub fn m_t_for_leg(&self, l: u8) -> Result<nalgebra::Matrix4<f64>, Error> {
        let torso: &Torso = &self.torso;
        let leg: &Leg = torso.leg(l)?;

        Self::m_t(leg, torso)
    }

    fn m_0(leg: &Leg) -> Result<nalgebra::Matrix4<f64>, Error> {
        let s_x = leg.signs().x;

        let l_0 = leg.lengths().x;
        let theta_0 = leg.thetas().x;

        Ok(nalgebra::Matrix4::<f64>::new(
            1.0,
            0.0,
            0.0,
            l_0 * s_x,
            0.0,
            theta_0.cos(),
            -theta_0.sin(),
            0.0,
            0.0,
            theta_0.sin(),
            theta_0.cos(),
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ))
    }

    #[inline(always)]
    pub fn m_0_for_leg(&self, l: u8) -> Result<nalgebra::Matrix4<f64>, Error> {
        let torso: &Torso = &self.torso;
        let leg: &Leg = torso.leg(l)?;

        Self::m_0(leg)
    }

    fn m_1(leg: &Leg) -> Result<nalgebra::Matrix4<f64>, Error> {
        let l_1 = leg.lengths().y;
        let theta_1 = leg.thetas().y;

        Ok(nalgebra::Matrix4::<f64>::new(
            theta_1.cos(),
            -theta_1.sin(),
            0.0,
            l_1 * theta_1.sin(),
            theta_1.sin(),
            theta_1.cos(),
            0.0,
            -l_1 * theta_1.cos(),
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ))
    }

    #[inline(always)]
    pub fn m_1_for_leg(&self, l: u8) -> Result<nalgebra::Matrix4<f64>, Error> {
        let torso: &Torso = &self.torso;
        let leg: &Leg = torso.leg(l)?;

        Self::m_1(leg)
    }

    fn m_2(leg: &Leg) -> Result<nalgebra::Matrix4<f64>, Error> {
        let l_2 = leg.lengths().z;
        let theta_2 = leg.thetas().z;

        Ok(nalgebra::Matrix4::<f64>::new(
            theta_2.cos(),
            -theta_2.sin(),
            0.0,
            l_2 * theta_2.sin(),
            theta_2.sin(),
            theta_2.cos(),
            0.0,
            -l_2 * theta_2.cos(),
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ))
    }

    #[inline(always)]
    pub fn m_2_for_leg(&self, l: u8) -> Result<nalgebra::Matrix4<f64>, Error> {
        let torso: &Torso = &self.torso;
        let leg: &Leg = torso.leg(l)?;

        Self::m_2(leg)
    }

    fn fk_vertices(leg: &Leg, torso: &Torso) -> Result<[nalgebra::Vector3<f64>; 5], Error> {
        let (m_t, m_0, m_1, m_2) = (
            Self::m_t(leg, torso)?,
            Self::m_0(leg)?,
            Self::m_1(leg)?,
            Self::m_2(leg)?,
        );

        let vertex_0: nalgebra::Vector3<f64> = torso.position().clone();

        let mut m: nalgebra::Matrix4<f64> = m_t;
        let vertex_1: nalgebra::Vector3<f64> = m.fixed_slice::<3, 1>(0, 3).into();
        m *= m_0;
        let vertex_2: nalgebra::Vector3<f64> = m.fixed_slice::<3, 1>(0, 3).into();
        m *= m_1;
        let vertex_3: nalgebra::Vector3<f64> = m.fixed_slice::<3, 1>(0, 3).into();
        m *= m_2;
        let vertex_4: nalgebra::Vector3<f64> = m.fixed_slice::<3, 1>(0, 3).into();

        Ok([vertex_0, vertex_1, vertex_2, vertex_3, vertex_4])
    }

    #[inline(always)]
    pub fn fk_vertices_for_leg(&self, l: u8) -> Result<[nalgebra::Vector3<f64>; 5], Error> {
        let torso: &Torso = &self.torso;
        let leg: &Leg = torso.leg(l)?;

        Self::fk_vertices(leg, torso)
    }

    #[allow(unused_variables)]
    fn ik_paw_ef_position(
        leg: &Leg,
        torso: &Torso,
        delta_position: &nalgebra::Vector3<f64>,
        pseudo_inverse_epsilon: f64,
    ) -> Result<nalgebra::Vector3<f64>, Error> {
        let (s_x, s_z) = (leg.signs().x, leg.signs().y);

        let (alpha_t, beta_t, gamma_t) = (
            torso.orientation().x,
            torso.orientation().y,
            torso.orientation().z,
        );
        let (x_t, y_t, z_t) = (torso.position().x, torso.position().y, torso.position().z);
        let (w_t, h_t) = (torso.dimensions().x, torso.dimensions().y);

        let (l_0, l_1, l_2) = (leg.lengths().x, leg.lengths().y, leg.lengths().z);
        let (theta_0, theta_1, theta_2) = (leg.thetas().x, leg.thetas().y, leg.thetas().z);

        let jacobian = nalgebra::Matrix3::<f64>::new(
            -l_1 * alpha_t.sin() * (gamma_t + theta_0).sin() * theta_1.cos()
                - l_1 * beta_t.sin() * alpha_t.cos() * theta_1.cos() * (gamma_t + theta_0).cos()
                - l_2 * alpha_t.sin() * (gamma_t + theta_0).sin() * (theta_1 + theta_2).cos()
                - l_2
                    * beta_t.sin()
                    * alpha_t.cos()
                    * (gamma_t + theta_0).cos()
                    * (theta_1 + theta_2).cos(),
            -l_1 * alpha_t.sin() * theta_1.sin() * (gamma_t + theta_0).cos()
                + l_1 * beta_t.sin() * theta_1.sin() * (gamma_t + theta_0).sin() * alpha_t.cos()
                + l_1 * alpha_t.cos() * beta_t.cos() * theta_1.cos()
                - l_2 * alpha_t.sin() * (theta_1 + theta_2).sin() * (gamma_t + theta_0).cos()
                + l_2
                    * beta_t.sin()
                    * (gamma_t + theta_0).sin()
                    * (theta_1 + theta_2).sin()
                    * alpha_t.cos()
                + l_2 * alpha_t.cos() * beta_t.cos() * (theta_1 + theta_2).cos(),
            l_2 * (-alpha_t.sin() * (theta_1 + theta_2).sin() * (gamma_t + theta_0).cos()
                + beta_t.sin()
                    * (gamma_t + theta_0).sin()
                    * (theta_1 + theta_2).sin()
                    * alpha_t.cos()
                + alpha_t.cos() * beta_t.cos() * (theta_1 + theta_2).cos()),
            -l_1 * alpha_t.sin() * beta_t.sin() * theta_1.cos() * (gamma_t + theta_0).cos()
                + l_1 * (gamma_t + theta_0).sin() * alpha_t.cos() * theta_1.cos()
                - l_2
                    * alpha_t.sin()
                    * beta_t.sin()
                    * (gamma_t + theta_0).cos()
                    * (theta_1 + theta_2).cos()
                + l_2 * (gamma_t + theta_0).sin() * alpha_t.cos() * (theta_1 + theta_2).cos(),
            l_1 * alpha_t.sin() * beta_t.sin() * theta_1.sin() * (gamma_t + theta_0).sin()
                + l_1 * alpha_t.sin() * beta_t.cos() * theta_1.cos()
                + l_1 * theta_1.sin() * alpha_t.cos() * (gamma_t + theta_0).cos()
                + l_2
                    * alpha_t.sin()
                    * beta_t.sin()
                    * (gamma_t + theta_0).sin()
                    * (theta_1 + theta_2).sin()
                + l_2 * alpha_t.sin() * beta_t.cos() * (theta_1 + theta_2).cos()
                + l_2 * (theta_1 + theta_2).sin() * alpha_t.cos() * (gamma_t + theta_0).cos(),
            l_2 * (alpha_t.sin()
                * beta_t.sin()
                * (gamma_t + theta_0).sin()
                * (theta_1 + theta_2).sin()
                + alpha_t.sin() * beta_t.cos() * (theta_1 + theta_2).cos()
                + (theta_1 + theta_2).sin() * alpha_t.cos() * (gamma_t + theta_0).cos()),
            -(l_1 * theta_1.cos() + l_2 * (theta_1 + theta_2).cos())
                * beta_t.cos()
                * (gamma_t + theta_0).cos(),
            -l_1 * beta_t.sin() * theta_1.cos()
                + l_1 * theta_1.sin() * (gamma_t + theta_0).sin() * beta_t.cos()
                - l_2 * beta_t.sin() * (theta_1 + theta_2).cos()
                + l_2 * (gamma_t + theta_0).sin() * (theta_1 + theta_2).sin() * beta_t.cos(),
            l_2 * (-beta_t.sin() * (theta_1 + theta_2).cos()
                + (gamma_t + theta_0).sin() * (theta_1 + theta_2).sin() * beta_t.cos()),
        );

        let jacobian_inverse = match jacobian.pseudo_inverse(pseudo_inverse_epsilon) {
            Ok(mat) => mat,
            Err(error) => return Err(Error::PseudoInverse(error)),
        };

        let delta_angles = jacobian_inverse * delta_position;

        Ok(delta_angles)
    }

    #[inline(always)]
    pub fn ik_paw_ef_position_for_leg(
        &self,
        l: u8,
        delta_position: &nalgebra::Vector3<f64>,
        pseudo_inverse_epsilon: f64,
    ) -> Result<nalgebra::Vector3<f64>, Error> {
        let torso: &Torso = &self.torso;
        let leg: &Leg = torso.leg(l)?;

        Self::ik_paw_ef_position(leg, torso, delta_position, pseudo_inverse_epsilon)
    }

    #[allow(unused_variables)]
    fn fk_paw_ef_position(leg: &Leg, torso: &Torso) -> Result<nalgebra::Vector3<f64>, Error> {
        let (s_x, s_z) = (leg.signs().x, leg.signs().y);

        let (alpha_t, beta_t, gamma_t) = (
            torso.orientation().x,
            torso.orientation().y,
            torso.orientation().z,
        );
        let (x_t, y_t, z_t) = (torso.position().x, torso.position().y, torso.position().z);
        let (w_t, h_t) = (torso.dimensions().x, torso.dimensions().y);

        let (l_0, l_1, l_2) = (leg.lengths().x, leg.lengths().y, leg.lengths().z);
        let (theta_0, theta_1, theta_2) = (leg.thetas().x, leg.thetas().y, leg.thetas().z);

        Ok(nalgebra::Vector3::<f64>::new(
            (1_f64 / 2.0) * h_t * s_x * alpha_t.cos() * beta_t.cos()
                + l_0 * s_x * alpha_t.cos() * beta_t.cos()
                + l_1 * alpha_t.sin() * theta_1.cos() * (gamma_t + theta_0).cos()
                - l_1 * beta_t.sin() * (gamma_t + theta_0).sin() * alpha_t.cos() * theta_1.cos()
                + l_1 * theta_1.sin() * alpha_t.cos() * beta_t.cos()
                + l_2 * alpha_t.sin() * (gamma_t + theta_0).cos() * (theta_1 + theta_2).cos()
                - l_2
                    * beta_t.sin()
                    * (gamma_t + theta_0).sin()
                    * alpha_t.cos()
                    * (theta_1 + theta_2).cos()
                + l_2 * (theta_1 + theta_2).sin() * alpha_t.cos() * beta_t.cos()
                + (1_f64 / 2.0) * s_z * w_t * alpha_t.sin() * gamma_t.sin()
                + (1_f64 / 2.0) * s_z * w_t * beta_t.sin() * alpha_t.cos() * gamma_t.cos()
                + x_t * alpha_t.cos() * beta_t.cos()
                - y_t * alpha_t.sin() * gamma_t.cos()
                + y_t * beta_t.sin() * gamma_t.sin() * alpha_t.cos()
                + z_t * alpha_t.sin() * gamma_t.sin()
                + z_t * beta_t.sin() * alpha_t.cos() * gamma_t.cos(),
            (1_f64 / 2.0) * h_t * s_x * alpha_t.sin() * beta_t.cos()
                + l_0 * s_x * alpha_t.sin() * beta_t.cos()
                - l_1 * alpha_t.sin() * beta_t.sin() * (gamma_t + theta_0).sin() * theta_1.cos()
                + l_1 * alpha_t.sin() * theta_1.sin() * beta_t.cos()
                - l_1 * alpha_t.cos() * theta_1.cos() * (gamma_t + theta_0).cos()
                - l_2
                    * alpha_t.sin()
                    * beta_t.sin()
                    * (gamma_t + theta_0).sin()
                    * (theta_1 + theta_2).cos()
                + l_2 * alpha_t.sin() * (theta_1 + theta_2).sin() * beta_t.cos()
                - l_2 * alpha_t.cos() * (gamma_t + theta_0).cos() * (theta_1 + theta_2).cos()
                + (1_f64 / 2.0) * s_z * w_t * alpha_t.sin() * beta_t.sin() * gamma_t.cos()
                - 1_f64 / 2.0 * s_z * w_t * gamma_t.sin() * alpha_t.cos()
                + x_t * alpha_t.sin() * beta_t.cos()
                + y_t * alpha_t.sin() * beta_t.sin() * gamma_t.sin()
                + y_t * alpha_t.cos() * gamma_t.cos()
                + z_t * alpha_t.sin() * beta_t.sin() * gamma_t.cos()
                - z_t * gamma_t.sin() * alpha_t.cos(),
            -1_f64 / 2.0 * h_t * s_x * beta_t.sin()
                - l_0 * s_x * beta_t.sin()
                - l_1 * beta_t.sin() * theta_1.sin()
                - l_1 * (gamma_t + theta_0).sin() * beta_t.cos() * theta_1.cos()
                - l_2 * beta_t.sin() * (theta_1 + theta_2).sin()
                - l_2 * (gamma_t + theta_0).sin() * beta_t.cos() * (theta_1 + theta_2).cos()
                + (1_f64 / 2.0) * s_z * w_t * beta_t.cos() * gamma_t.cos()
                - x_t * beta_t.sin()
                + y_t * gamma_t.sin() * beta_t.cos()
                + z_t * beta_t.cos() * gamma_t.cos(),
        ))
    }

    #[inline(always)]
    pub fn fk_paw_ef_position_for_leg(&self, l: u8) -> Result<nalgebra::Vector3<f64>, Error> {
        let torso: &Torso = &self.torso;
        let leg: &Leg = torso.leg(l)?;

        Self::fk_paw_ef_position(leg, torso)
    }

    pub fn move_paw_absolute(
        &mut self,
        l: u8,
        target_position: &nalgebra::Vector3<f64>,
        epsilon: Option<f64>,
    ) -> Result<(), Error> {
        let torso: &mut Torso = &mut self.torso;
        let mut leg = torso.leg(l)?.clone();

        let mut error: f64 = 0.0;

        let mut current_position: nalgebra::Vector3<f64> = Self::fk_paw_ef_position(&leg, torso)?;
        let mut delta_position: nalgebra::Vector3<f64> = target_position - current_position;

        for _ in 1..30 {
            let delta_angles: nalgebra::Vector3<f64> = Self::ik_paw_ef_position(
                &leg,
                torso,
                &delta_position,
                self.pseudo_inverse_epsilon,
            )?;

            *leg.thetas_mut() += delta_angles;

            if epsilon.is_none() {
                *torso.leg_mut(l)? = leg;
                return Ok(());
            }

            current_position = Self::fk_paw_ef_position(&leg, torso)?;
            delta_position = target_position - current_position;
            error = delta_position.magnitude();

            if error < epsilon.as_ref().unwrap().clone() {
                *torso.leg_mut(l)? = leg;
                return Ok(());
            }
        }

        Err(Error::UnreachableTargetPosition(error, target_position.clone()))
    }

    pub fn move_paw_relative(
        &mut self,
        l: u8,
        relative_position: &nalgebra::Vector3<f64>,
        epsilon: Option<f64>,
    ) -> Result<(), Error> {
        let torso: &Torso = &self.torso;
        let leg: &Leg = self.torso.leg(l)?;

        let current_position: nalgebra::Vector3<f64> = Self::fk_paw_ef_position(leg, torso)?;
        let target_position: nalgebra::Vector3<f64> = current_position + relative_position;

        self.move_paw_absolute(l, &target_position, epsilon)
    }
}

pub struct SolverBuilder {
    torso: Torso,
    pseudo_inverse_epsilon: f64,
}

impl SolverBuilder {
    #[inline(always)]
    pub fn new(torso: Torso) -> Self {
        Self {
            torso,
            pseudo_inverse_epsilon: 0.001,
        }
    }

    #[inline(always)]
    pub fn pseudo_inverse_epsilon(mut self, pseudo_inverse_epsilon: f64) -> Self {
        self.pseudo_inverse_epsilon = pseudo_inverse_epsilon;

        self
    }

    #[inline(always)]
    pub fn build(self) -> Solver {
        Solver::new(self.torso, self.pseudo_inverse_epsilon)
    }
}
