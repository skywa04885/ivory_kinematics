use crate::{
    model::{leg::Leg, torso::Torso},
    Error,
};

#[derive(Debug, Clone)]
pub struct Solver {
    torso: Torso,
    legs: [Leg; 4],
    pseudo_inverse_epsilon: f64,
}

impl Solver {
    #[inline(always)]
    pub fn new(torso: Torso, pseudo_inverse_epsilon: f64, legs: [Leg; 4]) -> Self {
        Self {
            torso,
            pseudo_inverse_epsilon,
            legs,
        }
    }

    #[inline(always)]
    pub fn builder(torso: Torso, legs: [Leg; 4]) -> SolverBuilder {
        SolverBuilder::new(torso, legs)
    }

    #[inline(always)]
    pub fn torso(&self) -> &Torso {
        &self.torso
    }

    pub fn leg(&self, l: u8) -> Result<&Leg, Error> {
        if l > 3 {
            return Err(Error::LegNumberOutOfBounds(l));
        }

        Ok(&self.legs[l as usize])
    }

    pub fn leg_mut(&mut self, l: u8) -> Result<&mut Leg, Error> {
        if l > 3 {
            return Err(Error::LegNumberOutOfBounds(l));
        }

        Ok(&mut self.legs[l as usize])
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
        let leg: &Leg = self.leg(l)?;

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
        let leg: &Leg = self.leg(l)?;

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
        let leg: &Leg = self.leg(l)?;

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
        let leg: &Leg = self.leg(l)?;

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
        let leg: &Leg = self.leg(l)?;

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
        let leg: &Leg = self.leg(l)?;

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
        let leg: &Leg = self.leg(l)?;

        Self::fk_paw_ef_position(leg, torso)
    }

    fn ik_paw_ef_position_with_eps(
        torso: &Torso,
        leg: &Leg,
        target_position: &nalgebra::Vector3<f64>,
        epsilon: Option<f64>,
        pseudo_inverse_epsilon: f64,
    ) -> Result<nalgebra::Vector3<f64>, Error> {
        let mut leg = leg.clone();

        let mut error: f64 = 0.0;

        let mut current_position: nalgebra::Vector3<f64> = Self::fk_paw_ef_position(&leg, torso)?;
        let mut delta_position: nalgebra::Vector3<f64> = target_position - current_position;

        for _ in 1..30 {
            let delta_angles: nalgebra::Vector3<f64> =
                Self::ik_paw_ef_position(&leg, torso, &delta_position, pseudo_inverse_epsilon)?;

            *leg.thetas_mut() += delta_angles;

            if epsilon.is_none() {
                return Ok(leg.thetas().clone());
            }

            current_position = Self::fk_paw_ef_position(&leg, torso)?;
            delta_position = target_position - current_position;
            error = delta_position.magnitude();

            if error < epsilon.as_ref().unwrap().clone() {
                return Ok(leg.thetas().clone());
            }
        }

        Err(Error::UnreachableTargetPosition(
            error,
            target_position.clone(),
        ))
    }

    pub fn move_paw_absolute(
        &mut self,
        l: u8,
        target_position: &nalgebra::Vector3<f64>,
        epsilon: Option<f64>,
    ) -> Result<(), Error> {
        *self.leg_mut(l)?.thetas_mut() = Self::ik_paw_ef_position_with_eps(
            &self.torso,
            self.leg(l)?,
            target_position,
            epsilon,
            self.pseudo_inverse_epsilon,
        )?;

        Ok(())
    }

    pub fn move_paw_relative(
        &mut self,
        l: u8,
        relative_position: &nalgebra::Vector3<f64>,
        epsilon: Option<f64>,
    ) -> Result<(), Error> {
        let torso: &Torso = &self.torso;
        let leg: &Leg = self.leg(l)?;

        // Gets the current position and based from that computes the new absolute position for the
        // paw.
        let current_position: nalgebra::Vector3<f64> = Self::fk_paw_ef_position(leg, torso)?;
        let target_position: nalgebra::Vector3<f64> = current_position + relative_position;

        // Performs the absolute paw movement with the computed target position.
        self.move_paw_absolute(l, &target_position, epsilon)
    }

    /// Changes the orientation of the torso and performs kinematics on the paws so that they
    /// maintain their original positions.
    ///
    /// # Examples
    ///
    /// ```
    /// use ivory_kinematics::{Error, Solver, Leg, Torso};
    ///
    /// let epsilon: f64 = 0.001;
    ///
    /// let torso: Torso = Torso::builder()
    ///     .build();
    ///
    /// let legs: [Leg; 4] = [
    ///     Leg::builder(0).build(),
    ///     Leg::builder(1).build(),
    ///     Leg::builder(2).build(),
    ///     Leg::builder(3).build(),
    /// ];
    /// let mut solver = Solver::builder(torso, legs).build();
    ///
    /// let orig_paw_positions: [nalgebra::Vector3<f64>; 4] = [
    ///     solver.fk_paw_ef_position_for_leg(0).unwrap(),
    ///     solver.fk_paw_ef_position_for_leg(1).unwrap(),
    ///     solver.fk_paw_ef_position_for_leg(2).unwrap(),
    ///     solver.fk_paw_ef_position_for_leg(3).unwrap(),
    /// ];
    ///
    /// let absolute_orientation: nalgebra::Vector3<f64> = nalgebra::Vector3::<f64>::new(
    ///     f64::to_radians(10.0),
    ///     f64::to_radians(12.0),
    ///     f64::to_radians(10.0),
    /// );
    /// solver.orient_torso_relative(&absolute_orientation, Some(epsilon)).unwrap();
    ///
    /// assert_eq!(nalgebra::Vector3::<f64>::new(
    ///     f64::to_radians(10.0),
    ///     f64::to_radians(12.0),
    ///     f64::to_radians(10.0),
    /// ), solver.torso().orientation().clone());
    ///
    /// let final_paw_positions: [nalgebra::Vector3<f64>; 4] = [
    ///     solver.fk_paw_ef_position_for_leg(0).unwrap(),
    ///     solver.fk_paw_ef_position_for_leg(1).unwrap(),
    ///     solver.fk_paw_ef_position_for_leg(2).unwrap(),
    ///     solver.fk_paw_ef_position_for_leg(3).unwrap(),
    /// ];
    ///
    /// assert_eq!((orig_paw_positions[0] - final_paw_positions[0]).magnitude() < epsilon, true);
    /// assert_eq!((orig_paw_positions[1] - final_paw_positions[1]).magnitude() < epsilon, true);
    /// assert_eq!((orig_paw_positions[2] - final_paw_positions[2]).magnitude() < epsilon, true);
    /// assert_eq!((orig_paw_positions[3] - final_paw_positions[3]).magnitude() < epsilon, true);
    /// ```
    pub fn orient_torso_absolute(
        &mut self,
        absolute_orientation: &nalgebra::Vector3<f64>,
        epsilon: Option<f64>,
    ) -> Result<(), Error> {
        let torso: &mut Torso = &mut self.torso;
        let legs: &mut [Leg; 4] = &mut self.legs;

        // Clones the torso and the legs so that we can perfors inverse kinematics on them but still
        // keep the current state clean.
        let mut result_torso: Torso = torso.clone();
        let mut result_legs: [Leg; 4] = legs.clone();

        // Changes the orientation in the (temprary) result torso.
        *result_torso.orientation_mut() = absolute_orientation.clone();

        // Updates all of the four legs so that they don't lose their previous paw positions.
        for l in 0..4 {
            let leg: &Leg = &legs[l];
            let result_leg: &mut Leg = &mut result_legs[l];

            // Computes the previous paw end effector position so that we can perform inverse
            // kinematics to make it reach once more again.
            let prev_absolute_position: nalgebra::Vector3<f64> =
                Self::fk_paw_ef_position(leg, torso)?;

            // Performs inverse kinematics to make the paw end effector position reach the same
            // position as it did without the new orientation.
            *result_leg.thetas_mut() = Self::ik_paw_ef_position_with_eps(
                &result_torso,
                &result_leg,
                &prev_absolute_position,
                epsilon,
                self.pseudo_inverse_epsilon,
            )?;
        }

        // Puts the changed toros and legs back into the current solver, only because the desired
        // target was reachable.
        self.torso = result_torso;
        self.legs = result_legs;

        Ok(())
    }

    /// Changes the orientation of the torso relative to the current orientation and then performs
    /// inverse kinematics on the paws so that they maintain their original positions.
    ///
    /// # Examples
    ///
    /// ```
    /// use ivory_kinematics::{Error, Solver, Leg, Torso};
    ///
    /// let epsilon: f64 = 0.001;
    ///
    /// let torso: Torso = Torso::builder()
    ///     .build();
    ///
    /// let legs: [Leg; 4] = [
    ///     Leg::builder(0).build(),
    ///     Leg::builder(1).build(),
    ///     Leg::builder(2).build(),
    ///     Leg::builder(3).build(),
    /// ];
    /// let mut solver = Solver::builder(torso, legs).build();
    ///
    /// let orig_paw_positions: [nalgebra::Vector3<f64>; 4] = [
    ///     solver.fk_paw_ef_position_for_leg(0).unwrap(),
    ///     solver.fk_paw_ef_position_for_leg(1).unwrap(),
    ///     solver.fk_paw_ef_position_for_leg(2).unwrap(),
    ///     solver.fk_paw_ef_position_for_leg(3).unwrap(),
    /// ];
    ///
    /// let relative_orientation: nalgebra::Vector3<f64> = nalgebra::Vector3::<f64>::new(
    ///     f64::to_radians(10.0),
    ///     f64::to_radians(10.0),
    ///     f64::to_radians(10.0),
    /// );
    /// solver.orient_torso_relative(&relative_orientation, Some(epsilon)).unwrap();
    ///
    /// assert_eq!(nalgebra::Vector3::<f64>::new(
    ///     f64::to_radians(10.0),
    ///     f64::to_radians(10.0),
    ///     f64::to_radians(10.0),
    /// ), solver.torso().orientation().clone());
    ///
    /// let final_paw_positions: [nalgebra::Vector3<f64>; 4] = [
    ///     solver.fk_paw_ef_position_for_leg(0).unwrap(),
    ///     solver.fk_paw_ef_position_for_leg(1).unwrap(),
    ///     solver.fk_paw_ef_position_for_leg(2).unwrap(),
    ///     solver.fk_paw_ef_position_for_leg(3).unwrap(),
    /// ];
    ///
    /// assert_eq!((orig_paw_positions[0] - final_paw_positions[0]).magnitude() < epsilon, true);
    /// assert_eq!((orig_paw_positions[1] - final_paw_positions[1]).magnitude() < epsilon, true);
    /// assert_eq!((orig_paw_positions[2] - final_paw_positions[2]).magnitude() < epsilon, true);
    /// assert_eq!((orig_paw_positions[3] - final_paw_positions[3]).magnitude() < epsilon, true);
    /// ```
    pub fn orient_torso_relative(
        &mut self,
        relative_orientation: &nalgebra::Vector3<f64>,
        epsilon: Option<f64>,
    ) -> Result<(), Error> {
        // Computes the absolute orientation by adding the relative to the current orientation.
        let absolute_orientation: nalgebra::Vector3<f64> =
            self.torso.orientation() + relative_orientation;

        // Performs the absolute orientation change.
        self.orient_torso_absolute(&absolute_orientation, epsilon)
    }

    /// Moves the torso to the given absolute position and performs inverse kinematics on each paw
    /// so that they maintain their original positions.
    ///
    /// # Examples
    ///
    /// ```
    /// use ivory_kinematics::{Error, Solver, Leg, Torso};
    ///
    /// let epsilon: f64 = 0.001;
    ///
    /// let torso: Torso = Torso::builder()
    ///     .build();
    ///
    /// let legs: [Leg; 4] = [
    ///     Leg::builder(0).build(),
    ///     Leg::builder(1).build(),
    ///     Leg::builder(2).build(),
    ///     Leg::builder(3).build(),
    /// ];
    /// let mut solver = Solver::builder(torso, legs).build();
    ///
    /// let orig_paw_positions: [nalgebra::Vector3<f64>; 4] = [
    ///     solver.fk_paw_ef_position_for_leg(0).unwrap(),
    ///     solver.fk_paw_ef_position_for_leg(1).unwrap(),
    ///     solver.fk_paw_ef_position_for_leg(2).unwrap(),
    ///     solver.fk_paw_ef_position_for_leg(3).unwrap(),
    /// ];
    ///
    /// let absolute_position: nalgebra::Vector3<f64> = nalgebra::Vector3::<f64>::new(2.0, 2.0, 2.0);
    /// solver.move_torso_absolute(&absolute_position, Some(epsilon)).unwrap();
    ///
    /// assert_eq!(nalgebra::Vector3::<f64>::new(2.0, 2.0, 2.0), solver.torso().position().clone());
    ///
    /// let final_paw_positions: [nalgebra::Vector3<f64>; 4] = [
    ///     solver.fk_paw_ef_position_for_leg(0).unwrap(),
    ///     solver.fk_paw_ef_position_for_leg(1).unwrap(),
    ///     solver.fk_paw_ef_position_for_leg(2).unwrap(),
    ///     solver.fk_paw_ef_position_for_leg(3).unwrap(),
    /// ];
    ///
    /// assert_eq!((orig_paw_positions[0] - final_paw_positions[0]).magnitude() < epsilon, true);
    /// assert_eq!((orig_paw_positions[1] - final_paw_positions[1]).magnitude() < epsilon, true);
    /// assert_eq!((orig_paw_positions[2] - final_paw_positions[2]).magnitude() < epsilon, true);
    /// assert_eq!((orig_paw_positions[3] - final_paw_positions[3]).magnitude() < epsilon, true);
    /// ```
    pub fn move_torso_absolute(
        &mut self,
        absolute_position: &nalgebra::Vector3<f64>,
        epsilon: Option<f64>,
    ) -> Result<(), Error> {
        let torso: &mut Torso = &mut self.torso;
        let legs: &mut [Leg; 4] = &mut self.legs;

        // Clones the torso and the legs so that we can perfors inverse kinematics on them but still
        // keep the current state clean.
        let mut result_torso: Torso = torso.clone();
        let mut result_legs: [Leg; 4] = legs.clone();

        // Changes the orientation in the (temprary) result torso.
        *result_torso.position_mut() = absolute_position.clone();

        // Updates all of the four legs so that they don't lose their previous paw positions.
        for l in 0..4 {
            let leg: &Leg = &legs[l];
            let result_leg: &mut Leg = &mut result_legs[l];

            // Computes the previous paw end effector position so that we can perform inverse
            // kinematics to make it reach once more again.
            let prev_absolute_position: nalgebra::Vector3<f64> =
                Self::fk_paw_ef_position(leg, torso)?;

            // Performs inverse kinematics to make the paw end effector position reach the same
            // position as it did without the new orientation.
            *result_leg.thetas_mut() = Self::ik_paw_ef_position_with_eps(
                &result_torso,
                &result_leg,
                &prev_absolute_position,
                epsilon,
                self.pseudo_inverse_epsilon,
            )?;
        }

        // Puts the changed toros and legs back into the current solver, only because the desired
        // target was reachable.
        self.torso = result_torso;
        self.legs = result_legs;

        Ok(())
    }

    /// Moves the torso relative to the current torso position and performs inverse kinematics on
    /// all the paws to ensure they mantain their original end-effector positions.
    ///
    /// # Examples
    ///
    /// ```
    /// use ivory_kinematics::{Error, Solver, Leg, Torso};
    ///
    /// let epsilon: f64 = 0.001;
    ///
    /// let torso: Torso = Torso::builder()
    ///     .build();
    ///
    /// let legs: [Leg; 4] = [
    ///     Leg::builder(0).build(),
    ///     Leg::builder(1).build(),
    ///     Leg::builder(2).build(),
    ///     Leg::builder(3).build(),
    /// ];
    /// let mut solver = Solver::builder(torso, legs).build();
    ///
    /// let orig_paw_positions: [nalgebra::Vector3<f64>; 4] = [
    ///     solver.fk_paw_ef_position_for_leg(0).unwrap(),
    ///     solver.fk_paw_ef_position_for_leg(1).unwrap(),
    ///     solver.fk_paw_ef_position_for_leg(2).unwrap(),
    ///     solver.fk_paw_ef_position_for_leg(3).unwrap(),
    /// ];
    ///
    /// let relative_position: nalgebra::Vector3<f64> = nalgebra::Vector3::<f64>::new(0.1, 0.1, 0.1);
    /// solver.move_torso_relative(&relative_position, Some(epsilon)).unwrap();
    ///
    /// assert_eq!(nalgebra::Vector3::<f64>::new(0.1, 0.1, 0.1), solver.torso().position().clone());
    ///
    /// let final_paw_positions: [nalgebra::Vector3<f64>; 4] = [
    ///     solver.fk_paw_ef_position_for_leg(0).unwrap(),
    ///     solver.fk_paw_ef_position_for_leg(1).unwrap(),
    ///     solver.fk_paw_ef_position_for_leg(2).unwrap(),
    ///     solver.fk_paw_ef_position_for_leg(3).unwrap(),
    /// ];
    ///
    /// assert_eq!((orig_paw_positions[0] - final_paw_positions[0]).magnitude() < epsilon, true);
    /// assert_eq!((orig_paw_positions[1] - final_paw_positions[1]).magnitude() < epsilon, true);
    /// assert_eq!((orig_paw_positions[2] - final_paw_positions[2]).magnitude() < epsilon, true);
    /// assert_eq!((orig_paw_positions[3] - final_paw_positions[3]).magnitude() < epsilon, true);
    /// ```
    pub fn move_torso_relative(
        &mut self,
        relative_position: &nalgebra::Vector3<f64>,
        epsilon: Option<f64>,
    ) -> Result<(), Error> {
        // Computes the absolute position by adding the current position to it.
        let absolute_position: nalgebra::Vector3<f64> = self.torso.position() + relative_position;

        // Performs the absolute position change.
        self.move_torso_absolute(&absolute_position, epsilon)
    }
}

pub struct SolverBuilder {
    torso: Torso,
    legs: [Leg; 4],
    pseudo_inverse_epsilon: f64,
}

impl SolverBuilder {
    /// Creates a new solver builder.
    #[inline(always)]
    pub fn new(torso: Torso, legs: [Leg; 4]) -> Self {
        Self {
            torso,
            legs,
            pseudo_inverse_epsilon: 0.001,
        }
    }

    /// Sets the epsilon value for the pseudo inversion of the jacobian matrix.
    #[inline(always)]
    pub fn pseudo_inverse_epsilon(mut self, pseudo_inverse_epsilon: f64) -> Self {
        self.pseudo_inverse_epsilon = pseudo_inverse_epsilon;

        self
    }

    /// Builds the solver.
    #[inline(always)]
    pub fn build(self) -> Solver {
        Solver::new(self.torso, self.pseudo_inverse_epsilon, self.legs)
    }
}
