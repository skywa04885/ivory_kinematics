use crate::{
    model::{arm::Arm, leg::Leg, torso::Torso},
    Error,
};

#[derive(Debug, Clone)]
pub struct Solver {
    torso: Torso,
    legs: [Leg; 4],
    arm: Arm,
    pseudo_inverse_epsilon: f64,
}

impl Solver {
    /// Constructs a new solver.
    #[inline(always)]
    pub fn new(torso: Torso, pseudo_inverse_epsilon: f64, legs: [Leg; 4], arm: Arm) -> Self {
        Self {
            torso,
            legs,
            arm,
            pseudo_inverse_epsilon,
        }
    }

    /// Constructs a new builder for a solver.
    #[inline(always)]
    pub fn builder(torso: Torso, legs: [Leg; 4], arm: Arm) -> SolverBuilder {
        SolverBuilder::new(torso, legs, arm)
    }

    /// Gets a reference to the torso.
    #[inline(always)]
    pub fn torso(&self) -> &Torso {
        &self.torso
    }

    /// Gets a reference to the leg with the given index.
    pub fn leg(&self, l: u8) -> Result<&Leg, Error> {
        if l > 3 {
            return Err(Error::LegNumberOutOfBounds(l));
        }

        Ok(&self.legs[l as usize])
    }

    /// Gets a mutable reference to the leg with the given index.
    pub fn leg_mut(&mut self, l: u8) -> Result<&mut Leg, Error> {
        if l > 3 {
            return Err(Error::LegNumberOutOfBounds(l));
        }

        Ok(&mut self.legs[l as usize])
    }

    /// Computes the Mt forward kinematics matrix for the given leg and torso.
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

    /// Computes the Mt forward kinematics matrix for the given leg.
    #[inline(always)]
    pub fn m_t_for_leg(&self, l: u8) -> Result<nalgebra::Matrix4<f64>, Error> {
        let torso: &Torso = &self.torso;
        let leg: &Leg = self.leg(l)?;

        Self::m_t(leg, torso)
    }

    /// Computes the M0 forward kinematics matrix for the given leg.
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

    /// Computes the M0 forward kinematics matrix for the given leg.
    #[inline(always)]
    pub fn m_0_for_leg(&self, l: u8) -> Result<nalgebra::Matrix4<f64>, Error> {
        let leg: &Leg = self.leg(l)?;

        Self::m_0(leg)
    }

    /// Computes the M1 forward kinematics matrix for the given leg.
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

    /// Computes the M1 forward kinematics matrix for the given leg.
    #[inline(always)]
    pub fn m_1_for_leg(&self, l: u8) -> Result<nalgebra::Matrix4<f64>, Error> {
        let leg: &Leg = self.leg(l)?;

        Self::m_1(leg)
    }

    /// Computes the M2 forward kinematics matrix for the given leg.
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

    /// Computes the M2 foward kinematics matrix for the given leg.
    #[inline(always)]
    pub fn m_2_for_leg(&self, l: u8) -> Result<nalgebra::Matrix4<f64>, Error> {
        let leg: &Leg = self.leg(l)?;

        Self::m_2(leg)
    }

    /// Computes the vertices for the given leg on the given torso.
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

    /// Computes the vertices for the given leg using forward kinematics.
    #[inline(always)]
    pub fn fk_vertices_for_leg(&self, l: u8) -> Result<[nalgebra::Vector3<f64>; 5], Error> {
        let torso: &Torso = &self.torso;
        let leg: &Leg = self.leg(l)?;

        Self::fk_vertices(leg, torso)
    }

    /// Performs a single inverse kinematics step for the given leg, torso and change in the end
    /// effector position.
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

    /// Computes the end effector (paw) position of the given leg on the given torso.
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
            (1_f64 / 2_f64) * h_t * s_x * alpha_t.cos() * beta_t.cos()
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
                + (1_f64 / 2_f64) * s_z * w_t * alpha_t.sin() * gamma_t.sin()
                + (1_f64 / 2_f64) * s_z * w_t * beta_t.sin() * alpha_t.cos() * gamma_t.cos()
                + x_t * alpha_t.cos() * beta_t.cos()
                - y_t * alpha_t.sin() * gamma_t.cos()
                + y_t * beta_t.sin() * gamma_t.sin() * alpha_t.cos()
                + z_t * alpha_t.sin() * gamma_t.sin()
                + z_t * beta_t.sin() * alpha_t.cos() * gamma_t.cos(),
            (1_f64 / 2_f64) * h_t * s_x * alpha_t.sin() * beta_t.cos()
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
                + (1_f64 / 2_f64) * s_z * w_t * alpha_t.sin() * beta_t.sin() * gamma_t.cos()
                - 1_f64 / 2_f64 * s_z * w_t * gamma_t.sin() * alpha_t.cos()
                + x_t * alpha_t.sin() * beta_t.cos()
                + y_t * alpha_t.sin() * beta_t.sin() * gamma_t.sin()
                + y_t * alpha_t.cos() * gamma_t.cos()
                + z_t * alpha_t.sin() * beta_t.sin() * gamma_t.cos()
                - z_t * gamma_t.sin() * alpha_t.cos(),
            -1_f64 / 2_f64 * h_t * s_x * beta_t.sin()
                - l_0 * s_x * beta_t.sin()
                - l_1 * beta_t.sin() * theta_1.sin()
                - l_1 * (gamma_t + theta_0).sin() * beta_t.cos() * theta_1.cos()
                - l_2 * beta_t.sin() * (theta_1 + theta_2).sin()
                - l_2 * (gamma_t + theta_0).sin() * beta_t.cos() * (theta_1 + theta_2).cos()
                + (1_f64 / 2_f64) * s_z * w_t * beta_t.cos() * gamma_t.cos()
                - x_t * beta_t.sin()
                + y_t * gamma_t.sin() * beta_t.cos()
                + z_t * beta_t.cos() * gamma_t.cos(),
        ))
    }

    /// Computes the end effector (paw) position for the given leg.
    ///
    /// # Example
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
    /// let end_effector_pos: nalgebra::Vector3<f64> =
    ///     solver.fk_paw_ef_position_for_leg(0).unwrap();
    ///
    /// assert_eq!((nalgebra::Vector3::<f64>::new(-45.0, -35.355339, -25.0)
    ///     - end_effector_pos).magnitude() < epsilon, true);
    /// ```
    #[inline(always)]
    pub fn fk_paw_ef_position_for_leg(&self, l: u8) -> Result<nalgebra::Vector3<f64>, Error> {
        let torso: &Torso = &self.torso;
        let leg: &Leg = self.leg(l)?;

        Self::fk_paw_ef_position(leg, torso)
    }

    /// Moves the end effector of a paw to the given target position with the maximum error of
    /// epsilon.
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

    /// Moves the paw owned by the given leg to the given relative position.
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
    /// let original_position: nalgebra::Vector3<f64> =
    ///     solver.fk_paw_ef_position_for_leg(0).unwrap();
    /// let absolute_position: nalgebra::Vector3<f64> = original_position + nalgebra::Vector3::<f64>::new(
    ///     0.1,
    ///     0.1,
    ///     0.1,
    /// );
    /// solver.move_paw_absolute(0, &absolute_position, Some(epsilon)).unwrap();
    ///
    /// assert_eq!((absolute_position - solver.fk_paw_ef_position_for_leg(0)
    ///     .unwrap()).magnitude() < epsilon, true);
    /// ```
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

    /// Moves the paw owned by the given leg to the given relative position.
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
    /// let original_position: nalgebra::Vector3<f64> =
    ///     solver.fk_paw_ef_position_for_leg(0).unwrap();
    /// let relative_position: nalgebra::Vector3<f64> = nalgebra::Vector3::<f64>::new(
    ///     0.1,
    ///     0.1,
    ///     0.1,
    /// );
    /// solver.move_paw_relative(0, &relative_position, Some(epsilon)).unwrap();
    ///
    /// assert_eq!(((original_position + relative_position) -
    ///     solver.fk_paw_ef_position_for_leg(0).unwrap()).magnitude() < epsilon, true);
    /// ```
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

        let ik_arm_end_effector_target_position: nalgebra::Vector3<f64> =
            Self::s_fk_arm_ef_position(
                torso.position(),
                torso.orientation(),
                self.arm.thetas(),
                self.arm.lengths(),
                self.arm.offsets(),
            )?;

        let ik_arm_thetas: nalgebra::Vector5<f64> = Self::s_ik_arm_ef_position_with_eps(
            result_torso.position(),
            result_torso.orientation(),
            self.arm.thetas(),
            self.arm.lengths(),
            self.arm.offsets(),
            &ik_arm_end_effector_target_position,
            epsilon,
            self.pseudo_inverse_epsilon,
        )?;

        *self.arm.thetas_mut() = ik_arm_thetas;

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

        let ik_arm_end_effector_target_position: nalgebra::Vector3<f64> =
            Self::s_fk_arm_ef_position(
                torso.position(),
                torso.orientation(),
                self.arm.thetas(),
                self.arm.lengths(),
                self.arm.offsets(),
            )?;

        let ik_arm_thetas: nalgebra::Vector5<f64> = Self::s_ik_arm_ef_position_with_eps(
            result_torso.position(),
            result_torso.orientation(),
            self.arm.thetas(),
            self.arm.lengths(),
            self.arm.offsets(),
            &ik_arm_end_effector_target_position,
            epsilon,
            self.pseudo_inverse_epsilon,
        )?;

        *self.arm.thetas_mut() = ik_arm_thetas;

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

    fn s_fk_arm_ef_position(
        torso_position: &nalgebra::Vector3<f64>,
        torso_orientation: &nalgebra::Vector3<f64>,
        arm_thetas: &nalgebra::Vector5<f64>,
        arm_lengths: &nalgebra::Vector6<f64>,
        arm_offsets: &nalgebra::Vector1<f64>,
    ) -> Result<nalgebra::Vector3<f64>, Error> {
        let (xt, yt, zt): (f64, f64, f64) = (torso_position.x, torso_position.y, torso_position.z);

        let (at, bt, gt): (f64, f64, f64) = (
            torso_orientation.x,
            torso_orientation.y,
            torso_orientation.z,
        );

        let (t0, t1, t2, t3): (f64, f64, f64, f64) =
            (arm_thetas.x, arm_thetas.y, arm_thetas.z, arm_thetas.w);

        let (lt, l0, l1, l2, l3, l4): (f64, f64, f64, f64, f64, f64) = (
            arm_lengths.x,
            arm_lengths.y,
            arm_lengths.z,
            arm_lengths.w,
            arm_lengths.a,
            arm_lengths.b,
        );

        let o2: f64 = arm_offsets.x;

        Ok(nalgebra::Vector3::<f64>::new(
            -l0 * (at.sin() * gt.cos() - bt.sin() * gt.sin() * at.cos())
                + l1 * ((at.sin() * gt.sin() + bt.sin() * at.cos() * gt.cos()) * t0.sin()
                    - at.cos() * bt.cos() * t0.cos())
                    * t1.sin()
                - l1 * (at.sin() * gt.cos() - bt.sin() * gt.sin() * at.cos()) * t1.cos()
                - l3 * (-at.sin() * gt.sin() * t0.sin() * (t1 + t2).sin()
                    + at.sin() * gt.cos() * (t1 + t2).cos()
                    - bt.sin() * gt.sin() * at.cos() * (t1 + t2).cos()
                    - bt.sin() * t0.sin() * (t1 + t2).sin() * at.cos() * gt.cos()
                    + (t1 + t2).sin() * at.cos() * bt.cos() * t0.cos())
                    * t3.cos()
                - l3 * (-at.sin() * gt.sin() * t0.sin() * (t1 + t2 + t3).sin()
                    + at.sin() * gt.cos() * (t1 + t2 + t3).cos()
                    - bt.sin() * gt.sin() * at.cos() * (t1 + t2 + t3).cos()
                    - bt.sin() * t0.sin() * (t1 + t2 + t3).sin() * at.cos() * gt.cos()
                    + (t1 + t2 + t3).sin() * at.cos() * bt.cos() * t0.cos())
                    * t3.cos()
                + l3 * (at.sin() * gt.sin() * t0.sin() * (t1 + t2).cos()
                    + at.sin() * (t1 + t2).sin() * gt.cos()
                    - bt.sin() * gt.sin() * (t1 + t2).sin() * at.cos()
                    + bt.sin() * t0.sin() * at.cos() * gt.cos() * (t1 + t2).cos()
                    - at.cos() * bt.cos() * t0.cos() * (t1 + t2).cos())
                    * t3.sin()
                + l3 * (at.sin() * gt.sin() * t0.sin() * (t1 + t2 + t3).cos()
                    + at.sin() * (t1 + t2 + t3).sin() * gt.cos()
                    - bt.sin() * gt.sin() * (t1 + t2 + t3).sin() * at.cos()
                    + bt.sin() * t0.sin() * at.cos() * gt.cos() * (t1 + t2 + t3).cos()
                    - at.cos() * bt.cos() * t0.cos() * (t1 + t2 + t3).cos())
                    * t3.sin()
                - l4 * ((-at.sin() * gt.sin() * t0.sin() * (t1 + t2 + t3).sin()
                    + at.sin() * gt.cos() * (t1 + t2 + t3).cos()
                    - bt.sin() * gt.sin() * at.cos() * (t1 + t2 + t3).cos()
                    - bt.sin() * t0.sin() * (t1 + t2 + t3).sin() * at.cos() * gt.cos()
                    + (t1 + t2 + t3).sin() * at.cos() * bt.cos() * t0.cos())
                    * t3.cos()
                    + (-at.sin() * gt.sin() * t0.sin() * (t1 + t2 + t3).cos()
                        - at.sin() * (t1 + t2 + t3).sin() * gt.cos()
                        + bt.sin() * gt.sin() * (t1 + t2 + t3).sin() * at.cos()
                        - bt.sin() * t0.sin() * at.cos() * gt.cos() * (t1 + t2 + t3).cos()
                        + at.cos() * bt.cos() * t0.cos() * (t1 + t2 + t3).cos())
                        * t3.sin())
                - yt * (at.sin() * gt.cos() - bt.sin() * gt.sin() * at.cos())
                + zt * (at.sin() * gt.sin() + bt.sin() * at.cos() * gt.cos())
                + (lt + xt) * at.cos() * bt.cos()
                + (l2 * t2.sin() - o2 * t2.cos())
                    * (((at.sin() * gt.sin() + bt.sin() * at.cos() * gt.cos()) * t0.sin()
                        - at.cos() * bt.cos() * t0.cos())
                        * t1.cos()
                        + (at.sin() * gt.cos() - bt.sin() * gt.sin() * at.cos()) * t1.sin())
                + (l2 * t2.cos() + o2 * t2.sin())
                    * (((at.sin() * gt.sin() + bt.sin() * at.cos() * gt.cos()) * t0.sin()
                        - at.cos() * bt.cos() * t0.cos())
                        * t1.sin()
                        - (at.sin() * gt.cos() - bt.sin() * gt.sin() * at.cos()) * t1.cos()),
            l0 * (at.sin() * bt.sin() * gt.sin() + at.cos() * gt.cos())
                + l1 * ((at.sin() * bt.sin() * gt.cos() - gt.sin() * at.cos()) * t0.sin()
                    - at.sin() * bt.cos() * t0.cos())
                    * t1.sin()
                + l1 * (at.sin() * bt.sin() * gt.sin() + at.cos() * gt.cos()) * t1.cos()
                + l3 * (-at.sin() * bt.sin() * gt.sin() * (t1 + t2).sin()
                    + at.sin() * bt.sin() * t0.sin() * gt.cos() * (t1 + t2).cos()
                    - at.sin() * bt.cos() * t0.cos() * (t1 + t2).cos()
                    - gt.sin() * t0.sin() * at.cos() * (t1 + t2).cos()
                    - (t1 + t2).sin() * at.cos() * gt.cos())
                    * t3.sin()
                + l3 * (-at.sin() * bt.sin() * gt.sin() * (t1 + t2 + t3).sin()
                    + at.sin() * bt.sin() * t0.sin() * gt.cos() * (t1 + t2 + t3).cos()
                    - at.sin() * bt.cos() * t0.cos() * (t1 + t2 + t3).cos()
                    - gt.sin() * t0.sin() * at.cos() * (t1 + t2 + t3).cos()
                    - (t1 + t2 + t3).sin() * at.cos() * gt.cos())
                    * t3.sin()
                - l3 * (-at.sin() * bt.sin() * gt.sin() * (t1 + t2).cos()
                    - at.sin() * bt.sin() * t0.sin() * (t1 + t2).sin() * gt.cos()
                    + at.sin() * (t1 + t2).sin() * bt.cos() * t0.cos()
                    + gt.sin() * t0.sin() * (t1 + t2).sin() * at.cos()
                    - at.cos() * gt.cos() * (t1 + t2).cos())
                    * t3.cos()
                + l3 * (at.sin() * bt.sin() * gt.sin() * (t1 + t2 + t3).cos()
                    + at.sin() * bt.sin() * t0.sin() * (t1 + t2 + t3).sin() * gt.cos()
                    - at.sin() * (t1 + t2 + t3).sin() * bt.cos() * t0.cos()
                    - gt.sin() * t0.sin() * (t1 + t2 + t3).sin() * at.cos()
                    + at.cos() * gt.cos() * (t1 + t2 + t3).cos())
                    * t3.cos()
                - l4 * (-(-at.sin() * bt.sin() * gt.sin() * (t1 + t2 + t3).sin()
                    + at.sin() * bt.sin() * t0.sin() * gt.cos() * (t1 + t2 + t3).cos()
                    - at.sin() * bt.cos() * t0.cos() * (t1 + t2 + t3).cos()
                    - gt.sin() * t0.sin() * at.cos() * (t1 + t2 + t3).cos()
                    - (t1 + t2 + t3).sin() * at.cos() * gt.cos())
                    * t3.sin()
                    + (-at.sin() * bt.sin() * gt.sin() * (t1 + t2 + t3).cos()
                        - at.sin() * bt.sin() * t0.sin() * (t1 + t2 + t3).sin() * gt.cos()
                        + at.sin() * (t1 + t2 + t3).sin() * bt.cos() * t0.cos()
                        + gt.sin() * t0.sin() * (t1 + t2 + t3).sin() * at.cos()
                        - at.cos() * gt.cos() * (t1 + t2 + t3).cos())
                        * t3.cos())
                + yt * (at.sin() * bt.sin() * gt.sin() + at.cos() * gt.cos())
                + zt * (at.sin() * bt.sin() * gt.cos() - gt.sin() * at.cos())
                + (lt + xt) * at.sin() * bt.cos()
                + (l2 * t2.sin() - o2 * t2.cos())
                    * (((at.sin() * bt.sin() * gt.cos() - gt.sin() * at.cos()) * t0.sin()
                        - at.sin() * bt.cos() * t0.cos())
                        * t1.cos()
                        - (at.sin() * bt.sin() * gt.sin() + at.cos() * gt.cos()) * t1.sin())
                + (l2 * t2.cos() + o2 * t2.sin())
                    * (((at.sin() * bt.sin() * gt.cos() - gt.sin() * at.cos()) * t0.sin()
                        - at.sin() * bt.cos() * t0.cos())
                        * t1.sin()
                        + (at.sin() * bt.sin() * gt.sin() + at.cos() * gt.cos()) * t1.cos()),
            l0 * gt.sin() * bt.cos()
                + l1 * (bt.sin() * t0.cos() + t0.sin() * bt.cos() * gt.cos()) * t1.sin()
                + l1 * gt.sin() * bt.cos() * t1.cos()
                + l3 * (bt.sin() * (t1 + t2).sin() * t0.cos()
                    + gt.sin() * bt.cos() * (t1 + t2).cos()
                    + t0.sin() * (t1 + t2).sin() * bt.cos() * gt.cos())
                    * t3.cos()
                + l3 * (bt.sin() * (t1 + t2 + t3).sin() * t0.cos()
                    + gt.sin() * bt.cos() * (t1 + t2 + t3).cos()
                    + t0.sin() * (t1 + t2 + t3).sin() * bt.cos() * gt.cos())
                    * t3.cos()
                + l3 * (bt.sin() * t0.cos() * (t1 + t2).cos()
                    - gt.sin() * (t1 + t2).sin() * bt.cos()
                    + t0.sin() * bt.cos() * gt.cos() * (t1 + t2).cos())
                    * t3.sin()
                + l3 * (bt.sin() * t0.cos() * (t1 + t2 + t3).cos()
                    - gt.sin() * (t1 + t2 + t3).sin() * bt.cos()
                    + t0.sin() * bt.cos() * gt.cos() * (t1 + t2 + t3).cos())
                    * t3.sin()
                - 1_f64 / 8_f64
                    * l4
                    * (2_f64 * (-bt - gt + t1 + t2 + 2_f64 * t3).sin()
                        - 2_f64 * (-bt + gt + t1 + t2 + 2_f64 * t3).sin()
                        + 2_f64 * (bt - gt + t1 + t2 + 2_f64 * t3).sin()
                        - 2_f64 * (bt + gt + t1 + t2 + 2_f64 * t3).sin()
                        - 2_f64 * (-bt - t0 + t1 + t2 + 2_f64 * t3).cos()
                        - 2_f64 * (-bt + t0 + t1 + t2 + 2_f64 * t3).cos()
                        + 2_f64 * (bt - t0 + t1 + t2 + 2_f64 * t3).cos()
                        + 2_f64 * (bt + t0 + t1 + t2 + 2_f64 * t3).cos()
                        + (-bt - gt + t0 + t1 + t2 + 2_f64 * t3).cos()
                        - (-bt + gt - t0 + t1 + t2 + 2_f64 * t3).cos()
                        + (-bt + gt + t0 + t1 + t2 + 2_f64 * t3).cos()
                        - (bt - gt - t0 + t1 + t2 + 2_f64 * t3).cos()
                        + (bt - gt + t0 + t1 + t2 + 2_f64 * t3).cos()
                        - (bt + gt - t0 + t1 + t2 + 2_f64 * t3).cos()
                        - (bt + gt + t0 - t1 - t2 - 2_f64 * t3).cos()
                        + (bt + gt + t0 + t1 + t2 + 2_f64 * t3).cos())
                + yt * gt.sin() * bt.cos()
                + zt * bt.cos() * gt.cos()
                - (lt + xt) * bt.sin()
                + (l2 * t2.sin() - o2 * t2.cos())
                    * ((bt.sin() * t0.cos() + t0.sin() * bt.cos() * gt.cos()) * t1.cos()
                        - gt.sin() * t1.sin() * bt.cos())
                + (l2 * t2.cos() + o2 * t2.sin())
                    * ((bt.sin() * t0.cos() + t0.sin() * bt.cos() * gt.cos()) * t1.sin()
                        + gt.sin() * bt.cos() * t1.cos()),
        ))
    }

    pub fn fk_arm_ef_position(&self) -> Result<nalgebra::Vector3<f64>, Error> {
        Self::s_fk_arm_ef_position(
            self.torso.position(),
            self.torso.orientation(),
            self.arm.thetas(),
            self.arm.lengths(),
            self.arm.offsets(),
        )
    }

    /// Gets the torso transformation matrix for the given parameters.
    fn arm_m_t(
        torso_position: &nalgebra::Vector3<f64>,
        torso_orientation: &nalgebra::Vector3<f64>,
        arm_lengths: &nalgebra::Vector6<f64>,
    ) -> Result<nalgebra::Matrix4<f64>, Error> {
        let (xt, yt, zt) = (torso_position.x, torso_position.y, torso_position.z);

        let (at, bt, gt) = (
            torso_orientation.x,
            torso_orientation.y,
            torso_orientation.z,
        );

        let lt = arm_lengths.x;

        Ok(nalgebra::Matrix4::<f64>::new(
            at.cos() * bt.cos(),
            -at.sin() * gt.cos() + bt.sin() * gt.sin() * at.cos(),
            at.sin() * gt.sin() + bt.sin() * at.cos() * gt.cos(),
            yt * (-at.sin() * gt.cos() + bt.sin() * gt.sin() * at.cos())
                + zt * (at.sin() * gt.sin() + bt.sin() * at.cos() * gt.cos())
                + (lt + xt) * at.cos() * bt.cos(),
            at.sin() * bt.cos(),
            at.sin() * bt.sin() * gt.sin() + at.cos() * gt.cos(),
            at.sin() * bt.sin() * gt.cos() - gt.sin() * at.cos(),
            yt * (at.sin() * bt.sin() * gt.sin() + at.cos() * gt.cos())
                + zt * (at.sin() * bt.sin() * gt.cos() - gt.sin() * at.cos())
                + (lt + xt) * at.sin() * bt.cos(),
            -bt.sin(),
            gt.sin() * bt.cos(),
            bt.cos() * gt.cos(),
            yt * gt.sin() * bt.cos() + zt * bt.cos() * gt.cos() - (lt + xt) * bt.sin(),
            0_f64,
            0_f64,
            0_f64,
            1_f64,
        ))
    }

    /// Gets the arm M0 transformation matrix for the given parameters.
    fn arm_m_0(
        arm_thetas: &nalgebra::Vector5<f64>,
        arm_lengths: &nalgebra::Vector6<f64>,
    ) -> Result<nalgebra::Matrix4<f64>, Error> {
        let t0: f64 = arm_thetas.x;
        let l0: f64 = arm_lengths.y;

        Ok(nalgebra::Matrix4::<f64>::new(
            t0.cos(),
            0_f64,
            t0.sin(),
            0_f64,
            0_f64,
            1_f64,
            0_f64,
            l0,
            -t0.sin(),
            0_f64,
            t0.cos(),
            0_f64,
            0_f64,
            0_f64,
            0_f64,
            1_f64,
        ))
    }

    /// Gets the arm M1 transformation matrix for the given parameters.
    fn arm_m_1(
        arm_thetas: &nalgebra::Vector5<f64>,
        arm_lengths: &nalgebra::Vector6<f64>,
    ) -> Result<nalgebra::Matrix4<f64>, Error> {
        let t1: f64 = arm_thetas.y;
        let l1: f64 = arm_lengths.z;

        Ok(nalgebra::Matrix4::<f64>::new(
            t1.cos(),
            -t1.sin(),
            0_f64,
            -l1 * t1.sin(),
            t1.sin(),
            t1.cos(),
            0_f64,
            l1 * t1.cos(),
            0_f64,
            0_f64,
            0_f64,
            1_f64,
            0_f64,
            0_f64,
            0_f64,
            1_f64,
        ))
    }

    /// Gets the arm M2 transformation matrix for the given parameters.
    fn arm_m_2(
        arm_thetas: &nalgebra::Vector5<f64>,
        arm_lengths: &nalgebra::Vector6<f64>,
        arm_offsets: &nalgebra::Vector1<f64>,
    ) -> Result<nalgebra::Matrix4<f64>, Error> {
        let t2: f64 = arm_thetas.z;
        let l2: f64 = arm_lengths.w;
        let o2: f64 = arm_offsets.x;

        Ok(nalgebra::Matrix4::<f64>::new(
            t2.cos(),
            -t2.sin(),
            0_f64,
            -l2 * t2.sin() + o2 * t2.cos(),
            t2.sin(),
            t2.cos(),
            0_f64,
            l2 * t2.cos() + o2 * t2.sin(),
            0_f64,
            0_f64,
            1_f64,
            0_f64,
            0_f64,
            0_f64,
            0_f64,
            1_f64,
        ))
    }

    /// Gets the arm M3 transformation matrix for the given parameters.
    fn arm_m_3(
        arm_thetas: &nalgebra::Vector5<f64>,
        arm_lengths: &nalgebra::Vector6<f64>,
    ) -> Result<nalgebra::Matrix4<f64>, Error> {
        let t3: f64 = arm_thetas.w;
        let l3: f64 = arm_lengths.a;

        Ok(nalgebra::Matrix4::<f64>::new(
            t3.cos(),
            -t3.sin(),
            0_f64,
            -l3 * t3.sin(),
            t3.sin(),
            t3.cos(),
            0_f64,
            l3 * t3.cos(),
            0_f64,
            0_f64,
            0_f64,
            1_f64,
            0_f64,
            0_f64,
            0_f64,
            1_f64,
        ))
    }

    /// Gets the arm M4 transformation matrix for the given parameters.
    fn arm_m_4(
        arm_thetas: &nalgebra::Vector5<f64>,
        arm_lengths: &nalgebra::Vector6<f64>,
    ) -> Result<nalgebra::Matrix4<f64>, Error> {
        let t4: f64 = arm_thetas.a;
        let l4: f64 = arm_lengths.b;

        Ok(nalgebra::Matrix4::<f64>::new(
            t4.cos(),
            0_f64,
            t4.sin(),
            0_f64,
            0_f64,
            1_f64,
            0_f64,
            l4,
            -t4.sin(),
            0_f64,
            t4.cos(),
            0_f64,
            0_f64,
            0_f64,
            0_f64,
            1_f64,
        ))
    }

    /// Gets all the vertices for an arm with the given parameters.
    fn s_arm_vertices(
        torso_position: &nalgebra::Vector3<f64>,
        torso_orientation: &nalgebra::Vector3<f64>,
        arm_thetas: &nalgebra::Vector5<f64>,
        arm_lengths: &nalgebra::Vector6<f64>,
        arm_offsets: &nalgebra::Vector1<f64>,
    ) -> Result<[nalgebra::Vector3<f64>; 7], Error> {
        let vertex_0: nalgebra::Vector3<f64> = torso_position.clone();

        let mut m: nalgebra::Matrix4<f64> =
            Self::arm_m_t(torso_position, torso_orientation, arm_lengths)?;

        let vertex_1: nalgebra::Vector3<f64> = m.fixed_slice::<3, 1>(0, 3).into();
        m *= Self::arm_m_0(arm_thetas, arm_lengths)?;
        let vertex_2: nalgebra::Vector3<f64> = m.fixed_slice::<3, 1>(0, 3).into();
        m *= Self::arm_m_1(arm_thetas, arm_lengths)?;
        let vertex_3: nalgebra::Vector3<f64> = m.fixed_slice::<3, 1>(0, 3).into();
        m *= Self::arm_m_2(arm_thetas, arm_lengths, arm_offsets)?;
        let vertex_4: nalgebra::Vector3<f64> = m.fixed_slice::<3, 1>(0, 3).into();
        m *= Self::arm_m_3(arm_thetas, arm_lengths)?;
        let vertex_5: nalgebra::Vector3<f64> = m.fixed_slice::<3, 1>(0, 3).into();
        m *= Self::arm_m_4(arm_thetas, arm_lengths)?;
        let vertex_6: nalgebra::Vector3<f64> = m.fixed_slice::<3, 1>(0, 3).into();

        Ok([
            vertex_0, vertex_1, vertex_2, vertex_3, vertex_4, vertex_5, vertex_6,
        ])
    }

    /// Gets all the vertices of the arm in the current solver.
    pub fn arm_vertices(&self) -> Result<[nalgebra::Vector3<f64>; 7], Error> {
        Self::s_arm_vertices(
            self.torso.position(),
            self.torso.orientation(),
            self.arm.thetas(),
            self.arm.lengths(),
            self.arm.offsets(),
        )
    }

    /// Performs a single inverse kinematics cycle with the given parameters.
    fn s_ik_arm_ef_position(
        torso_orientation: &nalgebra::Vector3<f64>,
        arm_thetas: &nalgebra::Vector5<f64>,
        arm_lengths: &nalgebra::Vector6<f64>,
        arm_offsets: &nalgebra::Vector1<f64>,
        delta_position: &nalgebra::Vector3<f64>,
        pseudo_inverse_epsilon: f64,
    ) -> Result<nalgebra::Vector5<f64>, Error> {
        let (at, bt, gt): (f64, f64, f64) = (
            torso_orientation.x,
            torso_orientation.y,
            torso_orientation.z,
        );

        let (t0, t1, t2, t3): (f64, f64, f64, f64) =
            (arm_thetas.x, arm_thetas.y, arm_thetas.z, arm_thetas.w);

        let (l1, l2, l3, l4): (f64, f64, f64, f64) =
            (arm_lengths.z, arm_lengths.w, arm_lengths.a, arm_lengths.b);

        let o2: f64 = arm_offsets.x;

        let jacobian: nalgebra::Matrix<f64, nalgebra::U3, nalgebra::U5, nalgebra::ArrayStorage<f64, 3, 5>> = nalgebra::Matrix::<f64, nalgebra::U3, nalgebra::U5, nalgebra::ArrayStorage<f64, 3, 5>>::new(
        l1*((at.sin()*gt.sin() + bt.sin()*at.cos()*gt.cos())*t0.cos() + t0.sin()*at.cos()*bt.cos())*t1.sin() + l3*(at.sin()*gt.sin()*t0.cos() + bt.sin()*at.cos()*gt.cos()*t0.cos() + t0.sin()*at.cos()*bt.cos())*t3.sin()*(t1 + t2).cos() + l3*(at.sin()*gt.sin()*t0.cos() + bt.sin()*at.cos()*gt.cos()*t0.cos() + t0.sin()*at.cos()*bt.cos())*t3.sin()*(t1 + t2 + t3).cos() + l3*(at.sin()*gt.sin()*t0.cos() + bt.sin()*at.cos()*gt.cos()*t0.cos() + t0.sin()*at.cos()*bt.cos())*(t1 + t2).sin()*t3.cos() + l3*(at.sin()*gt.sin()*t0.cos() + bt.sin()*at.cos()*gt.cos()*t0.cos() + t0.sin()*at.cos()*bt.cos())*(t1 + t2 + t3).sin()*t3.cos() + l4*(at.sin()*gt.sin()*t0.cos() + bt.sin()*at.cos()*gt.cos()*t0.cos() + t0.sin()*at.cos()*bt.cos())*(t1 + t2 + 2_f64*t3).sin() + (l2*t2.sin() - o2*t2.cos())*((at.sin()*gt.sin() + bt.sin()*at.cos()*gt.cos())*t0.cos() + t0.sin()*at.cos()*bt.cos())*t1.cos() + (l2*t2.cos() + o2*t2.sin())*((at.sin()*gt.sin() + bt.sin()*at.cos()*gt.cos())*t0.cos() + t0.sin()*at.cos()*bt.cos())*t1.sin(), l1*((at.sin()*gt.sin() + bt.sin()*at.cos()*gt.cos())*t0.sin() - at.cos()*bt.cos()*t0.cos())*t1.cos() + l1*(at.sin()*gt.cos() - bt.sin()*gt.sin()*at.cos())*t1.sin() - l3*(at.sin()*gt.sin()*t0.sin()*(t1 + t2).sin() - at.sin()*gt.cos()*(t1 + t2).cos() + bt.sin()*gt.sin()*at.cos()*(t1 + t2).cos() + bt.sin()*t0.sin()*(t1 + t2).sin()*at.cos()*gt.cos() - (t1 + t2).sin()*at.cos()*bt.cos()*t0.cos())*t3.sin() - l3*(at.sin()*gt.sin()*t0.sin()*(t1 + t2 + t3).sin() - at.sin()*gt.cos()*(t1 + t2 + t3).cos() + bt.sin()*gt.sin()*at.cos()*(t1 + t2 + t3).cos() + bt.sin()*t0.sin()*(t1 + t2 + t3).sin()*at.cos()*gt.cos() - (t1 + t2 + t3).sin()*at.cos()*bt.cos()*t0.cos())*t3.sin() + l3*(at.sin()*gt.sin()*t0.sin()*(t1 + t2).cos() + at.sin()*(t1 + t2).sin()*gt.cos() - bt.sin()*gt.sin()*(t1 + t2).sin()*at.cos() + bt.sin()*t0.sin()*at.cos()*gt.cos()*(t1 + t2).cos() - at.cos()*bt.cos()*t0.cos()*(t1 + t2).cos())*t3.cos() + l3*(at.sin()*gt.sin()*t0.sin()*(t1 + t2 + t3).cos() + at.sin()*(t1 + t2 + t3).sin()*gt.cos() - bt.sin()*gt.sin()*(t1 + t2 + t3).sin()*at.cos() + bt.sin()*t0.sin()*at.cos()*gt.cos()*(t1 + t2 + t3).cos() - at.cos()*bt.cos()*t0.cos()*(t1 + t2 + t3).cos())*t3.cos() - l4*((at.sin()*gt.sin()*t0.sin()*(t1 + t2 + t3).sin() - at.sin()*gt.cos()*(t1 + t2 + t3).cos() + bt.sin()*gt.sin()*at.cos()*(t1 + t2 + t3).cos() + bt.sin()*t0.sin()*(t1 + t2 + t3).sin()*at.cos()*gt.cos() - (t1 + t2 + t3).sin()*at.cos()*bt.cos()*t0.cos())*t3.sin() - (at.sin()*gt.sin()*t0.sin()*(t1 + t2 + t3).cos() + at.sin()*(t1 + t2 + t3).sin()*gt.cos() - bt.sin()*gt.sin()*(t1 + t2 + t3).sin()*at.cos() + bt.sin()*t0.sin()*at.cos()*gt.cos()*(t1 + t2 + t3).cos() - at.cos()*bt.cos()*t0.cos()*(t1 + t2 + t3).cos())*t3.cos()) - (l2*t2.sin() - o2*t2.cos())*(((at.sin()*gt.sin() + bt.sin()*at.cos()*gt.cos())*t0.sin() - at.cos()*bt.cos()*t0.cos())*t1.sin() - (at.sin()*gt.cos() - bt.sin()*gt.sin()*at.cos())*t1.cos()) + (l2*t2.cos() + o2*t2.sin())*(((at.sin()*gt.sin() + bt.sin()*at.cos()*gt.cos())*t0.sin() - at.cos()*bt.cos()*t0.cos())*t1.cos() + (at.sin()*gt.cos() - bt.sin()*gt.sin()*at.cos())*t1.sin()), -l3*(at.sin()*gt.sin()*t0.sin()*(t1 + t2).sin() - at.sin()*gt.cos()*(t1 + t2).cos() + bt.sin()*gt.sin()*at.cos()*(t1 + t2).cos() + bt.sin()*t0.sin()*(t1 + t2).sin()*at.cos()*gt.cos() - (t1 + t2).sin()*at.cos()*bt.cos()*t0.cos())*t3.sin() - l3*(at.sin()*gt.sin()*t0.sin()*(t1 + t2 + t3).sin() - at.sin()*gt.cos()*(t1 + t2 + t3).cos() + bt.sin()*gt.sin()*at.cos()*(t1 + t2 + t3).cos() + bt.sin()*t0.sin()*(t1 + t2 + t3).sin()*at.cos()*gt.cos() - (t1 + t2 + t3).sin()*at.cos()*bt.cos()*t0.cos())*t3.sin() + l3*(at.sin()*gt.sin()*t0.sin()*(t1 + t2).cos() + at.sin()*(t1 + t2).sin()*gt.cos() - bt.sin()*gt.sin()*(t1 + t2).sin()*at.cos() + bt.sin()*t0.sin()*at.cos()*gt.cos()*(t1 + t2).cos() - at.cos()*bt.cos()*t0.cos()*(t1 + t2).cos())*t3.cos() + l3*(at.sin()*gt.sin()*t0.sin()*(t1 + t2 + t3).cos() + at.sin()*(t1 + t2 + t3).sin()*gt.cos() - bt.sin()*gt.sin()*(t1 + t2 + t3).sin()*at.cos() + bt.sin()*t0.sin()*at.cos()*gt.cos()*(t1 + t2 + t3).cos() - at.cos()*bt.cos()*t0.cos()*(t1 + t2 + t3).cos())*t3.cos() - l4*((at.sin()*gt.sin()*t0.sin()*(t1 + t2 + t3).sin() - at.sin()*gt.cos()*(t1 + t2 + t3).cos() + bt.sin()*gt.sin()*at.cos()*(t1 + t2 + t3).cos() + bt.sin()*t0.sin()*(t1 + t2 + t3).sin()*at.cos()*gt.cos() - (t1 + t2 + t3).sin()*at.cos()*bt.cos()*t0.cos())*t3.sin() - (at.sin()*gt.sin()*t0.sin()*(t1 + t2 + t3).cos() + at.sin()*(t1 + t2 + t3).sin()*gt.cos() - bt.sin()*gt.sin()*(t1 + t2 + t3).sin()*at.cos() + bt.sin()*t0.sin()*at.cos()*gt.cos()*(t1 + t2 + t3).cos() - at.cos()*bt.cos()*t0.cos()*(t1 + t2 + t3).cos())*t3.cos()) - (l2*t2.sin() - o2*t2.cos())*(((at.sin()*gt.sin() + bt.sin()*at.cos()*gt.cos())*t0.sin() - at.cos()*bt.cos()*t0.cos())*t1.sin() - (at.sin()*gt.cos() - bt.sin()*gt.sin()*at.cos())*t1.cos()) + (l2*t2.cos() + o2*t2.sin())*(((at.sin()*gt.sin() + bt.sin()*at.cos()*gt.cos())*t0.sin() - at.cos()*bt.cos()*t0.cos())*t1.cos() + (at.sin()*gt.cos() - bt.sin()*gt.sin()*at.cos())*t1.sin()), -l3*(at.sin()*gt.sin()*t0.sin()*(t1 + t2).sin() - at.sin()*gt.cos()*(t1 + t2).cos() + bt.sin()*gt.sin()*at.cos()*(t1 + t2).cos() + bt.sin()*t0.sin()*(t1 + t2).sin()*at.cos()*gt.cos() - (t1 + t2).sin()*at.cos()*bt.cos()*t0.cos())*t3.sin() - 2_f64*l3*(at.sin()*gt.sin()*t0.sin()*(t1 + t2 + t3).sin() - at.sin()*gt.cos()*(t1 + t2 + t3).cos() + bt.sin()*gt.sin()*at.cos()*(t1 + t2 + t3).cos() + bt.sin()*t0.sin()*(t1 + t2 + t3).sin()*at.cos()*gt.cos() - (t1 + t2 + t3).sin()*at.cos()*bt.cos()*t0.cos())*t3.sin() + l3*(at.sin()*gt.sin()*t0.sin()*(t1 + t2).cos() + at.sin()*(t1 + t2).sin()*gt.cos() - bt.sin()*gt.sin()*(t1 + t2).sin()*at.cos() + bt.sin()*t0.sin()*at.cos()*gt.cos()*(t1 + t2).cos() - at.cos()*bt.cos()*t0.cos()*(t1 + t2).cos())*t3.cos() + 2_f64*l3*(at.sin()*gt.sin()*t0.sin()*(t1 + t2 + t3).cos() + at.sin()*(t1 + t2 + t3).sin()*gt.cos() - bt.sin()*gt.sin()*(t1 + t2 + t3).sin()*at.cos() + bt.sin()*t0.sin()*at.cos()*gt.cos()*(t1 + t2 + t3).cos() - at.cos()*bt.cos()*t0.cos()*(t1 + t2 + t3).cos())*t3.cos() - 2_f64*l4*((at.sin()*gt.sin()*t0.sin()*(t1 + t2 + t3).sin() - at.sin()*gt.cos()*(t1 + t2 + t3).cos() + bt.sin()*gt.sin()*at.cos()*(t1 + t2 + t3).cos() + bt.sin()*t0.sin()*(t1 + t2 + t3).sin()*at.cos()*gt.cos() - (t1 + t2 + t3).sin()*at.cos()*bt.cos()*t0.cos())*t3.sin() - (at.sin()*gt.sin()*t0.sin()*(t1 + t2 + t3).cos() + at.sin()*(t1 + t2 + t3).sin()*gt.cos() - bt.sin()*gt.sin()*(t1 + t2 + t3).sin()*at.cos() + bt.sin()*t0.sin()*at.cos()*gt.cos()*(t1 + t2 + t3).cos() - at.cos()*bt.cos()*t0.cos()*(t1 + t2 + t3).cos())*t3.cos()), 0.0,
        l1*((at.sin()*bt.sin()*gt.cos() - gt.sin()*at.cos())*t0.cos() + at.sin()*t0.sin()*bt.cos())*t1.sin() + l3*(at.sin()*bt.sin()*gt.cos()*t0.cos() + at.sin()*t0.sin()*bt.cos() - gt.sin()*at.cos()*t0.cos())*t3.sin()*(t1 + t2).cos() + l3*(at.sin()*bt.sin()*gt.cos()*t0.cos() + at.sin()*t0.sin()*bt.cos() - gt.sin()*at.cos()*t0.cos())*t3.sin()*(t1 + t2 + t3).cos() + l3*(at.sin()*bt.sin()*gt.cos()*t0.cos() + at.sin()*t0.sin()*bt.cos() - gt.sin()*at.cos()*t0.cos())*(t1 + t2).sin()*t3.cos() + l3*(at.sin()*bt.sin()*gt.cos()*t0.cos() + at.sin()*t0.sin()*bt.cos() - gt.sin()*at.cos()*t0.cos())*(t1 + t2 + t3).sin()*t3.cos() + l4*(at.sin()*bt.sin()*gt.cos()*t0.cos() + at.sin()*t0.sin()*bt.cos() - gt.sin()*at.cos()*t0.cos())*(t1 + t2 + 2_f64*t3).sin() + (l2*t2.sin() - o2*t2.cos())*((at.sin()*bt.sin()*gt.cos() - gt.sin()*at.cos())*t0.cos() + at.sin()*t0.sin()*bt.cos())*t1.cos() + (l2*t2.cos() + o2*t2.sin())*((at.sin()*bt.sin()*gt.cos() - gt.sin()*at.cos())*t0.cos() + at.sin()*t0.sin()*bt.cos())*t1.sin(), l1*((at.sin()*bt.sin()*gt.cos() - gt.sin()*at.cos())*t0.sin() - at.sin()*bt.cos()*t0.cos())*t1.cos() - l1*(at.sin()*bt.sin()*gt.sin() + at.cos()*gt.cos())*t1.sin() - l3*(at.sin()*bt.sin()*gt.sin()*(t1 + t2).sin() - at.sin()*bt.sin()*t0.sin()*gt.cos()*(t1 + t2).cos() + at.sin()*bt.cos()*t0.cos()*(t1 + t2).cos() + gt.sin()*t0.sin()*at.cos()*(t1 + t2).cos() + (t1 + t2).sin()*at.cos()*gt.cos())*t3.cos() - l3*(at.sin()*bt.sin()*gt.sin()*(t1 + t2 + t3).sin() - at.sin()*bt.sin()*t0.sin()*gt.cos()*(t1 + t2 + t3).cos() + at.sin()*bt.cos()*t0.cos()*(t1 + t2 + t3).cos() + gt.sin()*t0.sin()*at.cos()*(t1 + t2 + t3).cos() + (t1 + t2 + t3).sin()*at.cos()*gt.cos())*t3.cos() - l3*(at.sin()*bt.sin()*gt.sin()*(t1 + t2).cos() + at.sin()*bt.sin()*t0.sin()*(t1 + t2).sin()*gt.cos() - at.sin()*(t1 + t2).sin()*bt.cos()*t0.cos() - gt.sin()*t0.sin()*(t1 + t2).sin()*at.cos() + at.cos()*gt.cos()*(t1 + t2).cos())*t3.sin() - l3*(at.sin()*bt.sin()*gt.sin()*(t1 + t2 + t3).cos() + at.sin()*bt.sin()*t0.sin()*(t1 + t2 + t3).sin()*gt.cos() - at.sin()*(t1 + t2 + t3).sin()*bt.cos()*t0.cos() - gt.sin()*t0.sin()*(t1 + t2 + t3).sin()*at.cos() + at.cos()*gt.cos()*(t1 + t2 + t3).cos())*t3.sin() - l4*((at.sin()*bt.sin()*gt.sin()*(t1 + t2 + t3).sin() - at.sin()*bt.sin()*t0.sin()*gt.cos()*(t1 + t2 + t3).cos() + at.sin()*bt.cos()*t0.cos()*(t1 + t2 + t3).cos() + gt.sin()*t0.sin()*at.cos()*(t1 + t2 + t3).cos() + (t1 + t2 + t3).sin()*at.cos()*gt.cos())*t3.cos() + (at.sin()*bt.sin()*gt.sin()*(t1 + t2 + t3).cos() + at.sin()*bt.sin()*t0.sin()*(t1 + t2 + t3).sin()*gt.cos() - at.sin()*(t1 + t2 + t3).sin()*bt.cos()*t0.cos() - gt.sin()*t0.sin()*(t1 + t2 + t3).sin()*at.cos() + at.cos()*gt.cos()*(t1 + t2 + t3).cos())*t3.sin()) - (l2*t2.sin() - o2*t2.cos())*(((at.sin()*bt.sin()*gt.cos() - gt.sin()*at.cos())*t0.sin() - at.sin()*bt.cos()*t0.cos())*t1.sin() + (at.sin()*bt.sin()*gt.sin() + at.cos()*gt.cos())*t1.cos()) + (l2*t2.cos() + o2*t2.sin())*(((at.sin()*bt.sin()*gt.cos() - gt.sin()*at.cos())*t0.sin() - at.sin()*bt.cos()*t0.cos())*t1.cos() - (at.sin()*bt.sin()*gt.sin() + at.cos()*gt.cos())*t1.sin()), -l3*(at.sin()*bt.sin()*gt.sin()*(t1 + t2).sin() - at.sin()*bt.sin()*t0.sin()*gt.cos()*(t1 + t2).cos() + at.sin()*bt.cos()*t0.cos()*(t1 + t2).cos() + gt.sin()*t0.sin()*at.cos()*(t1 + t2).cos() + (t1 + t2).sin()*at.cos()*gt.cos())*t3.cos() - l3*(at.sin()*bt.sin()*gt.sin()*(t1 + t2 + t3).sin() - at.sin()*bt.sin()*t0.sin()*gt.cos()*(t1 + t2 + t3).cos() + at.sin()*bt.cos()*t0.cos()*(t1 + t2 + t3).cos() + gt.sin()*t0.sin()*at.cos()*(t1 + t2 + t3).cos() + (t1 + t2 + t3).sin()*at.cos()*gt.cos())*t3.cos() - l3*(at.sin()*bt.sin()*gt.sin()*(t1 + t2).cos() + at.sin()*bt.sin()*t0.sin()*(t1 + t2).sin()*gt.cos() - at.sin()*(t1 + t2).sin()*bt.cos()*t0.cos() - gt.sin()*t0.sin()*(t1 + t2).sin()*at.cos() + at.cos()*gt.cos()*(t1 + t2).cos())*t3.sin() - l3*(at.sin()*bt.sin()*gt.sin()*(t1 + t2 + t3).cos() + at.sin()*bt.sin()*t0.sin()*(t1 + t2 + t3).sin()*gt.cos() - at.sin()*(t1 + t2 + t3).sin()*bt.cos()*t0.cos() - gt.sin()*t0.sin()*(t1 + t2 + t3).sin()*at.cos() + at.cos()*gt.cos()*(t1 + t2 + t3).cos())*t3.sin() - l4*((at.sin()*bt.sin()*gt.sin()*(t1 + t2 + t3).sin() - at.sin()*bt.sin()*t0.sin()*gt.cos()*(t1 + t2 + t3).cos() + at.sin()*bt.cos()*t0.cos()*(t1 + t2 + t3).cos() + gt.sin()*t0.sin()*at.cos()*(t1 + t2 + t3).cos() + (t1 + t2 + t3).sin()*at.cos()*gt.cos())*t3.cos() + (at.sin()*bt.sin()*gt.sin()*(t1 + t2 + t3).cos() + at.sin()*bt.sin()*t0.sin()*(t1 + t2 + t3).sin()*gt.cos() - at.sin()*(t1 + t2 + t3).sin()*bt.cos()*t0.cos() - gt.sin()*t0.sin()*(t1 + t2 + t3).sin()*at.cos() + at.cos()*gt.cos()*(t1 + t2 + t3).cos())*t3.sin()) - (l2*t2.sin() - o2*t2.cos())*(((at.sin()*bt.sin()*gt.cos() - gt.sin()*at.cos())*t0.sin() - at.sin()*bt.cos()*t0.cos())*t1.sin() + (at.sin()*bt.sin()*gt.sin() + at.cos()*gt.cos())*t1.cos()) + (l2*t2.cos() + o2*t2.sin())*(((at.sin()*bt.sin()*gt.cos() - gt.sin()*at.cos())*t0.sin() - at.sin()*bt.cos()*t0.cos())*t1.cos() - (at.sin()*bt.sin()*gt.sin() + at.cos()*gt.cos())*t1.sin()), -l3*(at.sin()*bt.sin()*gt.sin()*(t1 + t2).sin() - at.sin()*bt.sin()*t0.sin()*gt.cos()*(t1 + t2).cos() + at.sin()*bt.cos()*t0.cos()*(t1 + t2).cos() + gt.sin()*t0.sin()*at.cos()*(t1 + t2).cos() + (t1 + t2).sin()*at.cos()*gt.cos())*t3.cos() - 2_f64*l3*(at.sin()*bt.sin()*gt.sin()*(t1 + t2 + t3).sin() - at.sin()*bt.sin()*t0.sin()*gt.cos()*(t1 + t2 + t3).cos() + at.sin()*bt.cos()*t0.cos()*(t1 + t2 + t3).cos() + gt.sin()*t0.sin()*at.cos()*(t1 + t2 + t3).cos() + (t1 + t2 + t3).sin()*at.cos()*gt.cos())*t3.cos() - l3*(at.sin()*bt.sin()*gt.sin()*(t1 + t2).cos() + at.sin()*bt.sin()*t0.sin()*(t1 + t2).sin()*gt.cos() - at.sin()*(t1 + t2).sin()*bt.cos()*t0.cos() - gt.sin()*t0.sin()*(t1 + t2).sin()*at.cos() + at.cos()*gt.cos()*(t1 + t2).cos())*t3.sin() - 2_f64*l3*(at.sin()*bt.sin()*gt.sin()*(t1 + t2 + t3).cos() + at.sin()*bt.sin()*t0.sin()*(t1 + t2 + t3).sin()*gt.cos() - at.sin()*(t1 + t2 + t3).sin()*bt.cos()*t0.cos() - gt.sin()*t0.sin()*(t1 + t2 + t3).sin()*at.cos() + at.cos()*gt.cos()*(t1 + t2 + t3).cos())*t3.sin() - 2_f64*l4*((at.sin()*bt.sin()*gt.sin()*(t1 + t2 + t3).sin() - at.sin()*bt.sin()*t0.sin()*gt.cos()*(t1 + t2 + t3).cos() + at.sin()*bt.cos()*t0.cos()*(t1 + t2 + t3).cos() + gt.sin()*t0.sin()*at.cos()*(t1 + t2 + t3).cos() + (t1 + t2 + t3).sin()*at.cos()*gt.cos())*t3.cos() + (at.sin()*bt.sin()*gt.sin()*(t1 + t2 + t3).cos() + at.sin()*bt.sin()*t0.sin()*(t1 + t2 + t3).sin()*gt.cos() - at.sin()*(t1 + t2 + t3).sin()*bt.cos()*t0.cos() - gt.sin()*t0.sin()*(t1 + t2 + t3).sin()*at.cos() + at.cos()*gt.cos()*(t1 + t2 + t3).cos())*t3.sin()), 0.0,
        -l1*(bt.sin()*t0.sin() - bt.cos()*gt.cos()*t0.cos())*t1.sin() - l3*(bt.sin()*t0.sin() - bt.cos()*gt.cos()*t0.cos())*t3.sin()*(t1 + t2).cos() - l3*(bt.sin()*t0.sin() - bt.cos()*gt.cos()*t0.cos())*t3.sin()*(t1 + t2 + t3).cos() - l3*(bt.sin()*t0.sin() - bt.cos()*gt.cos()*t0.cos())*(t1 + t2).sin()*t3.cos() - l3*(bt.sin()*t0.sin() - bt.cos()*gt.cos()*t0.cos())*(t1 + t2 + t3).sin()*t3.cos() + (1_f64/8.0)*l4*(2_f64*(-bt - t0 + t1 + t2 + 2_f64*t3).sin() - 2_f64*(-bt + t0 + t1 + t2 + 2_f64*t3).sin() - 2_f64*(bt - t0 + t1 + t2 + 2_f64*t3).sin() + 2_f64*(bt + t0 + t1 + t2 + 2_f64*t3).sin() + (-bt - gt + t0 + t1 + t2 + 2_f64*t3).sin() + (-bt + gt - t0 + t1 + t2 + 2_f64*t3).sin() + (-bt + gt + t0 + t1 + t2 + 2_f64*t3).sin() + (bt - gt - t0 + t1 + t2 + 2_f64*t3).sin() + (bt - gt + t0 + t1 + t2 + 2_f64*t3).sin() + (bt + gt - t0 + t1 + t2 + 2_f64*t3).sin() - (bt + gt + t0 - t1 - t2 - 2_f64*t3).sin() + (bt + gt + t0 + t1 + t2 + 2_f64*t3).sin()) - (l2*t2.sin() - o2*t2.cos())*(bt.sin()*t0.sin() - bt.cos()*gt.cos()*t0.cos())*t1.cos() - (l2*t2.cos() + o2*t2.sin())*(bt.sin()*t0.sin() - bt.cos()*gt.cos()*t0.cos())*t1.sin(), l1*(bt.sin()*t0.cos() + t0.sin()*bt.cos()*gt.cos())*t1.cos() - l1*gt.sin()*t1.sin()*bt.cos() - l3*(bt.sin()*(t1 + t2).sin()*t0.cos() + gt.sin()*bt.cos()*(t1 + t2).cos() + t0.sin()*(t1 + t2).sin()*bt.cos()*gt.cos())*t3.sin() - l3*(bt.sin()*(t1 + t2 + t3).sin()*t0.cos() + gt.sin()*bt.cos()*(t1 + t2 + t3).cos() + t0.sin()*(t1 + t2 + t3).sin()*bt.cos()*gt.cos())*t3.sin() + l3*(bt.sin()*t0.cos()*(t1 + t2).cos() - gt.sin()*(t1 + t2).sin()*bt.cos() + t0.sin()*bt.cos()*gt.cos()*(t1 + t2).cos())*t3.cos() + l3*(bt.sin()*t0.cos()*(t1 + t2 + t3).cos() - gt.sin()*(t1 + t2 + t3).sin()*bt.cos() + t0.sin()*bt.cos()*gt.cos()*(t1 + t2 + t3).cos())*t3.cos() + (1_f64/8.0)*l4*(-2_f64*(-bt - t0 + t1 + t2 + 2_f64*t3).sin() - 2_f64*(-bt + t0 + t1 + t2 + 2_f64*t3).sin() + 2_f64*(bt - t0 + t1 + t2 + 2_f64*t3).sin() + 2_f64*(bt + t0 + t1 + t2 + 2_f64*t3).sin() + (-bt - gt + t0 + t1 + t2 + 2_f64*t3).sin() - (-bt + gt - t0 + t1 + t2 + 2_f64*t3).sin() + (-bt + gt + t0 + t1 + t2 + 2_f64*t3).sin() - (bt - gt - t0 + t1 + t2 + 2_f64*t3).sin() + (bt - gt + t0 + t1 + t2 + 2_f64*t3).sin() - (bt + gt - t0 + t1 + t2 + 2_f64*t3).sin() + (bt + gt + t0 - t1 - t2 - 2_f64*t3).sin() + (bt + gt + t0 + t1 + t2 + 2_f64*t3).sin() - 2_f64*(-bt - gt + t1 + t2 + 2_f64*t3).cos() + 2_f64*(-bt + gt + t1 + t2 + 2_f64*t3).cos() - 2_f64*(bt - gt + t1 + t2 + 2_f64*t3).cos() + 2_f64*(bt + gt + t1 + t2 + 2_f64*t3).cos()) - (l2*t2.sin() - o2*t2.cos())*((bt.sin()*t0.cos() + t0.sin()*bt.cos()*gt.cos())*t1.sin() + gt.sin()*bt.cos()*t1.cos()) + (l2*t2.cos() + o2*t2.sin())*((bt.sin()*t0.cos() + t0.sin()*bt.cos()*gt.cos())*t1.cos() - gt.sin()*t1.sin()*bt.cos()), -l3*(bt.sin()*(t1 + t2).sin()*t0.cos() + gt.sin()*bt.cos()*(t1 + t2).cos() + t0.sin()*(t1 + t2).sin()*bt.cos()*gt.cos())*t3.sin() - l3*(bt.sin()*(t1 + t2 + t3).sin()*t0.cos() + gt.sin()*bt.cos()*(t1 + t2 + t3).cos() + t0.sin()*(t1 + t2 + t3).sin()*bt.cos()*gt.cos())*t3.sin() + l3*(bt.sin()*t0.cos()*(t1 + t2).cos() - gt.sin()*(t1 + t2).sin()*bt.cos() + t0.sin()*bt.cos()*gt.cos()*(t1 + t2).cos())*t3.cos() + l3*(bt.sin()*t0.cos()*(t1 + t2 + t3).cos() - gt.sin()*(t1 + t2 + t3).sin()*bt.cos() + t0.sin()*bt.cos()*gt.cos()*(t1 + t2 + t3).cos())*t3.cos() + (1_f64/8.0)*l4*(-2_f64*(-bt - t0 + t1 + t2 + 2_f64*t3).sin() - 2_f64*(-bt + t0 + t1 + t2 + 2_f64*t3).sin() + 2_f64*(bt - t0 + t1 + t2 + 2_f64*t3).sin() + 2_f64*(bt + t0 + t1 + t2 + 2_f64*t3).sin() + (-bt - gt + t0 + t1 + t2 + 2_f64*t3).sin() - (-bt + gt - t0 + t1 + t2 + 2_f64*t3).sin() + (-bt + gt + t0 + t1 + t2 + 2_f64*t3).sin() - (bt - gt - t0 + t1 + t2 + 2_f64*t3).sin() + (bt - gt + t0 + t1 + t2 + 2_f64*t3).sin() - (bt + gt - t0 + t1 + t2 + 2_f64*t3).sin() + (bt + gt + t0 - t1 - t2 - 2_f64*t3).sin() + (bt + gt + t0 + t1 + t2 + 2_f64*t3).sin() - 2_f64*(-bt - gt + t1 + t2 + 2_f64*t3).cos() + 2_f64*(-bt + gt + t1 + t2 + 2_f64*t3).cos() - 2_f64*(bt - gt + t1 + t2 + 2_f64*t3).cos() + 2_f64*(bt + gt + t1 + t2 + 2_f64*t3).cos()) - (l2*t2.sin() - o2*t2.cos())*((bt.sin()*t0.cos() + t0.sin()*bt.cos()*gt.cos())*t1.sin() + gt.sin()*bt.cos()*t1.cos()) + (l2*t2.cos() + o2*t2.sin())*((bt.sin()*t0.cos() + t0.sin()*bt.cos()*gt.cos())*t1.cos() - gt.sin()*t1.sin()*bt.cos()), -l3*(bt.sin()*(t1 + t2).sin()*t0.cos() + gt.sin()*bt.cos()*(t1 + t2).cos() + t0.sin()*(t1 + t2).sin()*bt.cos()*gt.cos())*t3.sin() - 2_f64*l3*(bt.sin()*(t1 + t2 + t3).sin()*t0.cos() + gt.sin()*bt.cos()*(t1 + t2 + t3).cos() + t0.sin()*(t1 + t2 + t3).sin()*bt.cos()*gt.cos())*t3.sin() + l3*(bt.sin()*t0.cos()*(t1 + t2).cos() - gt.sin()*(t1 + t2).sin()*bt.cos() + t0.sin()*bt.cos()*gt.cos()*(t1 + t2).cos())*t3.cos() + 2_f64*l3*(bt.sin()*t0.cos()*(t1 + t2 + t3).cos() - gt.sin()*(t1 + t2 + t3).sin()*bt.cos() + t0.sin()*bt.cos()*gt.cos()*(t1 + t2 + t3).cos())*t3.cos() + (1_f64/4.0)*l4*(-2_f64*(-bt - t0 + t1 + t2 + 2_f64*t3).sin() - 2_f64*(-bt + t0 + t1 + t2 + 2_f64*t3).sin() + 2_f64*(bt - t0 + t1 + t2 + 2_f64*t3).sin() + 2_f64*(bt + t0 + t1 + t2 + 2_f64*t3).sin() + (-bt - gt + t0 + t1 + t2 + 2_f64*t3).sin() - (-bt + gt - t0 + t1 + t2 + 2_f64*t3).sin() + (-bt + gt + t0 + t1 + t2 + 2_f64*t3).sin() - (bt - gt - t0 + t1 + t2 + 2_f64*t3).sin() + (bt - gt + t0 + t1 + t2 + 2_f64*t3).sin() - (bt + gt - t0 + t1 + t2 + 2_f64*t3).sin() + (bt + gt + t0 - t1 - t2 - 2_f64*t3).sin() + (bt + gt + t0 + t1 + t2 + 2_f64*t3).sin() - 2_f64*(-bt - gt + t1 + t2 + 2_f64*t3).cos() + 2_f64*(-bt + gt + t1 + t2 + 2_f64*t3).cos() - 2_f64*(bt - gt + t1 + t2 + 2_f64*t3).cos() + 2_f64*(bt + gt + t1 + t2 + 2_f64*t3).cos()), 0.0
        );

        let jacobian_inverse = match jacobian.pseudo_inverse(pseudo_inverse_epsilon) {
            Ok(mat) => mat,
            Err(error) => return Err(Error::PseudoInverse(error)),
        };

        let delta_angles = jacobian_inverse * delta_position;

        Ok(delta_angles)
    }

    /// Performs inverse kinematics to reach the given target position with the given parameters.
    fn s_ik_arm_ef_position_with_eps(
        torso_position: &nalgebra::Vector3<f64>,
        torso_orientation: &nalgebra::Vector3<f64>,
        arm_thetas: &nalgebra::Vector5<f64>,
        arm_lengths: &nalgebra::Vector6<f64>,
        arm_offsets: &nalgebra::Vector1<f64>,
        target_position: &nalgebra::Vector3<f64>,
        epsilon: Option<f64>,
        pseudo_inverse_epsilon: f64,
    ) -> Result<nalgebra::Vector5<f64>, Error> {
        let mut error: f64 = 0.0;

        // Initializes the end effector position and our current distance from it.
        let mut current_end_effector_position: nalgebra::Vector3<f64> = Self::s_fk_arm_ef_position(
            torso_position,
            torso_orientation,
            arm_thetas,
            arm_lengths,
            arm_offsets,
        )?;
        let mut currrent_distance_from_end_effector: nalgebra::Vector3<f64> =
            target_position - current_end_effector_position;

        // Clones the angles so we can work without modifying them.
        let mut cloned_arm_thetas: nalgebra::Vector5<f64> = arm_thetas.clone();

        // Performs up to thirty optimization iterations, and if we still haven't reached
        //  the desired error, return and void all the found angles, target cannot be reached.
        for _ in 1..6000 {
            // Computes the delta angles.
            let cloned_arm_delta_thetas: nalgebra::Vector5<f64> = Self::s_ik_arm_ef_position(
                torso_orientation,
                &cloned_arm_thetas,
                arm_lengths,
                arm_offsets,
                &currrent_distance_from_end_effector,
                pseudo_inverse_epsilon,
            )?;

            // Updates the current angles with the computed deltas.
            cloned_arm_thetas += cloned_arm_delta_thetas;

            // If there is no error threshold, just return, the callee is fine with whatever
            // accuracy we bring up.
            if epsilon.is_none() {
                return Ok(cloned_arm_thetas);
            }

            // Computes the current end effector position and then the distance we are from it,
            // based on that we then compute the error as the magnitude of the distance (scalar).
            current_end_effector_position = Self::s_fk_arm_ef_position(
                torso_position,
                torso_orientation,
                &cloned_arm_thetas,
                arm_lengths,
                arm_offsets,
            )?;
            currrent_distance_from_end_effector = target_position - current_end_effector_position;
            error = currrent_distance_from_end_effector.magnitude();

            // If we're lower than the specified error threshold (epsilon) return the found angles.
            if error < epsilon.as_ref().unwrap().clone() {
                return Ok(cloned_arm_thetas);
            }
        }

        // Returns the error indicating we could not reach the given target.
        Err(Error::UnreachableTargetPosition(
            error,
            target_position.clone(),
        ))
    }

    /// Performs inverse kinematics on the arm to try and reach the given target position within the
    /// given error threshold epsilon, None if we just want to do one iteration.
    #[allow(unused)]
    pub fn ik_arm_ef_position_with_eps(
        &self,
        target_position: &nalgebra::Vector3<f64>,
        epsilon: Option<f64>,
    ) -> Result<nalgebra::Vector5<f64>, Error> {
        Self::s_ik_arm_ef_position_with_eps(
            self.torso.position(),
            self.torso.orientation(),
            self.arm.thetas(),
            self.arm.lengths(),
            self.arm.offsets(),
            target_position,
            epsilon,
            self.pseudo_inverse_epsilon,
        )
    }
}

pub struct SolverBuilder {
    torso: Torso,
    legs: [Leg; 4],
    arm: Arm,
    pseudo_inverse_epsilon: f64,
}

impl SolverBuilder {
    /// Creates a new solver builder.
    #[inline(always)]
    pub fn new(torso: Torso, legs: [Leg; 4], arm: Arm) -> Self {
        Self {
            torso,
            legs,
            arm,
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
        Solver::new(self.torso, self.pseudo_inverse_epsilon, self.legs, self.arm)
    }
}
