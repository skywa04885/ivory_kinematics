#[derive(Debug)]
pub enum Error {
    LegNumberOutOfBounds(u8),
    MatrixInversionError,
}

pub struct Leg {
    theta_0: f64,
    theta_1: f64,
    theta_2: f64,
    l_0: f64,
    l_1: f64,
    l_2: f64,
}

pub struct Torso {
    alpha_t: f64,   // Torso orientation euler angle: yaw.
    beta_t: f64,    // Torso orientation euler angle: pitch.
    gamma_t: f64,   // Torso orientation euler angle: roll.
    x_t: f64,       // Torso position: x.
    y_t: f64,       // Torso position: y.
    z_t: f64,       // Torso position: z.
    w_t: f64,       // Torso dimension: width.
    h_t: f64,       // Torso dimension: height.
    legs: [Leg; 4], // Legs.
}

impl Torso {
    pub fn new(
        alpha_t: f64,
        beta_t: f64,
        gamma_t: f64,
        x_t: f64,
        y_t: f64,
        z_t: f64,
        w_t: f64,
        h_t: f64,
        legs: [Leg; 4],
    ) -> Self {
        Self {
            alpha_t,
            beta_t,
            gamma_t,
            x_t,
            y_t,
            z_t,
            w_t,
            h_t,
            legs,
        }
    }
    pub fn builder() -> TorsoBuilder {
        TorsoBuilder::new()
    }

    pub fn s_x(l: u8) -> f64 {
        match l {
            0 | 1 => -1.0,
            2 | 3 => 1.0,
            _ => panic!("Invalid leg."),
        }
    }

    pub fn s_z(l: u8) -> f64 {
        match l {
            0 | 2 => -1.0,
            1 | 3 => 1.0,
            _ => panic!("Invalid leg."),
        }
    }

    #[inline(always)]
    pub fn s_xz(l: u8) -> (f64, f64) {
        (Self::s_x(l), Self::s_z(l))
    }

    #[allow(unused_variables)]
    pub fn ik_paw_ef_position(
        &self,
        l: u8,
        delta_position: nalgebra::Vector3<f64>,
    ) -> Result<nalgebra::Vector3<f64>, Error> {
        if l > 3 {
            return Err(Error::LegNumberOutOfBounds(l));
        }

        let (s_x, s_z): (f64, f64) = Self::s_xz(l);

        let (alpha_t, beta_t, gamma_t) = (self.alpha_t, self.beta_t, self.gamma_t);
        let (x_t, y_t, z_t) = (self.x_t, self.y_t, self.z_t);
        let (w_t, h_t) = (self.w_t, self.h_t);

        let leg: &Leg = &self.legs[l as usize];
        let (l_0, l_1, l_2) = (leg.l_0, leg.l_1, leg.l_2);
        let (theta_0, theta_1, theta_2) = (leg.theta_0, leg.theta_1, leg.theta_2);

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

        println!("{}", jacobian);

        let jacobian_inverse = jacobian.pseudo_inverse(0.01).unwrap();

        let delta_angles = jacobian_inverse * delta_position;

        Ok(delta_angles)
    }

    #[allow(unused_variables)]
    pub fn fk_paw_ef_position(&self, l: u8) -> Result<nalgebra::Vector3<f64>, Error> {
        if l > 3{
            return Err(Error::LegNumberOutOfBounds(l));
        }

        let (s_x, s_z): (f64, f64) = Self::s_xz(l);

        let (alpha_t, beta_t, gamma_t) = (self.alpha_t, self.beta_t, self.gamma_t);
        let (x_t, y_t, z_t) = (self.x_t, self.y_t, self.z_t);
        let (w_t, h_t) = (self.w_t, self.h_t);

        let leg: &Leg = &self.legs[l as usize];
        let (l_0, l_1, l_2) = (leg.l_0, leg.l_1, leg.l_2);
        let (theta_0, theta_1, theta_2) = (leg.theta_0, leg.theta_1, leg.theta_2);

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
}

pub struct TorsoBuilder {
    alpha_t: f64,   // Torso orientation euler angle: yaw.
    beta_t: f64,    // Torso orientation euler angle: pitch.
    gamma_t: f64,   // Torso orientation euler angle: roll.
    x_t: f64,       // Torso position: x.
    y_t: f64,       // Torso position: y.
    z_t: f64,       // Torso position: z.
    w_t: f64,       // Torso dimension: width.
    h_t: f64,       // Torso dimension: height.
    legs: [Leg; 4], // Legs.
}

impl TorsoBuilder {
    pub fn new() -> Self {
        let (l_0, l_1, l_2) = (10.0, 10.0, 10.0);
        let (theta_0, theta_1, theta_2) = (f64::to_radians(45.0), f64::to_radians(45.0), 0.0);

        Self {
            alpha_t: 0.0,
            beta_t: 0.0,
            gamma_t: 0.0,
            x_t: 0.0,
            y_t: 0.0,
            z_t: 0.0,
            w_t: 50.0,
            h_t: 90.0,
            legs: [
                Leg {
                    l_0,
                    l_1,
                    l_2,
                    theta_0,
                    theta_1,
                    theta_2,
                },
                Leg {
                    l_0,
                    l_1,
                    l_2,
                    theta_0,
                    theta_1,
                    theta_2,
                },
                Leg {
                    l_0,
                    l_1,
                    l_2,
                    theta_0,
                    theta_1,
                    theta_2,
                },
                Leg {
                    l_0,
                    l_1,
                    l_2,
                    theta_0,
                    theta_1,
                    theta_2,
                },
            ],
        }
    }

    pub fn build(self) -> Torso {
        Torso::new(
            self.alpha_t,
            self.beta_t,
            self.gamma_t,
            self.x_t,
            self.y_t,
            self.z_t,
            self.w_t,
            self.h_t,
            self.legs,
        )
    }
}
