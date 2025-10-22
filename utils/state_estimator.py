import numpy as np


class MahonyIMU:
    def __init__(self):
        # Use MAHONY Quaternion IMU
        self.two_kp = 2.0 * 0.4
        self.two_ki = 2.0 * 0.001

        self.integralFBx = 0.0
        self.integralFBy = 0.0
        self.integralFBz = 0.0

        self.qw = 1.0
        self.qx = 0.0
        self.qy = 0.0
        self.qz = 0.0

    def sensfusion6Update(self, gx, gy, gz, ax, ay, az, dt):
        gx = gx * np.pi / 180
        gy = gy * np.pi / 180
        gz = gz * np.pi / 180

        if (ax != 0.0) and (ay != 0) and (az != 0):
            recip_norm = 1 / np.sqrt(ax * ax + ay * ay + az * az)
            ax *= recip_norm
            ay *= recip_norm
            az *= recip_norm

            halfvx = self.qx * self.qz - self.qw * self.qy
            halfvy = self.qw * self.qx + self.qy * self.qz
            halfvz = self.qw * self.qw - 0.5 + self.qz * self.qz

            # Error is sum of cross product between estimated and measured direction of gravity
            halfex = ay * halfvz - az * halfvy
            halfey = az * halfvx - ax * halfvz
            halfez = ax * halfvy - ay * halfvx

            if self.two_ki > 0:
                self.integralFBx += (
                    self.two_ki * halfex * dt
                )  # integral error scaled by Ki
                self.integralFBy += self.two_ki * halfey * dt
                self.integralFBz += self.two_ki * halfez * dt
                gx += self.integralFBx  # apply integral feedback
                gy += self.integralFBy
                gz += self.integralFBz

            else:
                self.integralFBx = 0.0
                self.integralFBy = 0.0
                self.integralFBz = 0.0

            # Apply proportional feedback
            gx += self.two_kp * halfex
            gy += self.two_kp * halfey
            gz += self.two_kp * halfez

        # Integrate rate of change of quaternion
        gx *= 0.5 * dt  #  pre-multiply common factors
        gy *= 0.5 * dt
        gz *= 0.5 * dt
        qa = self.qw
        qb = self.qx
        qc = self.qy
        self.qw += -qb * gx - qc * gy - self.qz * gz
        self.qx += qa * gx + qc * gz - self.qz * gy
        self.qy += qa * gy - qb * gz + self.qz * gx
        self.qz += qa * gz + qb * gy - qc * gx

        # Normalise quaternion
        recipNorm = 1 / np.sqrt(
            self.qw * self.qw
            + self.qx * self.qx
            + self.qy * self.qy
            + self.qz * self.qz
        )
        self.qw *= recipNorm
        self.qx *= recipNorm
        self.qy *= recipNorm
        self.qz *= recipNorm

        return self.qx, self.qy, self.qz, self.qw
