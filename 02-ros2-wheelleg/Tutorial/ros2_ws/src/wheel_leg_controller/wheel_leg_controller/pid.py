import numpy as np


class PIDController:
    def __init__(self, kp, ki, kd, max_out, max_i, feedforward=0.0):
        self.kp = kp; self.ki = ki; self.kd = kd; self.max_out = max_out; self.max_i = max_i; self.feedforward = feedforward
        self.integral = 0.0; self.prev_error = 0.0

    def compute(self, current, target):
        error = target - current; self.integral += error
        self.integral = np.clip(self.integral, -self.max_i, self.max_i)
        derivative = (error - self.prev_error); self.prev_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative + self.feedforward
        return np.clip(output, -self.max_out, self.max_out)
