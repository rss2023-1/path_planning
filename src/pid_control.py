import rospy
class PIDController:
    def __init__(self, setpoint):
        self.Kp = 0.1
        self.Ki = 0.1
        self.Kd = 0.1
        self.last_time = rospy.get_time()
        self.last_error = None
        self.setpoint = setpoint
        self.proportional = 0
        self.inverse = 0
        self.derivative = 0
    
    def call(self, theta):
        now = rospy.get_time()
        dt = now - self.last_time if now - self.last_time > 0 else 1e-16

        error = theta - self.setpoint
        d_error = error - self.last_error if self.last_error is not None else error

        self.proportional = self.Kp * error
        self.inverse = self.Ki * error * dt
        self.derivative = self.Kd * d_error/dt

        return self.proportional + self.inverse + self.derivative