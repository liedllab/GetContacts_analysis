from dataclasses import dataclass
import numpy as np

@dataclass
class Cartesian:
    x: float
    y: float

    def __post_init__(self):
        self._values = np.array((self.x, self.y))
        
    @property
    def values(self):
        return self._values
    
    def convert(self):
        r = np.sqrt(self.x**2+self.y**2)
        theta = np.arctan2(self.x,self.y)
        
        return Polar(theta=theta, r=r)

@dataclass
class Polar:
    theta: float
    r: float
        
    def __post_init__(self):
        self.theta %= np.pi*2
        self._values = np.array((self.theta, self.r))
        
    @property
    def values(self):
        return self._values
    
    def convert(self):
        x = self.r*np.sin(self.theta)
        y = self.r*np.cos(self.theta)
        return Cartesian(x,y)

    def in_degrees(self):
        return self.theta*180/np.pi, self.r

def bezier(a, b, zp: Cartesian = Cartesian(0,0)):
    if isinstance(a, Polar):
        a = a.convert()
    if isinstance(b, Polar):
        b = b.convert()
    if isinstance(zp, Polar):
        zp = zp.convert()
    if not all(map(lambda x: isinstance(x, Cartesian), (a,b,zp))):
        raise ValueError("Arguments need to be a Coordinate-Object")
        
    ts = np.linspace(0,1)
    
    bezier_curve = [Cartesian(
        *((1-t) * ( (1-t) * a.values + t * zp.values ) + t * ( (1-t) * zp.values + (t) * b.values ))) 
            for t in ts]

    return bezier_curve

if __name__ == "__main__":
    exit()

