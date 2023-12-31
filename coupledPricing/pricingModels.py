import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow.signal import ifft
from scipy.interpolate import interp1d

#Merton model
#############################################
class MertonJumpModel:
    def __init__(self, T, N, r, muJ, sigmaJ, sigma, lam, K, x0, func, limit):
        self.T = T                      #Maturity
        self.r = r                      #Interest rate
        self.sig = sigma                #sigma BM asset
        self.muJ = muJ                  #jump mean
        self.sigJ = sigmaJ              #jump std
        self.lam = lam                  #jump intensity
        self.K = K                      #srike
        self.N = N                      #steps number
        self.dt = T/N                   #steps length
        self.x0 = x0                    #current price asset
        self.func = func                #functor
        self.dist = tfd.Normal(loc=0., scale=1.)
        self.limit = limit              #limit Power Series

    #initialize
    def init(self, batchSize):
        self.batchSize = batchSize
        return self.x0 *tf.ones([batchSize])

    #Analytic approach
    #BS closed formula
    def BS(self,iStep ,X , rbs, sigbs):
        shape = rbs.shape[0]
        d1 = (tf.math.log(X/self.K) + (rbs + sigbs**2/2)*(self.T- iStep*self.dt))/(sigbs*tf.sqrt(self.T- iStep*self.dt))
        d2 = (tf.math.log(X/self.K) + (rbs - sigbs**2/2)*(self.T- iStep*self.dt))/(sigbs*tf.sqrt(self.T- iStep*self.dt))
        return X*self.dist.cdf(d1) - self.K*tf.exp(-rbs*(self.T- iStep*self.dt))*self.dist.cdf(d2)

    #Merton closed formula
    def A(self, iStep, X):
        if iStep < self.N :
          I = tf.range(self.limit, dtype = tf.float32)
          rBS = tf.tile(tf.expand_dims(self.r - self.lam*(tf.exp(self.muJ  + self.sigJ*self.sigJ*0.5) - 1) + I*(self.muJ + 0.5*self.sigJ*self.sigJ)/(self.T - iStep*self.dt), axis=0),[tf.shape(X)[0],1])
          sigBS = tf.tile(tf.expand_dims(tf.sqrt(self.sig**2 + I*(self.sigJ**2)/(self.T - iStep*self.dt)), axis=0),[tf.shape(X)[0],1])
          BSincrements = self.BS(iStep,tf.tile(tf.expand_dims(X, axis=-1),[1,self.limit]) , rBS, sigBS)
          lam2 = self.lam*tf.exp(self.muJ + 0.5*self.sigJ**2)
          coefficients = tf.tile(tf.expand_dims(tf.exp(-lam2*(self.T - iStep*self.dt))*((lam2*(self.T - iStep*self.dt))**I)/tf.exp(tf.math.lgamma(I + 1)), axis=0),[tf.shape(X)[0],1])
          return tf.reduce_sum(coefficients*BSincrements, axis = 1)
        return self.g(X)


    #Go to next step
    def oneStepFrom(self, iStep, X, dW, gaussJ, Y):
        return  X*tf.exp((self.r - 0.5*self.sig*self.sig - self.lam*(tf.exp(self.muJ  + self.sigJ*self.sigJ*0.5) - 1))*self.dt + self.sig*dW + gaussJ) + self.func(Y - self.A(iStep,X))*self.dt

    #sum of jumps
    def jumps(self, batchSize):
        lam = self.lam*tf.ones([batchSize])
        dN = tf.random.poisson([1], lam*self.dt, dtype = tf.float32)[0]
        gaussJ = dN*self.muJ + self.sigJ*tf.sqrt(dN)*tf.random.normal([batchSize], 0, 1)
        return gaussJ

    #Driver
    def f(self, Y):
        return -self.r*Y

    #Payoff
    def g(self, X):
        return tf.maximum(X-self.K, 0)

#VG Model with inverse Fourier method
####################################################
class VGmodelinvfourier:
    def __init__(self, T, N, r, theta, kappa, sigmaJ, K, x0, func):
        self.T = T                                                                         #Maturity
        self.r = r                                                                         #Interest rate
        self.sigJ = sigmaJ                                                                 #volatility of small jumps
        self.theta = theta                                                                 #the drift of the BM approximating small jumps
        self.kappa = kappa                                                                 #variance of the Gamma process
        self.K = K                                                                         #srike
        self.N = N                                                                         #steps number
        self.dt = T/N                                                                      #steps length
        self.x0 = x0                                                                       #spot price
        self.correction = -tf.math.log(1 - theta * kappa - kappa/2 * sigmaJ**2 ) /kappa    #correction of the drift of the jump part
        self.func = func                                                                   #couplage functor

    

    #initialize
    def init(self, batchSize):                                                                   
        self.batchSize = batchSize
        return self.x0*tf.ones([batchSize])

    #fourier inversion method
    def characteristicfunc(self, iStep, u):
      return tf.exp((self.T - iStep*self.dt)*(tf.dtypes.complex(0., (self.r - self.correction))*u - \
                                                   tf.math.log(1 - tf.dtypes.complex(0., self.theta*self.kappa)*u + tf.dtypes.complex(0.5*self.kappa*self.sigJ*self.sigJ, 0.)*u*u)/self.kappa))
    
    def A(self, iStep, X):
      k = tf.tile(tf.expand_dims(tf.math.log(self.K/X), axis = 0), [10**3,1])
      u  = tf.tile(tf.expand_dims(tf.linspace(1e-15, 5000, 10**3), axis = 1), [1,tf.shape(X)[0]])
      integrand1 = lambda u: tf.math.real((tf.exp(-tf.dtypes.complex(0., u*k))/ (tf.dtypes.complex(0., u))) * \
                                          self.characteristicfunc(iStep, tf.dtypes.complex(u, -1.))/ self.characteristicfunc(iStep, tf.dtypes.complex(0.,-1.0000000000001)))
      integrand2 = lambda u: tf.math.real(tf.exp(-tf.dtypes.complex(0., u*k))/ (tf.dtypes.complex(0., u))*self.characteristicfunc(iStep, tf.dtypes.complex(u, 0.)))      
      Q1 = 0.5 + 1/np.pi * tfp.math.trapz(integrand1(u), u, axis = 0) 
      Q2 = 0.5 + 1/np.pi * tfp.math.trapz(integrand2(u), u, axis = 0)
      return X*Q1 - self.K*tf.exp(-self.r*(self.T - iStep*self.dt))*Q2


    #Go to next step
    def oneStepFrom(self, iStep, X, gaussJ, Y):
        return  X*tf.exp((self.r - self.correction)*self.dt + gaussJ) + self.func(Y - self.A(iStep, X))*self.dt 
    
    #jumps
    def jumps(self):
        gauss = tf.random.normal([self.batchSize], 0, 1)
        gamma = tf.random.gamma([self.batchSize], self.dt/self.kappa, 1/self.kappa) 
        return tf.math.multiply(self.theta, gamma) + self.sigJ*tf.sqrt(gamma)*gauss

    #Driver
    def f(self, Y):
        return -self.r*Y

    #Payoff 
    def g(self, X):
        return tf.maximum(X-self.K, 0)

######################################################
#VG with FFT method
class VGmodel:
    def __init__(self, T, N, r, theta, kappa, sigmaJ, K, x0, func):
        self.T = T                                                                         #Maturity
        self.r = r                                                                         #Interest rate
        self.sigJ = sigmaJ                                                                 #volatility of small jumps
        self.theta = theta                                                                 #the drift of the BM approximating small jumps
        self.kappa = kappa                                                                 #variance of the Gamma process
        self.K = K                                                                         #srike
        self.N = N                                                                         #steps number
        self.dt = T/N                                                                      #steps length
        self.x0 = x0                                                                       #spot price
        self.correction = -tf.math.log(1 - theta * kappa - kappa/2 * sigmaJ**2 ) /kappa    #correction of the drift of the jump part
        self.func = func                                                                   #couplage functor



    #initialize
    def init(self, batchSize):
        self.batchSize = batchSize
        return self.x0*tf.ones([batchSize])

    #fft method
    def characteristicfunc(self, iStep, u):
      return tf.exp((self.T - iStep*self.dt)*(tf.dtypes.complex(0., (self.r - self.correction))*u - \
                                                   tf.math.log(1 - tf.dtypes.complex(0., self.theta*self.kappa)*u + tf.dtypes.complex(0.5*self.kappa*self.sigJ*self.sigJ, 0.)*u*u)/self.kappa))

    def A(self, iStep, X):
      # fft parameters
      fftN, B = 2**15, 500
      du = B/fftN
      rng = np.arange(fftN)
      u = rng*du
      lm = 2*np.pi/B
      b = fftN*lm/2
      ku = -b + lm*rng
      # weights
      weight = 3 + (-1)**(rng+1)
      weight[0], weight[fftN - 1] = 1, 1
      # integrands
      integrand = tf.exp(tf.dtypes.complex(0., tf.cast(-b*rng*du, dtype = tf.float32)))*self.characteristicfunc(iStep, tf.dtypes.complex(tf.cast(u, dtype = tf.float32), -0.5))*1/(u**2 + 0.25)*weight*du/3
      integral = tf.math.real(ifft(integrand)*fftN)
      # Compute the log of the interpolation point
      log_interp_point = tf.math.log(X/self.K)
      # Define the wrapped function
      def wrapped_interp1d(ku, integral, log_interp_point):
          spline = interp1d(ku, integral, kind='cubic')
          return tf.cast(spline(log_interp_point), dtype = tf.float32)
      # Use tf.numpy_function to call the wrapped function
      spline = tf.cast(tf.numpy_function(wrapped_interp1d, [ku, integral, log_interp_point], tf.float32), dtype = tf.float32)
      return X - tf.math.sqrt(X*self.K)*tf.exp(-self.r*(self.T - iStep*self.dt))/np.pi*spline



    #Go to next step
    def oneStepFrom(self, iStep, X, gaussJ, Y):
        return  X*tf.exp((self.r - self.correction)*self.dt + gaussJ)  + self.func(Y - self.A(iStep,X))*self.dt

    #jumps
    def jumps(self, batchSize):
        gauss = tf.random.normal([batchSize], 0, 1)
        gamma = tf.random.gamma([batchSize], self.dt/self.kappa, 1/self.kappa)
        return tf.math.multiply(self.theta, gamma) + self.sigJ*tf.sqrt(gamma)*gauss

    #Driver
    def f(self, Y):
        return -self.r*Y

    #Payoff
    def g(self, X):
        return tf.maximum(X-self.K, 0)
