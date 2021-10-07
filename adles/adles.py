from datetime import time
from warnings import filters
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint
from scikits.odes.dae import dae

from adles.filters.gfm_iaif import gfmiaif
from adles.utilites import binary_search

count = 0
t_count = 0
w_count = 0

class UserData:
    """
        Encapsulates user data to send to the RHS function
    """
    def __init__(self, residual, c, d, beta, delta, alpha,
                 right_disp, left_disp, right_vel, left_vel,
                 last_time, window, window_size):
        self.residual = residual
        self.c = c 
        self.d = d
        self.beta = beta
        self.delta = delta
        self.alpha = alpha
        self.right_disp = right_disp
        self.left_disp = left_disp
        self.right_vel = right_vel
        self.left_vel = left_vel
        self.t = last_time 
        self.w = window 
        self.window_size = window_size



def _coupled_vanderpol(z, t, delta, alpha, beta): 
    """ ODE Solver RHS

        RHS equation for computing vocal fold displacement and velocity parameters.
        Args:
            z       : initial input
            t       : timestamps
            delta   : asymmetric coefficient
            beta    : vocal fold rigidity
            alpha   : 
    """
    left_disp, left_vel, right_disp, right_vel = z
    acc = [left_vel,
           (delta*left_disp)/2-alpha*(left_vel+right_vel)-left_disp-beta*(1+np.square(left_disp))*left_vel,
           right_vel,
           (delta*right_disp)/2-alpha*(right_vel + left_vel)-right_disp-beta*(1+np.square(right_disp))*right_vel]

    return acc



def _adjoint_lagrangian(t, x, xdot, result, user_data):
    """
        RHS equation for computing lagrangian parameters.
        Args:
            t   : time step
            x   : input non-differential variables
            xdot : input first order differential variables 
            result : stores the results
    """
    global t_count, w_count
    win_count = w_count
    t_time = user_data.t[t_count]
    if t in user_data.w:
        time = t
    else:
        prev_val = user_data.w[t_time][win_count-1]

        if t <=prev_val:
            win_count -= max(win_count-1, 0)
            prev_val = user_data.w[t_time][win_count-1]
        
        curr_val = user_data.w[t_time][win_count]

        if t >= (curr_val-prev_val)/2:
            time = curr_val
        else:
            time = prev_val
        
    residual, c, d, beta, delta, alpha = user_data.residual[t_time][time], user_data.c, user_data.d, user_data.beta, user_data.delta, user_data.alpha
    right_disp, left_disp, right_vel, left_vel = user_data.right_disp[t_time][time], user_data.left_disp[t_time][time], user_data.right_vel[t_time][time], user_data.left_vel[t_time][time]

    result[0] = -2*c*d*residual -1*(2* beta *right_disp*(right_vel + 1 - delta/2))*x[0] - xdot[2]
    result[1] = -2*c*d*residual -1*(2* beta *left_disp*(left_vel + 1 - delta/2))*x[1] - xdot[3]
    result[2] = (alpha * x[1])/(beta*(1+np.square(right_disp))-alpha) - x[0]
    result[3] = (alpha * x[0])/(beta*(1+np.square(left_disp))-alpha) - x[1]

    w_count = win_count


class ADLES():
    def __init__(self, signal, sr, window_size=30, frame_shift=1,
                init_left_distend=0.1, init_right_distend=0.15):

        # Define ADLES constants
        self.air_velocity = 5e-5
        self.vocal_fold_length = 1.75
        self.half_glottal_width = 0.1
        self.alpha, self.beta, self.delta = 0.5, 0.32, 0.0

        self.window_size = window_size
        self.frame_shift = frame_shift
        self.sampling_rate = sr
        self.t = np.linspace(0, len(signal)/self.sampling_rate, len(signal))

        # Create framed window parameters
        self.right_distend, self.left_distend = [], []
        self.right_velocity, self.left_velocity = [], []
        self.lagrange_lambda, self.lagrange_lambda_dot = [], []
        self.largrange_eta, self.lagrange_eta_dot = [], []

        # Frame the signal into windows and initialize framed window parameters
        left = 0
        self.windows = []
        self.signal_window = []
        self.filters = []
        while (left + window_size) < len(self.t):
            self.windows.append(self.t[left:left+window_size])
            self.left_distend.append(np.array([init_left_distend]*window_size))
            self.right_distend.append(np.array([init_right_distend]*window_size))
            self.left_velocity.append(np.array([0.0]*window_size))
            self.right_velocity.append(np.array([0.0]*window_size))
            self.signal_window.append(signal[left:left+window_size])
            left += frame_shift

    def integrate(self):
        y0 = [self.left_distend[0][0], self.left_velocity[0][0], self.right_distend[0][0], self.right_velocity[0][0]]

        for i in range(len(self.windows)):
            sol = odeint(_coupled_vanderpol, y0, self.windows[i], args=(self.delta, self.alpha, self.beta), hmax=1/self.sampling_rate)

            self.left_distend[i], self.left_velocity[i] = sol[:, 0], sol[:, 1]
            self.right_distend[i], self.right_velocity[i] = sol[:, 2], sol[:, 3]

            if self.window_size != self.left_distend[i].shape[0]:
                print("Shape of left distend does not match time stamps")
            if self.window_size != self.right_distend[i].shape[0]:
                print("Shape of right distend does not match time stamps")
            if self.window_size != self.left_velocity[i].shape[0]:
                print("Shape of left velocity does not match time stamps")
            if self.window_size != self.right_velocity[i].shape[0]:
                print("Shape of right velocity does not match time stamps")

            y0 = [self.left_distend[i][-1], self.left_velocity[i][-1], self.right_distend[i][-1], self.right_velocity[i][-1]]

    def compute_residual(self):
        residuals = []
        for i in range(len(self.windows)):
            glottal_estimate = self.air_velocity*self.vocal_fold_length*(2*self.half_glottal_width + self.left_distend[i] + self.right_distend[i])

            filter = np.zeros(1) if i >= len(self.filters) else self.filters[i]
            inverse_glottal_estimate = gfmiaif(self.signal_window[i], self.sampling_rate, p_vt=48, p_gl=3,
                                                d=0.99, hpfilt=1, hpfilt_in=filter)

            g, _, _ , _, filt = inverse_glottal_estimate
            residual = glottal_estimate - g

            residuals.append(residual)
            self.filters.append(filt)

        return residuals

    def aggregate_variables(self, *args):
        aggr_list = []
        for a in args:
            curr_arg = {}
            for idx, w in enumerate(self.windows):
                curr_win = {}
                for jdx, t in enumerate(w):
                    curr_win[t] = a[idx][jdx]
                curr_arg[w[-1]] = curr_win

            aggr_list.append(curr_arg)

        return tuple(aggr_list)

            
    def solve(self):
        global t_count
        global w_count
        SOLVER = 'ida'
        self.residual = self.compute_residual()
        init_val = -2.0 * self.air_velocity*self.vocal_fold_length*self.residual[-1][-1]
        y0 = [0.0,0.0,0.0,0.0]
        yp0 = [0.0,0.0,init_val,init_val]
        
        
        residuals, left_distend, right_distend, right_velocity, left_velocity, time_wins = self.aggregate_variables(self.residual, self.left_distend,
                                                                                self.right_distend, self.right_velocity,
                                                                                self.left_velocity, self.windows)

        time_last_ls = sorted(list(time_wins.keys()))
        corresponding_wins = {}
        for t in time_last_ls:
            corresponding_wins[t] = sorted(list(time_wins[t].keys())) 

        user_data = UserData(residuals, self.air_velocity, self.vocal_fold_length, self.beta,
                                 self.delta, self.alpha, right_distend, left_distend,
                                 right_velocity, left_velocity, time_last_ls, corresponding_wins,
                                 self.window_size)


        t_count = len(self.windows)-1
        for i in range(len(self.windows)-1,-1,-1):
            solver = dae(SOLVER, _adjoint_lagrangian,
                         user_data=user_data, 
                         old_api = True,
                         rtol=1e-3,
                         atol=1e-3,
                         max_steps=5000,
                         max_step_size=1/self.sampling_rate,
                         compute_initcond='yp0',
                         compute_initcond_t0=-self.window_size*(1/self.sampling_rate),
                         algebraic_vars_idx=[2,3])

            w_count = self.window_size-1
            orig_t_count = t_count
            sol = solver.solve(self.windows[i][::-1], y0, yp0)

            y0 = [sol[2][0,0],sol[2][0,1],sol[2][0,2],sol[2][0,3]]
            yp0 = [sol[3][0,0],sol[3][0,1],sol[3][0,2],sol[3][0,3]]

            #print(sol[2][:,2])
            #print(sol[2][:,1])

            self.lagrange_lambda.insert(0,sol[2][:,2])
            self.largrange_eta.insert(0,sol[2][:,3])
            self.lagrange_lambda_dot.insert(0,sol[2][:,0])
            self.lagrange_eta_dot.insert(0,sol[3][:,1])

            t_count = orig_t_count
            t_count -= 1

    def compute_gradients(self):
        self.integrate()
        self.solve()

        f_alpha, f_beta, f_delta = 0.0, 0.0, 0.0
        for idx, w in enumerate(self.windows):
            for jdx, t in enumerate(w):
                #print(f"{idx} {jdx}: ", self.right_velocity[idx][jdx], self.left_velocity[idx][jdx], self.lagrange_lambda[idx][jdx], self.largrange_eta[idx][jdx])
                f_alpha += -1*(self.right_velocity[idx][jdx] + self.left_velocity[idx][jdx])*(self.lagrange_lambda[idx][jdx] + self.largrange_eta[idx][jdx])
                f_beta += ((1 + self.right_distend[idx][jdx]**2)*self.right_velocity[idx][jdx]*self.lagrange_lambda[idx][jdx]) + ((1 + self.left_distend[idx][jdx]**2)*self.left_velocity[idx][jdx]*self.largrange_eta[idx][jdx])
                f_delta += 0.5*(self.right_distend[idx][jdx]*self.largrange_eta[idx][jdx] - self.left_distend[idx][jdx]*self.lagrange_lambda[idx][jdx])

        return f_alpha, f_beta, f_delta

    def train(self, step_size=0.01, n_iters=200, 
              convergence_thresh=1e-5, patience=10,
              verbose=False):

        prev_alpha, prev_beta, prev_delta = self.alpha, self.beta, self.delta

        for k in range(n_iters):
            alpha_patience, beta_pateince, delta_patience = 0, 0, 0

            f_alpha, f_beta, f_delta = self.compute_gradients()

            if alpha_patience < patience:
                self.alpha -= step_size*f_alpha
            if beta_pateince < patience:
                self.beta -= step_size*f_beta
            if delta_patience < patience:
                self.delta -= step_size*f_delta

            if abs(self.alpha-prev_alpha) < convergence_thresh:
                alpha_patience += 1
            else:
                alpha_patience = 0

            if abs(self.beta-prev_beta) < convergence_thresh:
                beta_pateince += 1
            else:
                beta_pateince = 0

            if abs(self.alpha-prev_delta) < convergence_thresh:
                delta_patience += 1
            else: 
                delta_patience = 0


            prev_alpha, prev_beta, prev_delta = self.alpha, self.beta, self.delta

            if alpha_patience>patience or beta_pateince>patience or delta_patience>patience:
                break

            if verbose:
                print(f"Iter {k}:    alpha: {self.alpha}, beta: {self.beta}, delta: {self.delta}")

    def plot_phase_portrait(self):
        plt.plot(self.right_distend[0], self.right_velocity[0], color='b')
        plt.plot(self.left_distend[0], self.left_velocity[0], color='r')
        plt.gca().set_aspect('equal')
        plt.savefig('images/phase_portrait.jpg')