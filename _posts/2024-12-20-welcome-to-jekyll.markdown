---
layout: post
title:  "Numerical Simulation and Synthesis of Instrument Strings with Finite Difference Schemes"
date:   2024-12-20 19:27:17 +0900
categories: post
---

<script type="text/javascript" async
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.js">
</script>

The motivation for this topic is because of my work and interests:
![Caption](/assets/cfd.jpg)
- My study, work and research involves numerical methods and simulations
- I'm personally interested methods in DSP and numerical synthesis

In many stringed instruments, strings are subjected to a restoring force applied not only by the string tension but also by the string stiffness. Although this effect is small, it adds an audible inharmonicity to the sound, which is usually desirable. Throughout the course, we have explored the wave equation along with the damped wave equation. Now, I would like to extend that partial differential equation and attempt to simulate it using a finite difference scheme. Using the works proposed by Stefan Bilbao from the University of Edinburgh and his book, *Numerical Sound Synthesis: Finite Difference Schemes and Simulation in Musical Acoustics*, I will develop a stiff string simulation model.

In chapter 7 of the book, the following equation is presented:

$$U_{tt} = \gamma^2 u_{xx} - \kappa^2 u_{xxxx} = 0$$

This equation has been proposed by many people as the partial differential model for stiff strings. However, this equation does not consider frequency-dependent loss. In guitars or pianos, the decay in higher frequencies is much quicker than in lower frequencies, as the waves traveling through the string become more rounded. In chapter 7.3 of the book, Bilbao proposes the following equation:

$$u_{tt} = \gamma^2 u_{xx} - \kappa^2 u_{xxxx} - 2\sigma_0 u_t + 2\sigma_1 u_{txx}$$

This is the equation that I will use to model the stiff string. Now, I will go in-depth into the formulation of this equation. The first section of the equation,

$$u_{tt} = \gamma^2 u_{xx}$$

represents the original wave equation, where

$$\gamma = \frac{c}{L}$$

represents the wave propagation speed c, which is normalized to the string length L. The term 

$$- \kappa^2 u_{xxxx}$$

is the stiffness of the string. $$\kappa$$ is defined as follows by Bilbao:

$$\kappa = \sqrt{\frac{EI}{\rho A L^4}}$$

This inclusion in the equation could be interpreted as the resistance of the string to the local bending that it is subjected to. The following parameters are defined here:

- E [Pa]: Young's Modulus of Elasticity
- I [$$m^4$$]: Moment of Inertia (of the circular string cross-section)
- $$\rho$$ [$$kg/m^3$$]: The material density
- A [$$m^2$$]: The string cross-sectional area
- L [m]: The string length
- c [m/s]: Wave propagation speed

Lastly, to make this conceptual analysis of the partial differential equation complete, the following term is presented:

$$-2\sigma_0 u_t + 2\sigma_1 u_{txx}$$

This term accounts for the loss featured in the string. The first term accounts for the general loss across all the frequencies and, according to Bilbao, it can be interpreted as the constant $$T_{60}$$ across all frequencies. This loss is controlled by the $$\sigma_0$$ parameter. The $$\sigma_1$$ is used to adjust the frequency-dependent loss behavior of the model. Due to the limited scope of this project, these parameters were chosen as numerical constants.

Now that I have described the model, I would like to go in-depth into the scheme that was used to solve this equation. Firstly, the following formulation of the finite difference scheme is proposed by Bilbao:

$$\delta_{tt} u = \gamma^2 \delta_{xx} u - \kappa^2 \delta_{xxxx} u - 2\sigma_0 \delta_t u + 2\sigma_1 \delta_t \delta_{xx} u$$

From the formulation, it is apparent that this is an implicit scheme. Using this, multiple finite difference schemes were created to approximate the equation:

- The second derivative is approximated as:

  $$
  u_{xx} = \frac{u_{i-1} - 2u_i + u_{i+1}}{\Delta x^2}
  $$

- The fourth derivative is approximated as:

  $$
  u_{xxxx} = \frac{u_{i-2} - 4u_{i-1} + 6u_i - 4u_{i+1} + u_{i+2}}{\Delta x^4}
  $$

- The first time derivative is computed as:

  $$
  u_t = \frac{u^n - u^{n-1}}{\Delta t}
  $$

- The second time derivative is approximated as:

  $$
  u_{tt} = \frac{u^{n+1} - 2u^n + u^{n-1}}{\Delta t^2}
  $$

Now, the following scheme is used to compute the position of the i-th node at the n+1 timestep:

$$
u^{n+1}_i = 2u^n_i - u^{n-1}_i + \Delta t^2 \left[
\gamma u_{xx}^n - \kappa u_{xxxx}^n - 2\sigma_0 u_t^n + 2\sigma_1 u_{xx}^n
\right]
$$

Next, the boundary conditions were defined as follows:

$$
u^{n+1}_0 = 0, \quad u^{n+1}_N = 0
$$

$$
u^{n+1}_1 = u^{n+1}_0, \quad u^{n+1}_{N-1} = u^{n+1}_N
$$

These are in line with the assumption of fixed boundary conditions which were used here to simulate a clamped guitar strign. Therefore, the solution must have u=0 and $$u_x = 0$$ at the ends of the string.

Before the solver could be implemented, a few more parameters had to be defined to ensure that this scheme would be stable, consistent, and convergent. Luckily, Bilbao provides these for the reader. Although in the book they are defined for the stiff string partial differential equation without frequency-dependent loss, Bilbao mentions that these schemes are valid here. The following stability condition is defined by Bilbao:

$$
\lambda^2 + 4\mu^2 \leq 1
$$

$$
\lambda = \frac{\gamma k}{h}
$$

$$
\mu = \frac{\kappa k}{h^2}
$$

Lastly, the following condition needs to be met for the spatial grid element size, \(h\):

$$
h \geq h_{\text{min}} = \frac{\sqrt{\gamma^2 k^2 + \sqrt{\gamma^4 k^4 + 16\kappa^2 k^2}}}{2}
$$

where $$k = \frac{1}{f_s}$$. This result was rounded up to find the closest suitable string element, dx. Additionally, $$f_s$$ represents the sampling frequency in Hertz. The inverse of the sampling rate, k, was chosen as the time step, dt, which was used for the simulation of the string. The last element needed to approximate a solution to this differential equation was to set the initial conditions of the wave. Two initial waves were created: a half-sine wave and a triangular initial condition. These initial conditions are plotted in the following figure:
![](/assets/ics.png)

With the initial conditions now defined, everything was ready to run the simulation. A sampling frequency of 88400 Hz was used which results in a string segment length of approximately 6.5mm. All of the code was implemented with Pytorch in Python to allow the simulations to be solved using CUDA. Each of the simulations took approximately 3 minutes to generate, using the Nvidia A100 GPU. The GPU was accessed through Google Colab. The animations which are presented below took about 3-5 minutes each to generate. A total duration of 2 seconds was used for the simulation. To complete the simulation of the guitar string, all of the geometric and physical parameters were sourced from this website: [Physics of Guitar Strings](https://protonsforbreakfast.wordpress.com/2022/01/24/the-physics-of-guitar-strings/), which served as a major resourve for all of the parameters. 


![](/assets/1stSpectrogram.png)
Spectrogram for 1st Initial Condition

![](/assets/2D1stInitial.gif)
Simulation Results (1st Initial Condition) in 2D

![](/assets/3D1stInitial.gif)
Simulation Results (1st Initial Condition) in 3D

![](/assets/2ndSpectrogram.png)
Spectrogram for 2nd Initial Condition

![](/assets/2D2ndInitial.gif)
Simulation Results (2nd Initial Condition) in 2D

![](/assets/3D2ndInitial.gif)
Simulation Results (2nd Initial Condition) in 3D

As it can be clearly seen from the spectrograms, the results between the two simulations are not that different and there is no distinguishable harmonics in the spectrogram. Due to the long simulation time and this low quality, the code implementation was deemed not good enough for real time synthesis, which was the original goal of this project. Any attempts to generate sounds with this method were not greatly successful. Therefore, the generated sounds are not presented here.

In summary, the general simulation of the guitar string was quite successful but the synthesis aspect unfortunately left a lot to be desired. 

