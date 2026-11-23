// Shared orbit dynamics: compiled for both host (g++/nvcc) and device (nvcc).
#pragma once
#include <cmath>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

struct State {
    double x, y, z, vx, vy, vz;
};

constexpr double MU = 3.986004418e14;   // Earth gravitational parameter [m^3/s^2]
constexpr double RE = 6378137.0;        // Earth equatorial radius [m]
constexpr double J2 = 1.08262668e-3;    // Earth oblateness coefficient

// d(state)/dt : two-body gravity, plus the J2 oblateness perturbation if enabled.
HD inline State deriv(const State& s, bool useJ2) {
    const double r2 = s.x * s.x + s.y * s.y + s.z * s.z;
    const double r = sqrt(r2);
    const double k = -MU / (r2 * r);
    double ax = k * s.x, ay = k * s.y, az = k * s.z;
    if (useJ2) {
        const double z2 = s.z * s.z / r2;
        const double f = 1.5 * J2 * MU * RE * RE / (r2 * r2 * r);  // / r^5
        ax -= f * s.x * (1.0 - 5.0 * z2);
        ay -= f * s.y * (1.0 - 5.0 * z2);
        az -= f * s.z * (3.0 - 5.0 * z2);
    }
    return State{s.vx, s.vy, s.vz, ax, ay, az};
}

HD inline State axpy(const State& s, const State& k, double h) {
    return State{s.x + h * k.x, s.y + h * k.y, s.z + h * k.z,
                 s.vx + h * k.vx, s.vy + h * k.vy, s.vz + h * k.vz};
}

// One classical Runge-Kutta 4 step.
HD inline State rk4_step(const State& s, double dt, bool useJ2) {
    const State k1 = deriv(s, useJ2);
    const State k2 = deriv(axpy(s, k1, 0.5 * dt), useJ2);
    const State k3 = deriv(axpy(s, k2, 0.5 * dt), useJ2);
    const State k4 = deriv(axpy(s, k3, dt), useJ2);
    return State{
        s.x  + dt / 6.0 * (k1.x  + 2 * k2.x  + 2 * k3.x  + k4.x),
        s.y  + dt / 6.0 * (k1.y  + 2 * k2.y  + 2 * k3.y  + k4.y),
        s.z  + dt / 6.0 * (k1.z  + 2 * k2.z  + 2 * k3.z  + k4.z),
        s.vx + dt / 6.0 * (k1.vx + 2 * k2.vx + 2 * k3.vx + k4.vx),
        s.vy + dt / 6.0 * (k1.vy + 2 * k2.vy + 2 * k3.vy + k4.vy),
        s.vz + dt / 6.0 * (k1.vz + 2 * k2.vz + 2 * k3.vz + k4.vz)};
}

HD inline State propagate(State s, double dt, int steps, bool useJ2) {
    for (int i = 0; i < steps; ++i) s = rk4_step(s, dt, useJ2);
    return s;
}

// Host-only helpers ---------------------------------------------------------

// Classical orbital elements (angles in radians) -> inertial position/velocity.
inline State from_elements(double a, double e, double inc, double raan, double argp, double nu) {
    const double p = a * (1.0 - e * e);
    const double r = p / (1.0 + e * std::cos(nu));
    const double xp = r * std::cos(nu), yp = r * std::sin(nu);
    const double vf = std::sqrt(MU / p);
    const double vxp = -vf * std::sin(nu), vyp = vf * (e + std::cos(nu));
    const double cO = std::cos(raan), sO = std::sin(raan);
    const double cw = std::cos(argp), sw = std::sin(argp);
    const double ci = std::cos(inc), si = std::sin(inc);
    const double r11 = cO * cw - sO * sw * ci, r12 = -cO * sw - sO * cw * ci;
    const double r21 = sO * cw + cO * sw * ci, r22 = -sO * sw + cO * cw * ci;
    const double r31 = sw * si,                r32 = cw * si;
    return State{r11 * xp + r12 * yp, r21 * xp + r22 * yp, r31 * xp + r32 * yp,
                 r11 * vxp + r12 * vyp, r21 * vxp + r22 * vyp, r31 * vxp + r32 * vyp};
}

inline double specific_energy(const State& s) {
    const double r = std::sqrt(s.x * s.x + s.y * s.y + s.z * s.z);
    return 0.5 * (s.vx * s.vx + s.vy * s.vy + s.vz * s.vz) - MU / r;
}

// Right ascension of the ascending node from the angular momentum vector.
inline double raan_of(const State& s) {
    const double hx = s.y * s.vz - s.z * s.vy;
    const double hy = s.z * s.vx - s.x * s.vz;
    return std::atan2(hx, -hy);
}
