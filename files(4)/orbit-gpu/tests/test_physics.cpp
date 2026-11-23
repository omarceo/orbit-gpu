// CPU-only physics checks for the shared dynamics (no GPU needed).
#include <cstdio>
#include <cmath>
#include "../include/orbit.hpp"

static int failures = 0;
static void check(const char* name, bool ok, double value, double limit) {
    std::printf("[%s] %-34s value=%.4e limit=%.4e\n", ok ? "PASS" : "FAIL", name, value, limit);
    if (!ok) ++failures;
}

int main() {
    const double PI = std::acos(-1.0);
    const double a = 7.0e6, inc = 51.6 * PI / 180.0;
    const double dt = 10.0;
    const int steps_day = 8640;

    // 1) Two-body energy conservation over one day.
    {
        State s0 = from_elements(a, 0.001, inc, 0.3, 0.5, 0.0);
        State s1 = propagate(s0, dt, steps_day, false);
        double drift = std::fabs((specific_energy(s1) - specific_energy(s0)) / specific_energy(s0));
        check("two-body energy drift (1 day)", drift < 1e-8, drift, 1e-8);
    }

    // 2) Circular two-body orbit returns to its start after one period.
    {
        double T = 2.0 * PI * std::sqrt(a * a * a / MU);
        int n = 1000;
        State s0 = from_elements(a, 0.0, inc, 0.3, 0.5, 0.0);
        State s1 = propagate(s0, T / n, n, false);
        double err = std::sqrt((s1.x - s0.x) * (s1.x - s0.x) + (s1.y - s0.y) * (s1.y - s0.y) + (s1.z - s0.z) * (s1.z - s0.z));
        check("closure after one period [m]", err < 1.0, err, 1.0);
    }

    // 3) J2 nodal regression matches the analytic secular rate.
    {
        State s0 = from_elements(a, 0.0, inc, 0.3, 0.5, 0.0);
        State s1 = propagate(s0, dt, steps_day, true);
        double dOmega = raan_of(s1) - raan_of(s0);
        double nmean = std::sqrt(MU / (a * a * a));
        double rate = -1.5 * nmean * J2 * (RE / a) * (RE / a) * std::cos(inc);
        double expected = rate * dt * steps_day;
        double rel = std::fabs(dOmega - expected) / std::fabs(expected);
        std::printf("      simulated dRAAN=%.5f deg/day, analytic=%.5f deg/day\n",
                    dOmega * 180.0 / PI, expected * 180.0 / PI);
        check("J2 nodal regression rel. error", rel < 0.02, rel, 0.02);
    }

    std::printf(failures ? "\n%d test(s) FAILED\n" : "\nall tests passed\n", failures);
    return failures ? 1 : 0;
}
