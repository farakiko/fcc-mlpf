"""
Tier-1 track fit: differentiable weighted helix fit in torch (NOT a neural network).

Transverse: algebraic (Kasa) circle seed -> 3 Gauss-Newton iterations on the geometric
residual (differentiable: fixed iteration count, no data-dependent control flow).
Longitudinal: weighted straight line z(s) along the transverse arc length.
Covariance: (J^T W J)^-1 at the solution, propagated to perigee parameters.

Outputs per track, edm4hep TrackState conventions (lengths in mm):
  d0 [mm], phi0, omega [1/mm, signed curvature], z0 [mm], tan(lambda)
  + diag uncertainties (sd0, sphi, somega, sz0, stanl)
  + pt [GeV] = 0.3e-3 * B[T] / |omega[1/mm]|,  charge sign = sign convention of omega.

Only detector input: B. Material effects are NOT modeled (tier 1b's job): expect honest
behaviour where measurement errors dominate and optimistic covariance at low pT.

API (single track; loop or vmap outside):
  fit_helix(xy[N,2], z[N], w_xy[N], w_z[N], B=2.0) -> dict of scalars (torch)
  extrapolate_to_r(params, R) -> (x, y, z, px_hat, py_hat, pz_hat) at cylinder radius R
"""
import torch


def _kasa_seed(xy, w):
    """Weighted algebraic circle fit: minimize w*(x^2+y^2 + A x + B y + C)^2 -> center, R."""
    x, y = xy[:, 0], xy[:, 1]
    z = x * x + y * y
    X = torch.stack([x, y, torch.ones_like(x)], 1)                    # [N,3]
    Wd = w[:, None]
    A = X.T @ (Wd * X)
    b = X.T @ (w * (-z))
    sol = torch.linalg.solve(A + 1e-9 * torch.eye(3, dtype=xy.dtype, device=xy.device), b)
    a, c = -sol[0] / 2, -sol[1] / 2
    R2 = a * a + c * c - sol[2]
    R = torch.sqrt(torch.clamp(R2, min=1e-6))
    return a, c, R


def _gn_circle(xy, w, a, b, R, iters=3):
    """Gauss-Newton on geometric residuals r_i = |h_i - c| - R. Returns params + 3x3 cov."""
    for _ in range(iters):
        dx = xy[:, 0] - a
        dy = xy[:, 1] - b
        d = torch.sqrt(dx * dx + dy * dy + 1e-12)
        r = d - R                                                     # residuals
        J = torch.stack([-dx / d, -dy / d, -torch.ones_like(d)], 1)   # d r / d (a,b,R)
        JW = J * w[:, None]
        H = J.T @ JW + 1e-9 * torch.eye(3, dtype=xy.dtype, device=xy.device)
        g = J.T @ (w * r)
        step = torch.linalg.solve(H, g)
        a, b, R = a - step[0], b - step[1], R - step[2]
    # final covariance
    dx = xy[:, 0] - a; dy = xy[:, 1] - b
    d = torch.sqrt(dx * dx + dy * dy + 1e-12)
    J = torch.stack([-dx / d, -dy / d, -torch.ones_like(d)], 1)
    H = J.T @ (J * w[:, None]) + 1e-9 * torch.eye(3, dtype=xy.dtype, device=xy.device)
    cov = torch.linalg.inv(H)
    return a, b, R, cov


def fit_helix(xy, z, w_xy, w_z, B=2.0):
    """Weighted helix fit of one track. xy,z in mm; w = 1/sigma^2 [1/mm^2].

    Returns dict with d0, phi0, omega, z0, tanl, pt and diagonal sigmas
    (sd0, sphi, somega, sz0, stanl). All torch scalars (differentiable).
    """
    xy = xy.double(); z = z.double(); w_xy = w_xy.double(); w_z = w_z.double()
    a0, b0, R0 = _kasa_seed(xy, w_xy)
    a, b, R, cov_abr = _gn_circle(xy, w_xy, a0, b0, R0)

    # ---- perigee (PCA to origin) ----
    dc = torch.sqrt(a * a + b * b + 1e-12)                            # |center|
    # rotation sense from hit flow: angle at innermost vs outermost hit around the center
    rr = xy[:, 0] ** 2 + xy[:, 1] ** 2
    i_in, i_out = torch.argmin(rr), torch.argmax(rr)
    u = xy[i_in] - torch.stack([a, b])
    v = xy[i_out] - xy[i_in]
    s_rot = torch.sign(u[0] * v[1] - u[1] * v[0])                     # +1 = CCW
    # PCA position and direction (tangent, oriented outward with the hit flow)
    cx_hat, cy_hat = a / dc, b / dc
    x0, y0 = (dc - R) * cx_hat, (dc - R) * cy_hat                     # PCA point
    # velocity at PCA: z_hat x n_hat with n_hat = -c_hat, times rotation sense
    tx, ty = s_rot * cy_hat, -s_rot * cx_hat
    phi0 = torch.atan2(ty, tx)
    d0 = tx * y0 - ty * x0                                            # LCIO sign: (t x p0)_z
    omega = -s_rot / R                                                # edm4hep sign (calibrated vs KF)

    # ---- longitudinal: z(s) with s = signed arc length from PCA ----
    ang = torch.atan2(xy[:, 1] - b, xy[:, 0] - a)
    ang0 = torch.atan2(y0 - b, x0 - a)
    dang = torch.remainder(s_rot * (ang - ang0) + torch.pi, 2 * torch.pi) - torch.pi
    s = R * dang                                                      # [N] signed arc length
    Xl = torch.stack([torch.ones_like(s), s], 1)
    Hl = Xl.T @ (Xl * w_z[:, None]) + 1e-9 * torch.eye(2, dtype=xy.dtype)
    gl = Xl.T @ (w_z * z)
    sol = torch.linalg.solve(Hl, gl)
    z0, tanl = sol[0], sol[1]
    cov_l = torch.linalg.inv(Hl)

    # ---- covariance propagation (a,b,R) -> (d0, phi0, omega) via autograd jacobian ----
    def _perigee(p):
        a_, b_, R_ = p[0], p[1], p[2]
        dc_ = torch.sqrt(a_ * a_ + b_ * b_ + 1e-12)
        cxh, cyh = a_ / dc_, b_ / dc_
        x0_, y0_ = (dc_ - R_) * cxh, (dc_ - R_) * cyh
        tx_, ty_ = s_rot * cyh, -s_rot * cxh
        return torch.stack([tx_ * y0_ - ty_ * x0_,                    # d0 (LCIO sign)
                            torch.atan2(ty_, tx_),                    # phi0
                            -s_rot / R_])                             # omega
    p = torch.stack([a, b, R])
    Jp = torch.autograd.functional.jacobian(_perigee, p)              # [3,3]
    cov_per = Jp @ cov_abr @ Jp.T

    pt = 0.3e-3 * B * R                                               # R[mm] -> pt[GeV]
    return dict(
        d0=d0, phi0=phi0, omega=omega, z0=z0, tanl=tanl, pt=pt,
        sd0=torch.sqrt(torch.clamp(cov_per[0, 0], min=0)),
        sphi=torch.sqrt(torch.clamp(cov_per[1, 1], min=0)),
        somega=torch.sqrt(torch.clamp(cov_per[2, 2], min=0)),
        sz0=torch.sqrt(torch.clamp(cov_l[0, 0], min=0)),
        stanl=torch.sqrt(torch.clamp(cov_l[1, 1], min=0)),
        center_a=a, center_b=b, R=R, s_rot=s_rot,
    )


def extrapolate_to_r(params, Rcyl):
    """Helix state at cylinder radius Rcyl [mm] (calo face). Material-naive (tier 3,
    training-time path). Returns (x, y, z, ux, uy, uz) with u = unit momentum direction."""
    a, b, R, s_rot = params["center_a"], params["center_b"], params["R"], params["s_rot"]
    dc = torch.sqrt(a * a + b * b)
    # intersection of circle (center c, radius R) with cylinder radius Rcyl (law of cosines)
    cosg = torch.clamp((dc * dc + R * R - Rcyl * Rcyl) / (2 * dc * R + 1e-12), -1.0, 1.0)
    gamma = torch.acos(cosg)
    base = torch.atan2(-b, -a)                                        # center -> origin direction
    # two intersections; take the one reached moving forward along the track
    for sign in (+1.0, -1.0):
        ang = base + sign * gamma
        x = a + R * torch.cos(ang); y = b + R * torch.sin(ang)
        ang0 = torch.atan2(params["d0"] * 0 + (dc - R) * b / dc - b,
                           (dc - R) * a / dc - a)
        dang = torch.remainder(s_rot * (ang - ang0) + torch.pi, 2 * torch.pi) - torch.pi
        if bool(dang >= 0):
            break
    s = R * dang
    z = params["z0"] + params["tanl"] * s
    tx, ty = -s_rot * torch.sin(ang), s_rot * torch.cos(ang)          # tangent at that angle
    tl = params["tanl"]
    norm = torch.sqrt(1 + tl * tl)
    return x, y, z, tx / norm, ty / norm, tl / norm
