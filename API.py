import numpy as _n
import base64 as _b

class TimeSeriesAPI:
    _d = 20   # number of series (dimensions)
    _h = 7    # latent dimension
    _t = 13   # number of non-linear terms
    _g = _n.random.default_rng(2025)

    _q = ["c2lu", "Y29z", "dGFuaA==", "ZXhw", "Y29zaA=="]

    @classmethod
    def _u(cls, s):
        return getattr(_n, _b.b64decode(s).decode())

    _f = [
        lambda x, f=_u.__func__(None, _q[0]): f(x),
        lambda x, f=_u.__func__(None, _q[1]): f(x),
        lambda x, f=_u.__func__(None, _q[2]): f(x),
        lambda x, f=_u.__func__(None, _q[3]): 1.0 / (1.0 + f(-x)),
        lambda x, f=_u.__func__(None, _q[2]): x * f(x),
        lambda x, f=_u.__func__(None, _q[0]): f(x * x),
        lambda x, f=_u.__func__(None, _q[4]): f(x) - 1.0,
        lambda x, f=_u.__func__(None, _q[3]): f(-(x * x)),
        lambda x, f=_u.__func__(None, _q[2]), g=_u.__func__(None, _q[0]): f(x) + 0.1 * g(3.0 * x),
        lambda x, f=_u.__func__(None, _q[1]), g=_u.__func__(None, _q[0]): f(x) * g(x),
    ]

    # Latent AR(1) dynamics
    _Phi = _g.normal(0.0, 0.5, size=(_h, _h))
    _Phi /= (1.5 * _n.linalg.norm(_Phi, ord=2) + 1e-6)  # ensure stability

    # Nonlinear mixing to observations
    _W = _g.normal(0.0, 0.7, size=(_t, _h, _d))
    _B = _g.normal(0.0, 0.3, size=(_t, 1, _d))
    _A = _g.normal(0.0, 0.4, size=(_t, 1, _d))
    _J = _g.integers(0, len(_f), size=_t)
    _S = _g.normal(0.0, 0.15, size=(_t, 1, _d))

    # Current latent state (time memory)
    _z = _g.normal(0.0, 1.0, size=(1, _h))

    @classmethod
    def get(cls, n: int = 1):
        """
        Return n successive multivariate time-series points (n, _d),
        continuing from the internal time state.
        """
        y = _n.zeros((n, cls._d))

        for i in range(n):
            # AR(1) latent dynamics
            eps = cls._g.normal(0.0, 0.7, size=(1, cls._h))
            cls._z = cls._z @ cls._Phi + eps  # shape (1, _h)
            x = cls._z  # current latent state

            # Nonlinear, randomly mixed observation with cross-series coupling
            s = _n.zeros((1, cls._d))
            for k in range(cls._t):
                r = x @ cls._W[k] + cls._B[k]       # (1, _d)
                h = cls._f[cls._J[k]](r + cls._S[k])  # (1, _d)
                s += cls._A[k] * h

            y[i] = s

        return y

class DataAPI:
    _d = 20
    _h = 7
    _t = 13
    _g = _n.random.default_rng(2025)

    _q = ["c2lu", "Y29z", "dGFuaA==", "ZXhw", "Y29zaA=="]

    @classmethod
    def _u(cls, s):
        return getattr(_n, _b.b64decode(s).decode())

    _f = [
        lambda x, f=_u.__func__(None, _q[0]): f(x),
        lambda x, f=_u.__func__(None, _q[1]): f(x),
        lambda x, f=_u.__func__(None, _q[2]): f(x),
        lambda x, f=_u.__func__(None, _q[3]): 1.0 / (1.0 + f(-x)),
        lambda x, f=_u.__func__(None, _q[2]): x * f(x),
        lambda x, f=_u.__func__(None, _q[0]): f(x * x),
        lambda x, f=_u.__func__(None, _q[4]): f(x) - 1.0,
        lambda x, f=_u.__func__(None, _q[3]): f(-(x * x)),
        lambda x, f=_u.__func__(None, _q[2]), g=_u.__func__(None, _q[0]): f(x) + 0.1 * g(3.0 * x),
        lambda x, f=_u.__func__(None, _q[1]), g=_u.__func__(None, _q[0]): f(x) * g(x),
    ]

    _W = _g.normal(0.0, 0.7, size=(_t, _h, _d))
    _B = _g.normal(0.0, 0.3, size=(_t, 1, _d))
    _A = _g.normal(0.0, 0.4, size=(_t, 1, _d))
    _J = _g.integers(0, len(_f), size=_t)
    _S = _g.normal(0.0, 0.15, size=(_t, 1, _d))

    @classmethod
    def get(cls, n: int = 1):
        z = cls._g.normal(0.0, 1.0, size=(n, cls._h))
        y = _n.zeros((n, cls._d))
        for k in range(cls._t):
            x = z @ cls._W[k] + cls._B[k]
            h = cls._f[cls._J[k]](x + cls._S[k])
            y += cls._A[k] * h
        return y
