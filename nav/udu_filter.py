import numpy as np

class UDUFilter:
    """UD-factorised EKF (P = U diag(d) U^T). Problem interface: dynamics(), predicted_measurement(), measurement_noise()."""

    def __init__(self, problem, x0, P0, q_diag, underweighting=1.0):
        self.problem = problem
        self.x = np.asarray(x0, float).copy()
        self.q = np.asarray(q_diag, float).copy()
        self.underweighting = float(underweighting)  # measurement underweighting factor (>1)

        # Factor the initial covariance:  P0 = U0 diag(d0) U0^T
        self.U = np.eye(problem.n_state)
        self.d = np.zeros(problem.n_state)
        n = P0.shape[0]

        for j in range(n - 1, -1, -1):
          val = P0[j, j]
          for k in range(j + 1, n):
              val -= self.U[j, k] * self.U[j, k] * self.d[k]

          self.d[j] = val

          for i in range(j):
              val = P0[i, j]
              for k in range(j + 1, n):
                  val -= self.U[i, k] * self.d[k] * self.U[j, k]

              self.U[i, j] = val / self.d[j]

    def P(self):
        return self.U @ np.diag(self.d) @ self.U.T

    def predict(self, t, dt):
        x_next, F = self.problem.dynamics(t, self.x, dt)

        # Thornton MWGS time update.
        # Form W = [F U | I_n] with weights diag([d; q]) and orthogonalize
        # row-by-row from the bottom up.

        n = self.problem.n_state

        # Augmented matrix W = [F U | I]
        W = np.hstack((F @ self.U, np.eye(n)))

        # Augmented diagonal weights d_tilde = [d; q]
        d_tilde = np.hstack((self.d, self.q))

        U_new = np.eye(n)
        d_new = np.zeros(n)

        # Modified weighted Gram-Schmidt, bottom row to top row
        for j in range(n - 1, -1, -1):

            dW_j     = d_tilde * W[j]           # (2n,) weighted row j - computed once
            d_new[j] = float(dW_j @ W[j])       # d_j = row_j . D_tilde . row_j^T

            # Orthogonalize all rows above row j
            for i in range(j):
                U_new[i, j] = float(dW_j @ W[i]) / d_new[j]
                W[i]       -= U_new[i, j] * W[j]

        self.U = U_new
        self.d = d_new
        self.x = x_next

    def update(self, t, z_meas):
        # skip if no measurements
        if len(z_meas) == 0:
            return
        for k in range(self.problem.n_meas):
            z_pred, H = self.problem.predicted_measurement(t, self.x, k)
            residual = z_meas[k] - z_pred
            R = self.problem.measurement_noise(k)

            # Bierman scalar update of (U, d, x) given H, residual, and R.
            f = self.U.T @ H
            v = self.d * f

            # 6-sigma innovation gate: S = H P H^T + R = f^T v + R
            S = float(np.dot(f, v)) + float(R)
            # if residual * residual > 36.0 * S:
            #     continue

            alpha = float(R) * self.underweighting

            K = np.zeros(self.problem.n_state)

            for j in range(self.problem.n_state):
                alpha_old = alpha
                alpha = alpha_old + f[j] * v[j]

                U_col_old = self.U[:, j].copy()

                self.d[j] = self.d[j] * alpha_old / alpha

                lam = -f[j] / alpha_old

                self.U[:, j] = self.U[:, j] + lam * K

                K = K + U_col_old * v[j]

            # State update
            self.x = self.x + K * residual / alpha

