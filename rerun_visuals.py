import re
import pandas as pd
import rerun as rr
import rerun.blueprint as rrb
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
from data_loader import load

# Frame definition: x forward, y left, z up
# OptiTrack z,x,y --> x,y,z, switch also for quaternions


def _mirror_point_through_plane(P, A, B, C):
    """Mirror point P through the plane defined by points A, B, C."""
    AB = B - A
    AC = C - A
    n = np.cross(AB, AC)
    n = n / np.linalg.norm(n)
    d = np.dot(P - A, n)
    return P - 2 * d * n


def _calculate_flapping_frequency(signal_window, sample_rate, fft_size, freq_range):
    """Calculate dominant frequency from signal window using FFT."""
    signal = signal_window - np.mean(signal_window)
    padded_signal = np.zeros(fft_size)
    padded_signal[: len(signal)] = signal

    freqs = np.fft.fftfreq(fft_size, 1 / sample_rate)
    fft_vals = np.abs(np.fft.fft(padded_signal))

    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    valid_freqs = freqs[mask]
    valid_fft = fft_vals[mask]

    if len(valid_freqs) > 0:
        return valid_freqs[np.argmax(valid_fft)]
    return None


def _calculate_dihedral_angle(forward_body, norm_dihedral):
    """Calculate dihedral angle with proper sign."""
    cross_product = np.cross(forward_body, norm_dihedral)
    angle = np.arcsin(
        np.linalg.norm(cross_product)
        / (np.linalg.norm(forward_body) * np.linalg.norm(norm_dihedral))
    )
    return angle if cross_product[2] >= 0 else -angle

def _calculate_dihedral_dot_product(lateral_body, up_body, BC):
    """Project BC onto the forward-lateral plane, then compute angle with lateral."""
    # Remove the up component to project onto the forward-lateral plane
    BC_proj = BC - np.dot(BC, up_body) * up_body
    norm = np.linalg.norm(BC_proj)
    if norm < 1e-9:
        return 0.0
    cos_angle = np.clip(
        np.dot(BC_proj, lateral_body) / (norm * np.linalg.norm(lateral_body)),
        -1.0, 1.0,
    )
    return np.arccos(cos_angle)

def _pt(df, col, i, prefix=""):
    """Read a ZXY OptiTrack point and return it in the rerun XYZ frame."""
    return np.array([
        df[f"{prefix}{col}z"].iloc[i],
        df[f"{prefix}{col}x"].iloc[i],
        df[f"{prefix}{col}y"].iloc[i],
    ])


def _quat(df, col, i, prefix=""):
    """Read a ZXY OptiTrack quaternion and return it in the rerun frame."""
    return np.array([
        df[f"{prefix}{col}z"].iloc[i],
        df[f"{prefix}{col}x"].iloc[i],
        df[f"{prefix}{col}y"].iloc[i],
        df[f"{prefix}{col}w"].iloc[i],
    ])


class FlapperLogger:
    """Encapsulates all Rerun logging for flapper flight data.

    Parameters
    ----------
    df : pd.DataFrame
        Flight data (raw OptiTrack or processed).
    marker_radius, line_radius, axes_radius : float
        Visual sizes for points, lines, and axes.
    show_body, show_wings, show_axes, show_dihedral, show_position : bool
        Toggle individual visualization layers.
    window_size, fft_size, sample_rate : int
        FFT parameters for flapping-frequency estimation.
    freq_range : tuple[float, float]
        Frequency band of interest (Hz).
    """

    BLUEPRINT = rrb.Blueprint(
        rrb.Vertical(
            rrb.Spatial3DView(origin="/flapper/", name="flapper"),
            rrb.TimeSeriesView(origin="/dihedral/", name="dihedral", visible=True),
            rrb.TimeSeriesView(origin="/rotations/", name="rotations", visible=False),
            rrb.TimeSeriesView(origin="/frequency/", name="frequency", visible=False),
            rrb.TimeSeriesView(origin="/position/", name="position", visible=False),
            rrb.TimeSeriesView(origin="/accelerations/", name="accelerations", visible=False),
        ),
        collapse_panels=False,
    )

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        marker_radius: float = 0.02,
        line_radius: float = 0.007,
        axes_radius: float = 0.002,
        prefix: str = "",
        yaw_offset: float = 0.0,
        show_body: bool = True,
        show_wings: bool = True,
        show_axes: bool = True,
        show_dihedral: bool = True,
        show_position: bool = True,
        window_size: int = 16,
        fft_size: int = 256,
        sample_rate: int = 100,
        freq_range: tuple[float, float] = (5, 25),
    ):
        self.df = df
        self.prefix = prefix
        self.marker_radius = marker_radius
        self.line_radius = line_radius
        self.axes_radius = axes_radius

        # Pre-compute yaw correction rotation (applied to all quaternions)
        self._yaw_correction = R.from_euler("z", yaw_offset, degrees=True)

        self.show_body = show_body
        self.show_wings = show_wings
        self.show_axes = show_axes
        self.show_dihedral = show_dihedral
        self.show_position = show_position

        self.window_size = window_size
        self.fft_size = fft_size
        self.sample_rate = sample_rate
        self.freq_range = freq_range

        # Flapping-angle history for frequency estimation
        self._angle_values_R: list[float] = []
        self._angle_values_L: list[float] = []

        # Pre-compute number of body markers once
        self._n_body_markers = sum(
            1 for c in df.columns if re.match(rf"^{re.escape(prefix)}fb\d+x$", c)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, i: int) -> None:
        """Log all enabled layers for timestep *i*."""
        rr.set_time("time", timestamp=self.df["time"].iloc[i])

        if self.show_body:
            self._log_body_markers(i)
            self._log_body_strips(i)

        if self.show_wings:
            for wing in ("right", "left"):
                self._log_wing_markers(i, wing)
                self._log_wing_strips(i, wing)

        if self.show_axes:
            self._log_body_axes(i)

        if self.show_dihedral:
            self._log_dihedral_frequency(i)

        if self.show_position:
            self._log_position(i)

    # ------------------------------------------------------------------
    # Body
    # ------------------------------------------------------------------

    def _log_body_markers(self, i: int) -> None:
        p = self.prefix
        for idx in range(1, self._n_body_markers + 1):
            rr.log(
                f"/flapper/fb_body_{idx}",
                rr.Points3D(
                    _pt(self.df, f"fb{idx}", i, p),
                    colors=[0, 255, 0],
                    radii=[self.marker_radius],
                ),
            )
        rr.log(
            "/flapper/fb_body",
            rr.Points3D(
                _pt(self.df, "fb", i, p),
                colors=[0, 255, 0],
                radii=[self.marker_radius],
            ),
        )

    def _log_body_strips(self, i: int) -> None:
        p = self.prefix
        fb = _pt(self.df, "fb", i, p)
        pts = {n: _pt(self.df, f"fb{n}", i, p) for n in range(1, 6)}

        rr.log(
            "/flapper/fb_body_strips",
            rr.LineStrips3D(
                [
                    [pts[1], fb, pts[2], pts[3], pts[5], pts[4], pts[2]],
                    [pts[3], fb, pts[4]],
                    [pts[5], fb],
                ],
                radii=[self.line_radius, self.line_radius],
                colors=[[0, 255, 0], [0, 255, 0]],
            ),
        )

    def _log_position(self, i: int) -> None:
        pt = _pt(self.df, "fb", i, self.prefix)
        rr.log("/position/x", rr.Scalars(pt[0]))
        rr.log("/position/y", rr.Scalars(pt[1]))
        rr.log("/position/z", rr.Scalars(pt[2]))

    # ------------------------------------------------------------------
    # Wings
    # ------------------------------------------------------------------

    def _log_wing_markers(self, i: int, wing: str) -> None:
        color = [255, 0, 0] if wing == "right" else [0, 0, 255]
        w = wing[0]
        p = self.prefix
        rr.log(
            f"/flapper/fb_{wing}_wing",
            rr.Points3D(
                _pt(self.df, f"fb{w}w", i, p),
                colors=[255, 0, 255],
                radii=[self.marker_radius],
            ),
        )
        for idx in (1, 2, 3):
            rr.log(
                f"/flapper/fb_{wing}_wing_{idx}",
                rr.Points3D(
                    _pt(self.df, f"fb{w}w{idx}", i, p),
                    colors=color,
                    radii=[self.marker_radius],
                ),
            )

    def _log_wing_strips(self, i: int, wing: str) -> None:
        color = [255, 0, 0] if wing == "right" else [0, 0, 255]
        w = wing[0]
        p = self.prefix
        rr.log(
            f"/flapper/fb_{w}w",
            rr.LineStrips3D(
                [
                    _pt(self.df, f"fb{w}w1", i, p),
                    _pt(self.df, f"fb{w}w2", i, p),
                    _pt(self.df, f"fb{w}w3", i, p),
                ],
                radii=[self.line_radius] * 3,
                colors=[color] * 3,
            ),
        )

    # ------------------------------------------------------------------
    # Body axes
    # ------------------------------------------------------------------

    def _log_body_axes(self, i: int) -> None:
        origin = _pt(self.df, "fb", i, self.prefix)
        quat = _quat(self.df, "fbq", i, self.prefix)
        if np.any(np.isnan(quat)) or np.linalg.norm(quat) == 0:
            return
        r = (R.from_quat(quat) * self._yaw_correction).apply(np.eye(3))

        rr.log(
            "/flapper/axes",
            rr.Arrows3D(
                origins=origin,
                vectors=r * 0.5,
                radii=self.axes_radius,
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ),
        )

    # ------------------------------------------------------------------
    # Dihedral & flapping frequency
    # ------------------------------------------------------------------

    def _log_dihedral_frequency(self, i: int) -> None:
        p = self.prefix
        body = _pt(self.df, "fb", i, p)
        top_marker = _pt(self.df, "fb1", i, p)
        quat_body = _quat(self.df, "fbq", i, p)

        if np.any(np.isnan(quat_body)) or np.linalg.norm(quat_body) == 0:
            return
        r = R.from_quat(quat_body, scalar_first=False) * self._yaw_correction
        forward_body = r.apply([1, 0, 0])
        lateral_body = r.apply([0, 1, 0])
        up_body = r.apply([0, 0, 1])

        offset = 0.007 * forward_body

        for wing, angle_buf, label in (
            ("r", self._angle_values_R, "right"),
            ("l", self._angle_values_L, "left"),
        ):
            wing_root = _pt(self.df, f"fb{wing}w", i, p)
            wing_last = _pt(self.df, f"fb{wing}w3", i, p)

            BA = body - top_marker 
            BC = wing_root - body - offset

            # print(BC, offset)
            norm_dihedral = np.cross(BA, BC)

            wing_vec = wing_last - wing_root
            flapping_angle = np.arcsin(
                np.dot(wing_vec, norm_dihedral)
                / (np.linalg.norm(wing_vec) * np.linalg.norm(norm_dihedral))
            )
            angle_buf.append(flapping_angle)

            if len(angle_buf) >= self.window_size:
                freq = _calculate_flapping_frequency(
                    angle_buf[-self.window_size :],
                    self.sample_rate,
                    self.fft_size,
                    self.freq_range,
                )
                if freq is not None:
                    rr.log(f"/frequency/frequency_{label}", rr.Scalars(freq))

            dihedral = _calculate_dihedral_angle(forward_body, norm_dihedral)
            wing_lateral = -lateral_body if wing == "r" else lateral_body
            dihedral_dotproduct = _calculate_dihedral_dot_product(wing_lateral, up_body, BC)
            rr.log(f"/dihedral/dihedral_{label[0].upper()}", rr.Scalars(dihedral))
            rr.log(f"/dihedral/dihedral_dp_{label[0].upper()}", rr.Scalars(dihedral_dotproduct))

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rerun visuals for flapper flight data")
    parser.add_argument(
        "flight",
        nargs="?",
        default="hover1",
        help="Flight experiment name (e.g. hover1, climb2, lateral1)",
    )
    parser.add_argument(
        "--processed",
        action="store_true",
        default=False,
        help="Use processed data instead of raw data",
    )
    args = parser.parse_args()

    cfg = load(args.flight)

    if args.processed:
        df = pd.read_csv(f"{cfg.processed_path}{cfg.flight_exp}-processed.csv")
        prefix = "optitrack."
    else:
        prefix = ""
        df = pd.read_csv(
            cfg.optitrack_path,
            skiprows=7,
            usecols=range(1, len(cfg.optitrack_cols) + 1),
            names=cfg.optitrack_cols,
            header=None,
        )

    rr.init("Rerun_Flapper", spawn=True)
    rr.send_blueprint(FlapperLogger.BLUEPRINT)

    logger = FlapperLogger(
        df,
        prefix=prefix,
        yaw_offset=cfg.yaw_offset,
        show_body=True,
        show_wings=True,
        show_axes=True,
        show_dihedral=True,
        show_position=True,
    )

    for i in range(len(df)):
        logger.update(i)