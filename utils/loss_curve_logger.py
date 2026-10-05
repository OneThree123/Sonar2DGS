import csv
import json
import os
import shutil
import uuid

from utils.training_clock import TIMING_POLICY_VERSION, TRAINING_TIME_SCOPES, validate_seconds


IMAGE_LOSS_SERIES = (
    {"key": "image_loss", "label": "Image loss", "color": "#2563eb"},
)

TOTAL_LOSS_SERIES = (
    {"key": "total_loss", "label": "Total loss", "color": "#dc2626"},
)

REGULARIZATION_SERIES = (
    {"key": "regularization_loss", "label": "Regularization loss", "color": "#9333ea"},
    {"key": "empty_loss", "label": "Empty loss", "color": "#65a30d"},
    {"key": "dist_loss", "label": "Dist loss", "color": "#ea580c"},
    {"key": "normal_loss", "label": "Normal loss", "color": "#0891b2"},
)

LOSS_SERIES = IMAGE_LOSS_SERIES + TOTAL_LOSS_SERIES + REGULARIZATION_SERIES

GAUSSIAN_SERIES = (
    {"key": "total_gaussians", "label": "Gaussians", "color": "#16a34a"},
)

TRAINING_TIME_SERIES = (
    {"key": "training_time_seconds", "label": "Training time", "color": "#7c3aed"},
)


def write_loss_curve_html_files(output_dir, rows):
    combined_html_path = os.path.join(output_dir, "loss_curves.html")
    image_html_path = os.path.join(output_dir, "image_loss.html")
    total_html_path = os.path.join(output_dir, "total_loss.html")
    regularization_html_path = os.path.join(output_dir, "regularization_losses.html")
    empty_html_path = os.path.join(output_dir, "empty_loss.html")
    dist_html_path = os.path.join(output_dir, "dist_loss.html")
    normal_html_path = os.path.join(output_dir, "normal_loss.html")
    gaussian_html_path = os.path.join(output_dir, "gaussian_count.html")
    training_time_html_path = os.path.join(output_dir, "training_time.html")

    _write_html_file(combined_html_path, "Loss curves", rows, LOSS_SERIES, "Loss")
    _write_html_file(image_html_path, "Image loss", rows, IMAGE_LOSS_SERIES, "Loss")
    _write_html_file(total_html_path, "Total loss", rows, TOTAL_LOSS_SERIES, "Loss")
    _write_html_file(regularization_html_path, "Regularization losses", rows, REGULARIZATION_SERIES, "Loss")
    _write_html_file(empty_html_path, "Empty loss", rows, [REGULARIZATION_SERIES[1]], "Loss")
    _write_html_file(dist_html_path, "Dist loss", rows, [REGULARIZATION_SERIES[2]], "Loss")
    _write_html_file(normal_html_path, "Normal loss", rows, [REGULARIZATION_SERIES[3]], "Loss")
    _write_html_file(gaussian_html_path, "Gaussian count", rows, GAUSSIAN_SERIES, "Gaussians")
    scope = rows[-1].get("training_time_scope") if rows else None
    time_title = f"Training time ({scope})" if scope else "Training time"
    if rows and not rows[-1].get("timing_policy_version"):
        time_title += " (legacy timing)"
    _write_html_file(training_time_html_path, time_title, rows, TRAINING_TIME_SERIES, "Training time (s)")

    return {
        "all": combined_html_path,
        "image": image_html_path,
        "total": total_html_path,
        "regularization": regularization_html_path,
        "empty": empty_html_path,
        "dist": dist_html_path,
        "normal": normal_html_path,
        "gaussians": gaussian_html_path,
        "training_time": training_time_html_path,
    }


class LossCurveLogger:
    columns = (
        "iteration",
        "group",
        "sample_in_group",
        "image_loss",
        "total_loss",
        "regularization_loss",
        "empty_loss",
        "dist_loss",
        "normal_loss",
        "total_gaussians",
        "training_time_seconds",
        "timing_policy_version",
        "training_time_scope",
    )

    def __init__(
        self, model_path, resume_from_iter=0, plot_interval=100, *,
        training_time_scope="full_training", resume_training_seconds=None,
        legacy_resume=False,
    ):
        if training_time_scope not in TRAINING_TIME_SCOPES:
            raise ValueError(f"Unsupported training time scope: {training_time_scope!r}")
        self.training_time_scope = training_time_scope
        self.resume_training_seconds = (
            None if resume_training_seconds is None else validate_seconds(resume_training_seconds)
        )
        self.output_dir = os.path.join(model_path, "loss_curves")
        self.csv_path = os.path.join(self.output_dir, "losses.csv")
        self.combined_html_path = os.path.join(self.output_dir, "loss_curves.html")
        self.image_html_path = os.path.join(self.output_dir, "image_loss.html")
        self.total_html_path = os.path.join(self.output_dir, "total_loss.html")
        self.regularization_html_path = os.path.join(self.output_dir, "regularization_losses.html")
        self.empty_html_path = os.path.join(self.output_dir, "empty_loss.html")
        self.dist_html_path = os.path.join(self.output_dir, "dist_loss.html")
        self.normal_html_path = os.path.join(self.output_dir, "normal_loss.html")
        self.gaussian_html_path = os.path.join(self.output_dir, "gaussian_count.html")
        self.training_time_html_path = os.path.join(self.output_dir, "training_time.html")
        self.plot_interval = max(1, int(plot_interval))
        self.rows = []
        self._closed = False
        self.archive_path = None

        os.makedirs(self.output_dir, exist_ok=True)
        if os.path.exists(self.csv_path) and (
            legacy_resume or resume_from_iter == 0 or not self._compatible_history()
        ):
            self._archive_history()
        elif resume_from_iter > 0:
            self.rows = self._load_rows_until(resume_from_iter)

        self._csv_file = open(self.csv_path, "w", newline="")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.columns)
        self._csv_writer.writeheader()
        for row in self.rows:
            self._csv_writer.writerow(row)
        self._csv_file.flush()

        try:
            self.write_html()
        except BaseException:
            self._csv_file.close()
            self._closed = True
            raise

    def _compatible_history(self):
        with open(self.csv_path, "r", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if not {"timing_policy_version", "training_time_scope"}.issubset(reader.fieldnames or ()):
                return False
            return all(
                row.get("timing_policy_version") == str(TIMING_POLICY_VERSION)
                and row.get("training_time_scope") == self.training_time_scope
                for row in reader
            )

    def _archive_history(self):
        # Preserve the old files before overwriting; UUID prevents collisions.
        self.archive_path = os.path.join(self.output_dir, "archive_" + uuid.uuid4().hex)
        os.makedirs(self.archive_path)
        for name in os.listdir(self.output_dir):
            source = os.path.join(self.output_dir, name)
            if os.path.isfile(source) and (name == "losses.csv" or name.endswith(".html")):
                shutil.copy2(source, os.path.join(self.archive_path, name))

    def _load_rows_until(self, max_iteration):
        if not os.path.exists(self.csv_path):
            return []

        rows = []
        with open(self.csv_path, "r", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                try:
                    iteration = int(row["iteration"])
                    if iteration > max_iteration:
                        continue
                    training_seconds = _optional_float(row.get("training_time_seconds"))
                    if training_seconds is not None:
                        validate_seconds(training_seconds)
                        if self.resume_training_seconds is not None and training_seconds > self.resume_training_seconds:
                            continue
                    image_loss = float(row["image_loss"])
                    total_loss = float(row["total_loss"])
                    rows.append(
                        {
                            "iteration": iteration,
                            "group": _optional_int(row.get("group")),
                            "sample_in_group": _optional_int(
                                row.get("sample_in_group")
                            ),
                            "image_loss": image_loss,
                            "total_loss": total_loss,
                            "regularization_loss": _optional_float(
                                row.get("regularization_loss"),
                                default=total_loss - image_loss,
                            ),
                            "empty_loss": _optional_float(row.get("empty_loss")),
                            "dist_loss": _optional_float(row.get("dist_loss")),
                            "normal_loss": _optional_float(row.get("normal_loss")),
                            "total_gaussians": _optional_int(row.get("total_gaussians")),
                            "training_time_seconds": training_seconds,
                            "timing_policy_version": TIMING_POLICY_VERSION,
                            "training_time_scope": self.training_time_scope,
                        }
                    )
                except (KeyError, TypeError, ValueError):
                    continue
        return rows

    def log(
        self,
        iteration,
        image_loss,
        total_loss,
        total_gaussians=None,
        empty_loss=None,
        dist_loss=None,
        normal_loss=None,
        training_time_seconds=None,
        group=None,
        sample_in_group=None,
        refresh_html=None,
    ):
        regularization_loss = _sum_optional_losses(empty_loss, dist_loss, normal_loss)
        if regularization_loss is None:
            regularization_loss = float(total_loss) - float(image_loss)

        row = {
            "iteration": int(iteration),
            "group": _optional_int(group),
            "sample_in_group": _optional_int(sample_in_group),
            "image_loss": float(image_loss),
            "total_loss": float(total_loss),
            "regularization_loss": float(regularization_loss),
            "empty_loss": _optional_float(empty_loss),
            "dist_loss": _optional_float(dist_loss),
            "normal_loss": _optional_float(normal_loss),
            "total_gaussians": _optional_int(total_gaussians),
            "training_time_seconds": (
                None if training_time_seconds is None else validate_seconds(training_time_seconds)
            ),
            "timing_policy_version": TIMING_POLICY_VERSION,
            "training_time_scope": self.training_time_scope,
        }
        self.rows.append(row)
        self._csv_writer.writerow(row)
        self._csv_file.flush()

        should_refresh_html = (
            row["iteration"] % self.plot_interval == 0
            if refresh_html is None
            else bool(refresh_html)
        )
        if should_refresh_html:
            self.write_html()

    def write_html(self):
        write_loss_curve_html_files(self.output_dir, self.rows)

    def close(self):
        if self._closed:
            return

        try:
            self._csv_file.flush()
            self.write_html()
        finally:
            self._csv_file.close()
            self._closed = True

def _optional_int(value):
    if value in (None, ""):
        return None
    return int(float(value))


def _optional_float(value, default=None):
    if value in (None, ""):
        return default
    return float(value)


def _sum_optional_losses(*values):
    total = 0.0
    found = False
    for value in values:
        if value in (None, ""):
            continue
        total += float(value)
        found = True
    return total if found else None


def _write_html_file(path, title, rows, series, y_axis_label):
    rows_json = json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
    series_json = json.dumps(series, separators=(",", ":"), ensure_ascii=True)
    html = HTML_TEMPLATE.replace("__TITLE__", title)
    html = html.replace("__ROWS_JSON__", rows_json)
    html = html.replace("__SERIES_JSON__", series_json)
    html = html.replace("__Y_AXIS_LABEL__", y_axis_label)

    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as html_file:
        html_file.write(html)
    os.replace(tmp_path, path)


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f7f4;
      --panel: #ffffff;
      --ink: #171717;
      --muted: #666a73;
      --line: #d9dde5;
      --grid: #eef0f4;
      --accent: #1769aa;
      --shadow: 0 18px 50px rgba(23, 23, 23, 0.08);
    }
    * {
      box-sizing: border-box;
    }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      width: min(1180px, calc(100vw - 32px));
      margin: 24px auto;
    }
    header {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 16px;
    }
    h1 {
      margin: 0 0 4px;
      font-size: 24px;
      font-weight: 720;
      letter-spacing: 0;
    }
    .meta {
      color: var(--muted);
      font-size: 13px;
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: flex-end;
    }
    button {
      min-height: 34px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #ffffff;
      color: var(--ink);
      padding: 0 12px;
      font: inherit;
      font-size: 13px;
      cursor: pointer;
    }
    button:hover {
      border-color: #b7becb;
      background: #fafafa;
    }
    .chart-shell {
      position: relative;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }
    canvas {
      display: block;
      width: 100%;
      height: min(72vh, 720px);
      min-height: 430px;
      touch-action: none;
      cursor: grab;
    }
    canvas.dragging {
      cursor: grabbing;
    }
    .tooltip {
      position: absolute;
      display: none;
      min-width: 150px;
      max-width: 280px;
      pointer-events: none;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: rgba(255, 255, 255, 0.96);
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.12);
      padding: 8px 10px;
      font-size: 12px;
      line-height: 1.5;
      color: var(--ink);
    }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
    }
    .legend-item {
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .swatch {
      width: 11px;
      height: 11px;
      border-radius: 3px;
      display: inline-block;
    }
    .hint {
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
    }
    @media (max-width: 700px) {
      main {
        width: min(100vw - 20px, 1180px);
        margin: 12px auto;
      }
      header {
        align-items: stretch;
        flex-direction: column;
      }
      .toolbar {
        justify-content: flex-start;
      }
      canvas {
        min-height: 360px;
      }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>__TITLE__</h1>
        <div class="meta" id="meta">Loading data...</div>
      </div>
      <div class="toolbar">
        <button type="button" id="reset">Reset</button>
        <button type="button" id="fitY">Fit Y</button>
      </div>
    </header>
    <section class="chart-shell">
      <canvas id="chart" aria-label="Training curve chart"></canvas>
      <div class="tooltip" id="tooltip"></div>
    </section>
    <div class="legend" id="legend"></div>
    <div class="hint">Mouse wheel zooms around the pointer. Drag pans. Double-click resets the view.</div>
  </main>

  <script>
    const rows = __ROWS_JSON__;
    const series = __SERIES_JSON__;

    const canvas = document.getElementById("chart");
    const ctx = canvas.getContext("2d");
    const tooltip = document.getElementById("tooltip");
    const meta = document.getElementById("meta");
    const legend = document.getElementById("legend");
    const resetButton = document.getElementById("reset");
    const fitYButton = document.getElementById("fitY");

    const pad = { left: 74, right: 24, top: 28, bottom: 50 };
    let originalView = null;
    let view = null;
    let drag = null;
    let pointer = { x: 0, y: 0, active: false };

    function finite(value) {
      return Number.isFinite(value);
    }

    function formatNumber(value) {
      if (!finite(value)) return "";
      const abs = Math.abs(value);
      if ((abs >= 10000 || (abs > 0 && abs < 0.001))) return value.toExponential(3);
      if (abs >= 100) return value.toFixed(1);
      if (abs >= 10) return value.toFixed(2);
      if (abs >= 1) return value.toFixed(3);
      return value.toFixed(5).replace(/0+$/, "").replace(/\.$/, "");
    }

    function niceTicks(min, max, count) {
      if (!finite(min) || !finite(max) || min === max) return [min || 0];
      const span = max - min;
      const rawStep = span / Math.max(1, count);
      const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
      const norm = rawStep / mag;
      const nice = norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10;
      const step = nice * mag;
      const first = Math.ceil(min / step) * step;
      const ticks = [];
      for (let v = first; v <= max + step * 0.5; v += step) ticks.push(v);
      return ticks;
    }

    function buildDomain(useVisibleX = false) {
      if (!rows.length) {
        return { x0: 0, x1: 1, y0: 0, y1: 1 };
      }

      let x0 = Infinity;
      let x1 = -Infinity;
      let y0 = Infinity;
      let y1 = -Infinity;

      for (const row of rows) {
        const x = row.iteration;
        if (!finite(x)) continue;
        if (!useVisibleX || (x >= view.x0 && x <= view.x1)) {
          x0 = Math.min(x0, x);
          x1 = Math.max(x1, x);
          for (const item of series) {
            const y = row[item.key];
            if (finite(y)) {
              y0 = Math.min(y0, y);
              y1 = Math.max(y1, y);
            }
          }
        }
      }

      if (x0 === Infinity || x1 === -Infinity) {
        x0 = rows[0].iteration || 0;
        x1 = rows[rows.length - 1].iteration || x0 + 1;
      }
      if (x0 === x1) x1 = x0 + 1;
      if (y0 === Infinity || y1 === -Infinity) {
        y0 = 0;
        y1 = 1;
      }
      if (y0 === y1) {
        const delta = Math.max(Math.abs(y0) * 0.05, 1e-6);
        y0 -= delta;
        y1 += delta;
      }
      const yPad = (y1 - y0) * 0.08;
      return { x0, x1, y0: y0 - yPad, y1: y1 + yPad };
    }

    function resetView() {
      originalView = buildDomain(false);
      view = { ...originalView };
      draw();
    }

    function fitYToVisibleX() {
      const yDomain = buildDomain(true);
      view.y0 = yDomain.y0;
      view.y1 = yDomain.y1;
      draw();
    }

    function width() {
      return canvas.clientWidth;
    }

    function height() {
      return canvas.clientHeight;
    }

    function plotWidth() {
      return Math.max(1, width() - pad.left - pad.right);
    }

    function plotHeight() {
      return Math.max(1, height() - pad.top - pad.bottom);
    }

    function insidePlot(px, py) {
      return px >= pad.left && px <= width() - pad.right && py >= pad.top && py <= height() - pad.bottom;
    }

    function xToPx(x) {
      return pad.left + ((x - view.x0) / (view.x1 - view.x0)) * plotWidth();
    }

    function yToPx(y) {
      return pad.top + ((view.y1 - y) / (view.y1 - view.y0)) * plotHeight();
    }

    function pxToX(px) {
      return view.x0 + ((px - pad.left) / plotWidth()) * (view.x1 - view.x0);
    }

    function pxToY(py) {
      return view.y1 - ((py - pad.top) / plotHeight()) * (view.y1 - view.y0);
    }

    function drawAxes() {
      ctx.save();
      ctx.lineWidth = 1;
      ctx.strokeStyle = "#d9dde5";
      ctx.fillStyle = "#666a73";
      ctx.font = "12px Inter, system-ui, sans-serif";
      ctx.textBaseline = "middle";

      const xTicks = niceTicks(view.x0, view.x1, 8);
      const yTicks = niceTicks(view.y0, view.y1, 7);

      for (const tick of xTicks) {
        const x = xToPx(tick);
        ctx.strokeStyle = "#eef0f4";
        ctx.beginPath();
        ctx.moveTo(x, pad.top);
        ctx.lineTo(x, height() - pad.bottom);
        ctx.stroke();
        ctx.fillStyle = "#666a73";
        ctx.textAlign = "center";
        ctx.fillText(formatNumber(tick), x, height() - 24);
      }

      for (const tick of yTicks) {
        const y = yToPx(tick);
        ctx.strokeStyle = "#eef0f4";
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(width() - pad.right, y);
        ctx.stroke();
        ctx.fillStyle = "#666a73";
        ctx.textAlign = "right";
        ctx.fillText(formatNumber(tick), pad.left - 10, y);
      }

      ctx.strokeStyle = "#b7becb";
      ctx.beginPath();
      ctx.moveTo(pad.left, pad.top);
      ctx.lineTo(pad.left, height() - pad.bottom);
      ctx.lineTo(width() - pad.right, height() - pad.bottom);
      ctx.stroke();

      ctx.fillStyle = "#171717";
      ctx.textAlign = "center";
      ctx.fillText("Iteration", pad.left + plotWidth() / 2, height() - 10);
      ctx.save();
      ctx.translate(18, pad.top + plotHeight() / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("__Y_AXIS_LABEL__", 0, 0);
      ctx.restore();

      ctx.restore();
    }

    function drawSeries() {
      ctx.save();
      ctx.beginPath();
      ctx.rect(pad.left, pad.top, plotWidth(), plotHeight());
      ctx.clip();

      for (const item of series) {
        ctx.strokeStyle = item.color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        let started = false;

        for (const row of rows) {
          const xValue = row.iteration;
          const yValue = row[item.key];
          if (!finite(xValue) || !finite(yValue)) {
            started = false;
            continue;
          }
          const x = xToPx(xValue);
          const y = yToPx(yValue);
          if (!started) {
            ctx.moveTo(x, y);
            started = true;
          } else {
            ctx.lineTo(x, y);
          }
        }
        ctx.stroke();
      }
      ctx.restore();
    }

    function nearestRow() {
      if (!pointer.active || !insidePlot(pointer.x, pointer.y) || !rows.length) return null;

      let best = null;
      let bestDistance = Infinity;
      for (const row of rows) {
        const x = xToPx(row.iteration);
        const distance = Math.abs(x - pointer.x);
        if (distance < bestDistance) {
          bestDistance = distance;
          best = row;
        }
      }
      return bestDistance < 42 ? best : null;
    }

    function drawHover() {
      const row = nearestRow();
      if (!row) {
        tooltip.style.display = "none";
        return;
      }

      const x = xToPx(row.iteration);
      ctx.save();
      ctx.beginPath();
      ctx.rect(pad.left, pad.top, plotWidth(), plotHeight());
      ctx.clip();
      ctx.strokeStyle = "rgba(23, 23, 23, 0.32)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, pad.top);
      ctx.lineTo(x, height() - pad.bottom);
      ctx.stroke();

      for (const item of series) {
        const yValue = row[item.key];
        if (!finite(yValue)) continue;
        ctx.fillStyle = item.color;
        ctx.beginPath();
        ctx.arc(x, yToPx(yValue), 4, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();

      const lines = [`<strong>Iteration ${row.iteration}</strong>`];
      for (const item of series) {
        lines.push(`<span style="color:${item.color}">${item.label}: ${formatNumber(row[item.key])}</span>`);
      }
      tooltip.innerHTML = lines.join("<br>");
      tooltip.style.display = "block";

      const shellRect = canvas.parentElement.getBoundingClientRect();
      const canvasRect = canvas.getBoundingClientRect();
      const left = canvasRect.left - shellRect.left + pointer.x + 14;
      const top = canvasRect.top - shellRect.top + pointer.y + 14;
      tooltip.style.left = `${Math.min(left, shellRect.width - tooltip.offsetWidth - 10)}px`;
      tooltip.style.top = `${Math.min(top, shellRect.height - tooltip.offsetHeight - 10)}px`;
    }

    function drawEmpty() {
      ctx.clearRect(0, 0, width(), height());
      ctx.fillStyle = "#666a73";
      ctx.font = "14px Inter, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("No data has been written yet.", width() / 2, height() / 2);
      tooltip.style.display = "none";
    }

    function draw() {
      ctx.clearRect(0, 0, width(), height());
      if (!rows.length) {
        drawEmpty();
        return;
      }
      drawAxes();
      drawSeries();
      drawHover();
    }

    function resize() {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    }

    function zoomAt(px, py, factor) {
      const x = pxToX(px);
      const y = pxToY(py);
      view.x0 = x + (view.x0 - x) * factor;
      view.x1 = x + (view.x1 - x) * factor;
      view.y0 = y + (view.y0 - y) * factor;
      view.y1 = y + (view.y1 - y) * factor;
      draw();
    }

    function updateLegend() {
      legend.innerHTML = "";
      for (const item of series) {
        const span = document.createElement("span");
        span.className = "legend-item";
        span.innerHTML = `<span class="swatch" style="background:${item.color}"></span>${item.label}`;
        legend.appendChild(span);
      }
      if (rows.length) {
        const first = rows[0].iteration;
        const last = rows[rows.length - 1].iteration;
        meta.textContent = `${rows.length} samples, iterations ${first} to ${last}`;
      } else {
        meta.textContent = "0 samples";
      }
    }

    canvas.addEventListener("wheel", (event) => {
      if (!view || !insidePlot(event.offsetX, event.offsetY)) return;
      event.preventDefault();
      const factor = event.deltaY > 0 ? 1.18 : 1 / 1.18;
      zoomAt(event.offsetX, event.offsetY, factor);
    }, { passive: false });

    canvas.addEventListener("pointerdown", (event) => {
      if (!view || !insidePlot(event.offsetX, event.offsetY)) return;
      drag = {
        x: event.offsetX,
        y: event.offsetY,
        view: { ...view },
      };
      canvas.classList.add("dragging");
      canvas.setPointerCapture(event.pointerId);
    });

    canvas.addEventListener("pointermove", (event) => {
      pointer = { x: event.offsetX, y: event.offsetY, active: true };
      if (drag) {
        const dx = event.offsetX - drag.x;
        const dy = event.offsetY - drag.y;
        const xr = drag.view.x1 - drag.view.x0;
        const yr = drag.view.y1 - drag.view.y0;
        view.x0 = drag.view.x0 - (dx / plotWidth()) * xr;
        view.x1 = drag.view.x1 - (dx / plotWidth()) * xr;
        view.y0 = drag.view.y0 + (dy / plotHeight()) * yr;
        view.y1 = drag.view.y1 + (dy / plotHeight()) * yr;
      }
      draw();
    });

    canvas.addEventListener("pointerleave", () => {
      pointer.active = false;
      if (!drag) draw();
    });

    canvas.addEventListener("pointerup", () => {
      drag = null;
      canvas.classList.remove("dragging");
    });

    canvas.addEventListener("dblclick", resetView);
    resetButton.addEventListener("click", resetView);
    fitYButton.addEventListener("click", fitYToVisibleX);
    window.addEventListener("resize", resize);

    resetView();
    updateLegend();
    resize();
  </script>
</body>
</html>
"""
