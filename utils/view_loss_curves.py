import argparse
import csv
import os
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer

try:
    from utils.loss_curve_logger import write_loss_curve_html_files
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from loss_curve_logger import write_loss_curve_html_files


def read_loss_rows(csv_path):
    rows = []
    with open(csv_path, "r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            total_gaussians = row.get("total_gaussians")
            image_loss = float(row["image_loss"])
            total_loss = float(row["total_loss"])
            rows.append(
                {
                    "iteration": int(row["iteration"]),
                    "image_loss": image_loss,
                    "total_loss": total_loss,
                    "regularization_loss": optional_float(
                        row.get("regularization_loss"),
                        default=total_loss - image_loss,
                    ),
                    "empty_loss": optional_float(row.get("empty_loss")),
                    "dist_loss": optional_float(row.get("dist_loss")),
                    "normal_loss": optional_float(row.get("normal_loss")),
                    "total_gaussians": int(float(total_gaussians)) if total_gaussians else None,
                    "training_time_seconds": optional_float(row.get("training_time_seconds")),
                    "timing_policy_version": row.get("timing_policy_version"),
                    "training_time_scope": row.get("training_time_scope"),
                }
            )
    return rows


def optional_float(value, default=None):
    if value in (None, ""):
        return default
    return float(value)


def find_loss_curve_dir(path):
    path = os.path.abspath(path)
    if os.path.isdir(path) and os.path.basename(path) == "loss_curves":
        return path

    candidate = os.path.join(path, "loss_curves")
    if os.path.isdir(candidate):
        return candidate

    raise FileNotFoundError(
        "Cannot find loss_curves directory. Pass either a model output folder "
        "or the loss_curves folder itself."
    )


def serve_html(loss_curve_dir, port):
    class ReusableTCPServer(TCPServer):
        allow_reuse_address = True

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=loss_curve_dir, **kwargs)

    with ReusableTCPServer(("127.0.0.1", port), Handler) as server:
        print(f"Serving loss curves at http://127.0.0.1:{port}/loss_curves.html")
        print(f"Image loss: http://127.0.0.1:{port}/image_loss.html")
        print(f"Total loss: http://127.0.0.1:{port}/total_loss.html")
        print(f"Regularization losses: http://127.0.0.1:{port}/regularization_losses.html")
        print(f"Empty loss: http://127.0.0.1:{port}/empty_loss.html")
        print(f"Dist loss: http://127.0.0.1:{port}/dist_loss.html")
        print(f"Normal loss: http://127.0.0.1:{port}/normal_loss.html")
        print(f"Gaussian count: http://127.0.0.1:{port}/gaussian_count.html")
        print(f"Training time: http://127.0.0.1:{port}/training_time.html")
        print("Press Ctrl+C to stop the viewer.")
        server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="Regenerate and view training loss curve HTML files.")
    parser.add_argument("path", help="Model output folder, e.g. output/sonar_run, or its loss_curves folder.")
    parser.add_argument("--serve", action="store_true", help="Start a local HTTP viewer after regenerating HTML.")
    parser.add_argument("--port", type=int, default=8010, help="Local viewer port used with --serve.")
    args = parser.parse_args()

    loss_curve_dir = find_loss_curve_dir(args.path)
    csv_path = os.path.join(loss_curve_dir, "losses.csv")
    rows = read_loss_rows(csv_path)
    html_paths = write_loss_curve_html_files(loss_curve_dir, rows)

    print(f"CSV: {csv_path}")
    print(f"All curves: {html_paths['all']}")
    print(f"Image loss: {html_paths['image']}")
    print(f"Total loss: {html_paths['total']}")
    print(f"Regularization losses: {html_paths['regularization']}")
    print(f"Empty loss: {html_paths['empty']}")
    print(f"Dist loss: {html_paths['dist']}")
    print(f"Normal loss: {html_paths['normal']}")
    print(f"Gaussian count: {html_paths['gaussians']}")
    print(f"Training time: {html_paths['training_time']}")

    if args.serve:
        serve_html(loss_curve_dir, args.port)


if __name__ == "__main__":
    main()
