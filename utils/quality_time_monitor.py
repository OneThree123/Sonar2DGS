"""One evaluation transaction per completed training group."""

import time

from utils.geometry_metrics import GeometryEvaluator
from utils.intensity_normalization import TEST_RENDER_NORMALIZATION_CONFIG
from utils.quality_curve_logger import QualityCurveLogger
from utils.quality_time_schedule import (
    OPACITY_RESET_POLICY_VERSION, QUALITY_TEST_SECONDS, QualityTimeSchedule,
)
from utils.training_clock import TIMING_POLICY_VERSION


class QualityTimeMonitor:
    def __init__(self, dataset, scene, group_schedule, *, checkpoint_state=None,
                 initial_seconds=0, training_time_scope="full_training", save_images=False,
                 targets=QUALITY_TEST_SECONDS):
        if not scene.getTestCameras():
            raise ValueError("Quality evaluation requires a nonempty test camera set")
        self.geometry = GeometryEvaluator(dataset.source_path, scene.model_path)
        state = checkpoint_state.get("quality_time_test_state") if isinstance(checkpoint_state, dict) else None
        if state:
            self.schedule = QualityTimeSchedule(targets, consumed=state["consumed_targets"],
                                                missed=state["missed_before_resume_targets"])
            if list(self.schedule.targets) != state["targets"]:
                raise ValueError("Quality checkpoint time targets do not match")
        else:
            self.schedule = QualityTimeSchedule(targets)
            self.schedule.missed.update(self.schedule.due(initial_seconds))
        self.training_time_scope = training_time_scope
        self.save_images = save_images
        config = dict(targets=list(self.schedule.targets), geometry=self.geometry.config,
                      timing_policy_version=TIMING_POLICY_VERSION, training_time_scope=training_time_scope,
                      opacity_reset_policy_version=OPACITY_RESET_POLICY_VERSION, group_size=group_schedule.group_size,
                      test_views=[str(camera.image_name) for camera in scene.getTestCameras()],
                      **TEST_RENDER_NORMALIZATION_CONFIG,
                      save_test_images=save_images, triggers=["time", "iteration", "final"],
                      wall_policy="since_initialization_completed_excludes_reference_preparation")
        self.logger = QualityCurveLogger(scene.model_path, config, state=state,
                                        resume_iteration=checkpoint_state["global_step"] if state else 0)
        self.group_schedule = group_schedule
        self.started = None

    def start(self):
        self.started = time.perf_counter()

    def elapsed(self):
        return 0.0 if self.started is None else time.perf_counter() - self.started

    def evaluate(self, gaussians, *, actual_seconds, iteration, group, iteration_trigger, final_trigger, report):
        reasons, due = self.schedule.triggers(actual_seconds, iteration=iteration_trigger, final=final_trigger)
        if not reasons:
            return None
        wall_seconds = self.elapsed()
        started = time.perf_counter()
        directory = self.logger.evaluation_dir()
        psnr_result = report(directory, self.save_images)
        geometry_result, artifacts = self.geometry.evaluate(gaussians, directory / "geometry")
        # report() synchronizes its GPU work before returning, including diagnostics.
        row = dict(iteration=iteration, group=group,
                   requested_iterations=self.group_schedule.requested_iterations,
                   effective_iterations=self.group_schedule.effective_iterations,
                   timing_policy_version=TIMING_POLICY_VERSION, training_time_scope=self.training_time_scope,
                   opacity_reset_policy_version=OPACITY_RESET_POLICY_VERSION,
                   trigger_reasons=reasons, covered_target_seconds=due, actual_training_seconds=actual_seconds,
                   wall_seconds=wall_seconds, evaluation_seconds=time.perf_counter() - started,
                   **psnr_result, **geometry_result)
        self.logger.append(row)
        self.schedule.commit(due)
        print(f"[Quality test] group={group} training={actual_seconds:.6f}s triggers={reasons} "
              f"targets={due} PSNR={row['test_psnr_db']} P2S={row['p2s']}m")
        try:
            self.logger.write_html()
        except OSError as exc:
            print(f"[Quality tests] HTML update failed; metrics remain saved: {exc}")
        # A completed evaluation with failed geometry still counts as an attempt;
        # Scene.save must not start a second reconstruction in the same group.
        return artifacts if artifacts is not None else {}

    def state(self):
        return dict(schema_version=1, run_id=self.logger.run_id,
                    config_signature=self.logger.config_signature,
                    last_evaluation_id=self.logger.last_evaluation_id, **self.schedule.state())

    def close(self, *, reason, actual_seconds, groups, iteration, pending_reset):
        self.logger.close(dict(termination_reason=reason, final_training_seconds=actual_seconds,
                               total_run_seconds=self.elapsed(), completed_groups=groups, iteration=iteration,
                               pending_opacity_reset=pending_reset, remaining_targets=self.schedule.remaining,
                               reference_setup_seconds=self.geometry.setup_seconds,
                               failed_evaluations=sum(
                                   row.get("psnr_status") != "ok" or row.get("p2s_status") != "ok"
                                   for row in self.logger.rows),
                               **self.schedule.state()))
