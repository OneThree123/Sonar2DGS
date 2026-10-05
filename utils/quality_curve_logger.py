"""Durable quality records and a self-contained, offline observation page."""

import csv
import json
import math
import os
import re
import uuid
from pathlib import Path

from utils.geometry_metrics import config_digest, write_json_atomic


COLUMNS = (
    "schema_version", "run_id", "session_id", "evaluation_id", "iteration", "group",
    "requested_iterations", "effective_iterations", "timing_policy_version", "training_time_scope",
    "opacity_reset_policy_version", "trigger_reasons", "covered_target_seconds",
    "actual_training_seconds", "wall_seconds", "evaluation_seconds", "test_psnr_db", "p2s", "p2s_unit",
    "num_test_views", "num_poisson_input_points", "num_pred_mesh_vertices", "num_pred_mesh_faces",
    "num_geometry_samples", "geometry_status", "psnr_status", "p2s_status", "pred_mesh_path", "error",
)


def finite_record(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_record(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_record(item) for item in value]
    return value


class QualityCurveLogger:
    def __init__(self, model_path, config, *, state=None, resume_iteration=0):
        self.config = config
        self.config_signature = config_digest(config)
        self.session_id = uuid.uuid4().hex
        self.run_id = state["run_id"] if state else uuid.uuid4().hex
        if not re.fullmatch(r"[0-9a-f]{32}", self.run_id):
            raise ValueError("Invalid quality run ID in checkpoint")
        if state and (state.get("schema_version") != 1
                      or state.get("config_signature") != self.config_signature):
            raise ValueError("Quality checkpoint protocol does not match current configuration")
        self.output_dir = Path(model_path) / "quality_time_tests" / ("run_" + self.run_id)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "metrics.jsonl"
        self.csv_path = self.output_dir / "metrics.csv"
        self.html_path = self.output_dir / "quality_time.html"
        self.rows = []
        self.summary = {}
        self._closed = False
        self.last_evaluation_id = int(state.get("last_evaluation_id", 0)) if state else 0
        if state and self.jsonl_path.exists():
            lines = self.jsonl_path.read_text(encoding="utf-8").splitlines()
            for index, line in enumerate(lines):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    if index != len(lines) - 1:
                        raise ValueError("Corrupt quality history before its final record")
                    break  # An interrupted append may leave one incomplete tail.
                if (row["run_id"] == self.run_id and row["iteration"] <= resume_iteration
                        and row["evaluation_id"] <= self.last_evaluation_id):
                    if not self.rows or row["evaluation_id"] > self.rows[-1]["evaluation_id"]:
                        self.rows.append(row)
        existing_ids = {row["evaluation_id"] for row in self.rows}
        self.missing_history = self.last_evaluation_id - len(existing_ids)
        write_json_atomic(self.output_dir / "config.json", config)
        # The checkpoint model is authoritative; atomically discard future log records.
        temporary = self.jsonl_path.with_suffix(".tmp")
        temporary.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in self.rows), encoding="utf-8")
        os.replace(temporary, self.jsonl_path)
        self._jsonl_file = self.jsonl_path.open("a", encoding="utf-8")
        try:
            self._write_csv()
            self.write_html()
        except BaseException:
            self._jsonl_file.close()
            raise
        print(f"[Quality tests] CSV: {self.csv_path}\n[Quality tests] HTML: {self.html_path}")

    @property
    def next_evaluation_id(self):
        return self.last_evaluation_id + 1

    def evaluation_dir(self):
        return self.output_dir / "evaluations" / f"eval_{self.next_evaluation_id:06d}"

    def append(self, record):
        if self._closed:
            raise RuntimeError("Quality logger is closed")
        row = finite_record(dict(record, schema_version=1, run_id=self.run_id,
                                 session_id=self.session_id, evaluation_id=self.next_evaluation_id))
        self._jsonl_file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        self._jsonl_file.flush()
        os.fsync(self._jsonl_file.fileno())
        self.rows.append(row)
        self.last_evaluation_id = row["evaluation_id"]
        self._write_csv()
        return row

    def _write_csv(self):
        temporary = self.csv_path.with_suffix(".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=COLUMNS)
            writer.writeheader()
            for record in self.rows:
                row = {key: record.get(key) for key in COLUMNS}
                for key in ("trigger_reasons", "covered_target_seconds"):
                    row[key] = json.dumps(row[key], allow_nan=False)
                writer.writerow(row)
        os.replace(temporary, self.csv_path)

    def write_html(self):
        data = dict(rows=self.rows, config=self.config, summary=self.summary,
                    missing_history=self.missing_history)
        encoded = json.dumps(finite_record(data), ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
        temporary = self.html_path.with_suffix(".tmp")
        temporary.write_text(HTML.replace("__DATA__", encoded), encoding="utf-8")
        os.replace(temporary, self.html_path)

    def close(self, summary=None):
        if self._closed:
            return
        self._closed = True
        try:
            self.summary = finite_record(summary or {})
            self.summary["missing_history_records"] = self.missing_history
            write_json_atomic(self.output_dir / "summary.json", self.summary)
            self.write_html()
        finally:
            self._jsonl_file.close()


HTML = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Quality vs training time</title>
<style>
body{font:15px system-ui,sans-serif;background:#f4f6fa;color:#172033;max-width:1150px;margin:32px auto;padding:0 20px}
h1{font-size:26px} .card{background:white;border:1px solid #dce3ef;border-radius:10px;padding:20px;margin:18px 0}
svg{width:100%;display:block} label{cursor:pointer} .muted{color:#64748b} table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:9px;border-bottom:1px solid #e2e8f0;text-align:left;white-space:nowrap} .scroll{overflow:auto}
#tip{display:none;position:fixed;pointer-events:none;background:#172033;color:white;border-radius:7px;padding:12px;white-space:pre;z-index:2;font-size:13px}
button{margin:0 8px;padding:6px 12px;border:1px solid #b6c5d8;border-radius:6px;background:white;cursor:pointer}
</style>
<h1>Quality vs training time</h1>
<p class="muted">横轴为实际累计训练时间。PSNR 越高越好，P2S 越低越好。测试、重建与输出不计入训练时间。</p>
<div class="card"><label><input id="log" type="checkbox" checked> 对数时间轴</label><button id="download">下载 CSV</button><p id="status"></p><p id="remaining" class="muted"></p></div>
<div class="card"><h2>Test PSNR (dB)</h2><svg id="psnr" viewBox="0 0 1000 330"></svg></div>
<div class="card"><h2>P2S (m)</h2><svg id="p2s" viewBox="0 0 1000 330"></svg></div>
<div class="card scroll"><table><thead><tr><th>评测</th><th>训练时间 (s)</th><th>触发原因</th><th>覆盖节点 (s)</th><th>Iteration / Group</th><th>PSNR (dB)</th><th>P2S (m)</th><th>评测耗时 (s)</th><th>状态 / 错误</th></tr></thead><tbody id="rows"></tbody></table></div>
<div id="tip"></div><script id="quality-data" type="application/json">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById('quality-data').textContent),rows=data.rows,ns='http://www.w3.org/2000/svg';
const fmt=v=>v===null||v===undefined?'—':Number(v).toFixed(4);
function el(tag,attrs,text){const n=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs||{}))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;return n;}
const covered=new Set(rows.flatMap(r=>r.covered_target_seconds)),summary=data.summary;
document.getElementById('status').textContent=`评测 ${rows.length} 次 · 时间节点 ${summary.consumed_targets?summary.consumed_targets.length:covered.size}/${data.config.targets.length} · ${summary.termination_reason||'运行中'} · 训练 ${fmt(summary.final_training_seconds??rows.at(-1)?.actual_training_seconds)} s · 总时长 ${fmt(summary.total_run_seconds)} s`;
document.getElementById('remaining').textContent=`未执行节点: ${(summary.remaining_targets||data.config.targets.filter(t=>!covered.has(t))).join(', ')||'无'}${data.missing_history?' · 历史记录缺失 '+data.missing_history+' 条':''}${summary.missed_before_resume_targets?.length?' · 续训前缺测: '+summary.missed_before_resume_targets.join(', '):''}`;
for(const r of rows){const tr=document.createElement('tr');for(const v of [r.evaluation_id,fmt(r.actual_training_seconds),r.trigger_reasons.join(', '),r.covered_target_seconds.join(', '),`${r.iteration} / ${r.group}`,fmt(r.test_psnr_db),fmt(r.p2s),fmt(r.evaluation_seconds),`${r.psnr_status} / ${r.p2s_status}${r.error?' · '+r.error:''}`]){const td=document.createElement('td');td.textContent=v;tr.appendChild(td);}document.getElementById('rows').appendChild(tr);}
function draw(id,key,color){
 const svg=document.getElementById(id);svg.replaceChildren();const log=document.getElementById('log').checked;
 const domainRows=rows.filter(r=>!log||r.actual_training_seconds>0),points=domainRows.filter(r=>Number.isFinite(r[key]));
 if(!points.length){svg.appendChild(el('text',{x:500,y:165,'text-anchor':'middle',fill:'#64748b'},rows.length?'暂无有效指标':'尚无评测结果'));return;}
 const transform=t=>log?Math.log10(t):t;
 let xs=domainRows.map(r=>transform(r.actual_training_seconds)),xmin=Math.min(...xs),xmax=Math.max(...xs),ys=points.map(r=>r[key]),ymin=Math.min(...ys),ymax=Math.max(...ys);
 if(xmin===xmax){xmin-=log?.1:Math.max(1,Math.abs(xmin)*.05);xmax+=log?.1:Math.max(1,Math.abs(xmax)*.05);}const pad=ymin===ymax?Math.max(.001,Math.abs(ymin)*.05):(ymax-ymin)*.1;ymin-=pad;ymax+=pad;
 const X=t=>80+(transform(t)-xmin)/(xmax-xmin)*890,Y=v=>265-(v-ymin)/(ymax-ymin)*230;
 for(let i=0;i<=5;i++){const yy=35+i*46,yv=ymax-i*(ymax-ymin)/5;svg.appendChild(el('line',{x1:80,x2:970,y1:yy,y2:yy,stroke:'#e2e8f0'}));svg.appendChild(el('text',{x:72,y:yy+5,'text-anchor':'end',fill:'#64748b','font-size':12},yv.toPrecision(4)));const xv=xmin+i*(xmax-xmin)/5;svg.appendChild(el('text',{x:80+i*178,y:290,'text-anchor':'middle',fill:'#64748b','font-size':12},(log?10**xv:xv).toFixed(1)));}
 svg.appendChild(el('text',{x:525,y:318,'text-anchor':'middle',fill:'#64748b'},'实际训练时间 (s)'));
 let previous=null;for(const r of domainRows){if(!Number.isFinite(r[key])){previous=null;continue;}if(previous)svg.appendChild(el('line',{x1:X(previous.actual_training_seconds),y1:Y(previous[key]),x2:X(r.actual_training_seconds),y2:Y(r[key]),stroke:color,'stroke-width':2}));previous=r;const circle=el('circle',{cx:X(r.actual_training_seconds),cy:Y(r[key]),r:5,fill:color,tabindex:0});
 const show=e=>{const tip=document.getElementById('tip');tip.textContent=`训练: ${fmt(r.actual_training_seconds)} s\n触发: ${r.trigger_reasons.join(', ')}\n节点: ${r.covered_target_seconds.join(', ')||'无'}\nIteration / Group: ${r.iteration} / ${r.group}\nPSNR: ${fmt(r.test_psnr_db)} dB\nP2S: ${fmt(r.p2s)} m\n评测: ${fmt(r.evaluation_seconds)} s`;tip.style.display='block';tip.style.left=Math.max(4,Math.min(e.clientX+12,innerWidth-290))+'px';tip.style.top=Math.max(4,Math.min(e.clientY+12,innerHeight-190))+'px';};circle.addEventListener('mousemove',show);circle.addEventListener('mouseleave',()=>document.getElementById('tip').style.display='none');svg.appendChild(circle);}
}
function redraw(){draw('psnr','test_psnr_db','#2563eb');draw('p2s','p2s','#16a34a');}document.getElementById('log').addEventListener('change',redraw);redraw();
document.getElementById('download').onclick=()=>{const a=document.createElement('a');a.href='metrics.csv';a.download='metrics.csv';a.click();};
</script></html>'''
