"""Architecture diagrams for the report (drawn by `run_m3.py --stage collect`, figures/arch_*.png):
the five-stage modular pipeline (Step 7 deliverable) and the software block diagram of the cascade and LLM arbitration
(Step 8 deliverable). Boxes are sized from their text (measured after a draw), so nothing overflows; the cascade
diagram's counts are read from the results files, never typed."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, INK2, ACC, ACC2, WARN, GREEN, GREY = '#0b0b0b', '#3f3f3f', '#1f4e79', '#2a78d6', '#b8860b', '#3a7d44', '#7a7a7a'


class Canvas:
    def __init__(self, w, h):
        self.fig, self.ax = plt.subplots(figsize=(w, h), dpi=210); self.ax.set_xlim(0, 1); self.ax.set_ylim(0, 1); self.ax.axis('off')   # measure at the output dpi (text metrics are dpi-dependent)
        self.fig.subplots_adjust(0, 0, 1, 1); self.ax.patch.set_visible(False)

    def _ext(self, artist):
        self.fig.canvas.draw(); bb = artist.get_window_extent(self.fig.canvas.get_renderer())
        (x0, y0), (x1, y1) = self.ax.transAxes.inverted().transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        return x0, y0, x1, y1

    def block(self, xc, ytop, title, body, fc='#f2f5fb', ec=ACC, tsize=8.0, bsize=6.6, ls='-', tcolor=INK, pad=0.012, minw=0.0, left=None, gap=0.02):
        """Title (bold) over body, boxed to their measured extent. `left` places the box's left edge at that x instead of
        centring it at xc. Returns (x0, y0, x1, y1) in axes coordinates."""
        t = self.ax.text(xc, ytop, title, ha='center', va='top', fontsize=tsize, fontweight='bold', color=tcolor, linespacing=1.15, zorder=3)
        tx0, ty0, tx1, ty1 = self._ext(t); texts = [t]; parts = [(tx0, ty0, tx1, ty1)]
        if body:
            b = self.ax.text(xc, ty0 - 0.011 * (8.0 / max(self.fig.get_figheight(), 1)), body, ha='center', va='top', fontsize=bsize, color=INK2, linespacing=1.25, zorder=3); texts.append(b); parts.append(self._ext(b))
        x0 = min(p[0] for p in parts); x1 = max(p[2] for p in parts); y0 = min(p[1] for p in parts); y1 = max(p[3] for p in parts)
        if x1 - x0 < minw: cx = (x0 + x1) / 2; x0, x1 = cx - minw / 2, cx + minw / 2
        if left is not None:                                   # shift so that the padded box starts at `left`
            dx = (left + pad) - x0
            for tt in texts: tt.set_x(tt.get_position()[0] + dx)
            x0 += dx; x1 += dx
        x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad
        self.ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle='round,pad=0.004,rounding_size=0.012', fc=fc, ec=ec, lw=1.1, ls=ls, zorder=2))
        return x0, y0, x1, y1

    def arrow(self, p, q, text=None, color=INK2, ls='-', tsize=6.6, toff=(0, 0.012), rad=0.0, ha='center'):
        self.ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=11, lw=1.0, color=color, ls=ls, connectionstyle=f'arc3,rad={rad}', shrinkA=1, shrinkB=1, zorder=4))
        if text: self.ax.text((p[0] + q[0]) / 2 + toff[0], (p[1] + q[1]) / 2 + toff[1], text, ha=ha, va='bottom', fontsize=tsize, color=color, linespacing=1.15, zorder=5)

    def save(self, path):
        from matplotlib.transforms import Bbox
        self.fig.canvas.draw(); r = self.fig.canvas.get_renderer()
        bb = Bbox.union([a.get_window_extent(r) for a in list(self.ax.texts) + list(self.ax.patches) + list(self.ax.lines)]).transformed(self.fig.dpi_scale_trans.inverted())
        self.fig.savefig(path, dpi=210, bbox_inches=bb.expanded(1.0, 1.0), pad_inches=0.05); plt.close(self.fig)   # crop to the drawn artists


def draw_pipeline(path):
    """Five stages left to right, drawn at the width the report prints it (6.8 in), so the 6-pt labels stay 6 pt."""
    c = Canvas(7.2, 2.6); ax = c.ax
    ax.text(0.5, 0.975, 'run_m3.py --stage ingest | features | train | refine | sensitivity | cross | cascade | llm | collect  (checkpointed; config.py holds every parameter and seed)',
            ha='center', va='top', fontsize=5.0, color=INK2, family='DejaVu Sans Mono')
    stages = [
        ('Stage 1 · Source adapters\nm3pkg/ingest.py', 'one parser per raw source:\nLMD-2023 Sysmon CSV (94 col.)\nMordor APT29 event JSON\n(Sysmon, Security, System,\nPowerShell → “other”)\n→ intermediate event table,\n26 columns: row id, time, host,\nevent id, label, 20 fields', '#fdf1e6', WARN),
        ('Stage 2 · Features\nm3pkg/common_features.py', 'Milestone-2 code, one function\n42 candidate features / event:\nevent properties + trailing\nper-host windows (causal)\n→ 35-column unified schema\n(27-column reduced set)', '#f2f5fb', ACC),
        ('Stage 3 · Pre-processing\nm3pkg/features.py', 'StratifiedGroupKFold\n(host × time block, 5 folds)\ngrouped inner hold-out (20 %)\nrobust scaler fitted on\ntraining rows; quantile\nalignment (transfer only)', '#f2f5fb', ACC),
        ('Stage 4 · Models\nm3pkg/models.py, windows.py', 'Random Forest · LightGBM\nIsolation Forest · LSTM\none fit() / score() interface\n→ attack score in [0, 1]\nclass weights inside fit()\nearly stopping on inner AUC', '#f2f5fb', ACC),
        ('Stage 5 · Evaluation\nevaluate.py pipeline.py\ncascade.py llm.py collect.py', 'out-of-fold metrics, error\nforensics, sensitivity, cross-\ndataset transfer, refinement,\ncascade (Stages 1–3) + LLM\narbitration → m3_results.json\n→ report generator', '#f2f5fb', ACC),
    ]
    boxes = []; left = 0.004
    for title, body, fc, ec in stages:
        b = c.block(0.5, 0.885, title, body, fc=fc, ec=ec, tsize=5.8, bsize=5.3, left=left, pad=0.007); boxes.append(b); left = b[2] + 0.011
    for a, b in zip(boxes, boxes[1:]):
        ym = (a[1] + a[3]) / 2; c.arrow((a[2], ym), (b[0], ym), color=INK)
    yb = min(b[1] for b in boxes) - 0.05
    ax.text((boxes[0][0] + boxes[0][2]) / 2, yb, 'the only stage with\ndataset-specific knowledge', ha='center', va='top', fontsize=5.4, color=WARN, linespacing=1.2)
    xs, xe = boxes[1][0], boxes[4][2]
    ax.plot([xs, xs, xe, xe], [yb, yb - 0.045, yb - 0.045, yb], color=ACC, lw=0.9)
    ax.text((xs + xe) / 2, yb - 0.065, 'dataset-agnostic: from Stage 2 on every module receives only the intermediate table or the feature matrix and holds no dataset-specific branch (a source label\n'
            'travels as a bookkeeping column); the only dataset-specific inputs are two configuration numbers (grouping block length 10 / 2 min, LSTM stride 4 / 2)', ha='center', va='top', fontsize=5.4, color=ACC, linespacing=1.25)
    c.save(path)


def draw_cascade(path, cascade, llm, cfg):
    """cascade / llm: {ds: cascade_<ds>.json / llm_<ds>.json} (missing entries tolerated). Drawn at print width (6.8 in)."""
    c = Canvas(7.2, 3.75); ax = c.ax
    P, NPV = cfg.CASCADE_PRECISION_TARGET, cfg.CASCADE_NPV_TARGET; MAXE, MINE = cfg.LLM_MAX_ESCALATIONS, cfg.LLM_MIN_ESCALATIONS
    ax.text(0.5, 0.985, 'Routing uses the stored out-of-fold scores of the fold models; every threshold and band comes from the inner hold-out of that fold, never from the test fold.',
            ha='center', va='top', fontsize=4.9, color=INK2)
    T, B, PD = 5.6, 5.1, 0.006
    ev = c.block(0.1, 0.92, 'Event (one telemetry row)', 'feature vector (35-column\nschema) + out-of-fold fold id;\nanomaly percentile carried\nalong as evidence only', fc='#ffffff', ec=GREY, tsize=T, bsize=B, left=0.004, pad=PD)
    s1 = c.block(0.45, 0.93, 'Stage 1 — LightGBM', f'score s; per fold, from the inner hold-out: θ = F1-optimal threshold\n(→ Stage-1 verdict) and band (τ_lo, τ_hi): τ_hi = smallest score with\ncumulative precision ≥ {P:.0%}, τ_lo = largest score with cumulative NPV\n≥ {NPV:.1%} (400-quantile grid; τ_lo ≥ τ_hi → the band collapses to a point)', tsize=T, bsize=B, left=ev[2] + 0.022, pad=PD)
    c.arrow((ev[2], (ev[1] + ev[3]) / 2), (s1[0], (ev[1] + ev[3]) / 2), color=INK)
    RW = 0.235
    acc = c.block(0.85, 0.93, 'Outside the band: Stage-1 verdict kept', 'attack if s ≥ θ, else benign — the band\nonly decides whether a second opinion is\nbought (θ may lie outside the band)', fc='#eaf6ea', ec=GREEN, tsize=T, bsize=B, left=s1[2] + 0.03, pad=PD, minw=RW)
    c.arrow((s1[2], (acc[1] + acc[3]) / 2), (acc[0], (acc[1] + acc[3]) / 2), text='outside the band', color=GREEN, toff=(0, 0.006), tsize=4.9)
    s2 = c.block(0.45, s1[1] - 0.085, 'Stage 2 — Random Forest', 'ambiguous rows only (τ_lo < s < τ_hi): verdict at its own inner-validation\nthreshold θ₂ — on the same side as the Stage-1 lean (s ≥ θ)?', tsize=T, bsize=B, left=s1[0], pad=PD)
    xm = (s1[0] + s1[2]) / 2; c.arrow((xm, s1[1]), (xm, s2[3]), text='ambiguous', color=ACC, toff=(0.05, -0.012), tsize=4.9)
    res = c.block(0.85, s2[3], 'Resolved by Stage 2', 'the two models agree → Stage-2 verdict', fc='#eaf6ea', ec=GREEN, tsize=T, bsize=B, left=acc[0], pad=PD, minw=RW)
    c.arrow((s2[2], (res[1] + res[3]) / 2), (res[0], (res[1] + res[3]) / 2), text='agree', color=GREEN, toff=(0, 0.02), tsize=4.9)
    dis = c.block(0.85, res[1] - 0.03, 'Disputed', 'Random Forest vetoes / overrides LightGBM\n→ provisional RF verdict; escalation order:\nsmallest |s − θ| first (equal scores:\narbitrary draw, disclosed)', fc='#fdf1e6', ec=WARN, tsize=T, bsize=B, left=acc[0], pad=PD, minw=RW)
    c.arrow((s2[2], s2[1] + 0.01), (dis[0], dis[3] - 0.03), text='disagree', color=WARN, toff=(-0.045, 0.004), tsize=4.9)
    s3 = c.block(0.33, s2[1] - 0.085, f'Stage 3 — local LLM arbiter (up to {MAXE} disputed rows per dataset)',
                 'Llama 3.1 8B Instruct (Q4_K_M) via Ollama, temperature 0, seed 42; context = event fields (masked,\ncommand line truncated) + host trailing-window block + both scores with their thresholds + anomaly\npercentile → JSON {verdict, confidence, reason}; the LLM verdict replaces the Random-Forest verdict', fc='#f2f5fb', ec=ACC2, tsize=T, bsize=B, left=0.004, pad=PD)
    c.arrow((dis[0], dis[1] + 0.02), (s3[2], s3[3] - 0.02), text=f'first {MAXE} (cap)', color=WARN, toff=(0.0, 0.008), tsize=4.9)
    rem = c.block(0.85, dis[1] - 0.06, 'Remaining disputed rows', 'keep the Random-Forest verdict (no LLM call)', fc='#ffffff', ec=GREY, tsize=T, bsize=B, left=acc[0], pad=PD, minw=RW)
    xr = (rem[0] + rem[2]) / 2; c.arrow((xr, dis[1]), (xr, rem[3]), text='beyond the cap', color=GREY, toff=(0.06, -0.006), tsize=4.9)
    top = c.block(0.33, s3[1] - 0.045, f'Diagnostic top-ups (fewer than {MINE} disputed rows): the non-disputed rows nearest θ are\nsent to the LLM too, flagged — they appear in the arbitration table only, never in\nthe cascade metrics',
                  '', fc='#ffffff', ec=GREY, ls='--', tsize=B, bsize=B, left=0.004, pad=PD, tcolor=INK2)
    xt = (top[0] + top[2]) / 2; c.arrow((xt, top[3]), (xt, s3[1]), color=GREY, ls='--')
    rows = []
    for ds, name in (('lmd', 'LMD-2023    '), ('mordor', 'Mordor APT29')):
        cc = cascade.get(ds); l = llm.get(ds)
        if not cc: continue
        n_esc = len(cc.get('escalated_positions', [])); n_top = len(cc.get('near_threshold_topup_positions', []))
        s3f = f"{l['stage3']['f1']:.3f}" if l and l.get('stage3') and cc['n_disputed'] else '  —  '
        rows.append(f"{name}: {cc['n']:>9,} rows → {cc['n_ambiguous']:>5,} ambiguous ({cc['ambiguous_rate']:.2%}) → {cc['n_disputed']:>5,} disputed ({cc['disputed_rate']:.2%}) → {n_esc - n_top:>3} escalated + {n_top:>2} top-ups | F1 S1 {cc['stage1']['f1']:.3f} → S2 {cc['stage2']['f1']:.3f} → S3 {s3f}")
    if rows:
        ax.text(0.5, min(top[1], rem[1]) - 0.03, '\n'.join(rows), ha='center', va='top', fontsize=4.9, color=INK, family='DejaVu Sans Mono', linespacing=1.4)
    c.save(path)
