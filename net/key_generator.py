import numpy as np
import random
from tqdm import tqdm
from classes.annotation import Annotation

def generate_data_keys_sequential(config, recs_list, verbose=True):
    """Create data segment keys in a sequential time manner.
       Now returns [rec_idx, seg_start, seg_stop, bin_label, tri_label].

       tri_label: 0=interictal, 1=preictal (5 min before onset), 2=ictal
    """
    PREICTAL_SEC = 300  # 发作前5分钟
    boundary = getattr(config, "boundary", 0.5)  # 用于重叠判定的阈值（若未配置则用 0.5）

    def make_intervals(events):
        """返回 ictal 区间 & preictal 区间（已与前一次发作结束做截断，且>=0）"""
        ictals = []
        preictals = []
        last_off = -np.inf
        for (on, off) in events:
            ictals.append((max(0.0, on), max(0.0, off)))
            pre_start = max(last_off, on - PREICTAL_SEC)
            pre_end = on
            if pre_end > pre_start:
                preictals.append((max(0.0, pre_start), max(0.0, pre_end)))
            last_off = off
        return ictals, preictals

    def overlap_frac(a0, a1, b0, b1):
        """区间 [a0,a1] 与 [b0,b1] 的相对重叠（对 a 区间归一化）"""
        inter = max(0.0, min(a1, b1) - max(a0, b0))
        return inter / max(1e-12, (a1 - a0))

    def tri_label_for_window(t0, t1, ictals, preictals, boundary):
        # 先判 ictal（优先级最高）
        for (a, b) in ictals:
            if overlap_frac(t0, t1, a, b) >= boundary:
                return 2
        # 再判 preictal（与任一 preictal 区间有足够覆盖）
        for (a, b) in preictals:
            if overlap_frac(t0, t1, a, b) >= boundary:
                return 1
        # 否则 interictal
        return 0

    segments = []

    for idx, f in tqdm(enumerate(recs_list), disable=not verbose):
        annotations = Annotation.loadAnnotation(config.data_path, f)
        # 构造 ictal / preictal 区间
        ictal_intervals, preictal_intervals = make_intervals(annotations.events if annotations.events else [])

        # 无事件：全是 interictal（bin=0, tri=0）
        if not annotations.events:
            n_segs = int(np.floor((np.floor(annotations.rec_duration) - config.frame) / config.stride))
            if n_segs > 0:
                seg_start = np.arange(0, n_segs) * config.stride
                seg_stop = seg_start + config.frame
                bin_lab = np.zeros(n_segs)  # 原二分类
                tri_lab = np.zeros(n_segs)  # 三分类=0
                segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, bin_lab, tri_lab)))
            continue

        # 有事件：保持原有切片逻辑，同时为每个窗计算 tri_label
        evts = annotations.events
        if len(evts) == 1:
            ev = evts[0]
            # 事件前
            n_segs = int(np.floor((ev[0]) / config.stride) - 1)
            if n_segs < 0: n_segs = 0
            if n_segs > 0:
                seg_start = np.arange(0, n_segs) * config.stride
                seg_stop = seg_start + config.frame
                bin_lab = np.zeros(n_segs)
                tri_lab = np.array([tri_label_for_window(s, e, ictal_intervals, preictal_intervals, boundary)
                                    for s, e in zip(seg_start, seg_stop)])
                segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, bin_lab, tri_lab)))

            # 事件内（标为 ictal）
            n_segs = int(np.floor((ev[1] - ev[0]) / config.stride) + 1)
            if n_segs > 0:
                seg_start = np.arange(0, n_segs) * config.stride + ev[0] - config.stride
                seg_stop = seg_start + config.frame
                bin_lab = np.ones(n_segs)
                tri_lab = np.full(n_segs, 2)  # ictal
                segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, bin_lab, tri_lab)))

            # 事件后
            n_segs = int(np.floor(np.floor(annotations.rec_duration - ev[1]) / config.stride) - 1)
            if n_segs < 0: n_segs = 0
            if n_segs > 0:
                seg_start = np.arange(0, n_segs) * config.stride + ev[1]
                seg_stop = seg_start + config.frame
                bin_lab = np.zeros(n_segs)
                tri_lab = np.array([tri_label_for_window(s, e, ictal_intervals, preictal_intervals, boundary)
                                    for s, e in zip(seg_start, seg_stop)])
                segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, bin_lab, tri_lab)))

        else:
            for e, ev in enumerate(evts):
                if e == 0:
                    # 第一个事件前
                    n_segs = int(np.floor((ev[0]) / config.stride) - 1)
                    if n_segs < 0: n_segs = 0
                    if n_segs > 0:
                        seg_start = np.arange(0, n_segs) * config.stride
                        seg_stop = seg_start + config.frame
                        bin_lab = np.zeros(n_segs)
                        tri_lab = np.array([tri_label_for_window(s, e2, ictal_intervals, preictal_intervals, boundary)
                                            for s, e2 in zip(seg_start, seg_stop)])
                        segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, bin_lab, tri_lab)))

                    # 第一个事件内
                    n_segs = int(np.floor((ev[1] - ev[0]) / config.stride) + 1)
                    seg_start = np.arange(0, n_segs) * config.stride + ev[0] - config.stride
                    if np.sum(seg_start < 0) > 0:
                        n_segs -= np.sum(seg_start < 0)
                        seg_start = seg_start[seg_start >= 0]
                    seg_stop = seg_start + config.frame
                    bin_lab = np.ones(n_segs)
                    tri_lab = np.full(n_segs, 2)
                    segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, bin_lab, tri_lab)))

                elif e != len(evts) - 1:
                    prev_event = evts[e - 1]
                    # 相邻事件之间的间隔
                    n_segs = int(np.floor((ev[0] - prev_event[1]) / config.stride) - 1)
                    if n_segs < 0: n_segs = 0
                    if n_segs > 0:
                        seg_start = np.arange(0, n_segs) * config.stride + prev_event[1]
                        seg_stop = seg_start + config.frame
                        bin_lab = np.zeros(n_segs)
                        tri_lab = np.array([tri_label_for_window(s, e2, ictal_intervals, preictal_intervals, boundary)
                                            for s, e2 in zip(seg_start, seg_stop)])
                        segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, bin_lab, tri_lab)))

                    # 当前事件内
                    n_segs = int(np.floor((ev[1] - ev[0]) / config.stride) + 1)
                    if n_segs > 0:
                        seg_start = np.arange(0, n_segs) * config.stride + ev[0] - config.stride
                        seg_stop = seg_start + config.frame
                        bin_lab = np.ones(n_segs)
                        tri_lab = np.full(n_segs, 2)
                        segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, bin_lab, tri_lab)))

                elif e == len(evts) - 1:
                    prev_event = evts[e - 1]
                    # 倒数第二个事件后到最后一个事件前
                    n_segs = int(np.floor((ev[0] - prev_event[1]) / config.stride) - 1)
                    if n_segs < 0: n_segs = 0
                    if n_segs > 0:
                        seg_start = np.arange(0, n_segs) * config.stride + prev_event[1]
                        seg_stop = seg_start + config.frame
                        bin_lab = np.zeros(n_segs)
                        tri_lab = np.array([tri_label_for_window(s, e2, ictal_intervals, preictal_intervals, boundary)
                                            for s, e2 in zip(seg_start, seg_stop)])
                        segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, bin_lab, tri_lab)))

                    # 最后一个事件内
                    n_segs = int(np.floor((ev[1] - ev[0]) / config.stride) + 1)
                    if n_segs > 0:
                        seg_start = np.arange(0, n_segs) * config.stride + ev[0] - config.stride
                        seg_stop = seg_start + config.frame
                        bin_lab = np.ones(n_segs)
                        tri_lab = np.full(n_segs, 2)
                        segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, bin_lab, tri_lab)))

                    # 最后一个事件之后直到录制结束check
                    n_segs = int(np.floor((annotations.rec_duration - ev[1]) / config.stride) - 1) #check
                    if n_segs > 0:
                        seg_start = np.arange(0, n_segs) * config.stride + ev[1]
                        seg_stop = seg_start + config.frame
                        bin_lab = np.zeros(n_segs)
                        tri_lab = np.array([tri_label_for_window(s, e2, ictal_intervals, preictal_intervals, boundary)
                                            for s, e2 in zip(seg_start, seg_stop)])
                        segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, bin_lab, tri_lab)))

    return segments




def generate_data_keys_subsample(config, recs_list, mode='binary'):
    """Create data segment keys by subsampling the data.

    This refactor preserves the original structure/behavior and adds a `mode` argument:
      - mode='binary' : keep ALL ictal windows as positives; sample interictal negatives.
      - mode='triple' : keep ALL ictal + preictal windows as positives; sample interictal negatives.
        (preictal = onset - preictal_sec, truncated at previous ictal's end)

    Output rows match the sequential interface (5 columns):
        [rec_idx, seg_start, seg_stop, binary_label, tri_label]
      where:
        binary_label: 0=interictal, 1=ictal
        tri_label   : 0=interictal, 1=preictal, 2=ictal
    """
    # normalize mode
    m = (mode or 'binary').lower()
    if m in ('tri', 'triple', '3', 'triple_class', 'three'):
        m = 'triple'
    elif m != 'binary':
        raise ValueError("mode must be 'binary' or 'triple'")
    mode = m

    PREICTAL_SEC = float(getattr(config, 'preictal_sec', 300.0))  # default 5 minutes
    boundary = float(getattr(config, 'boundary', 0.5))
    frame   = float(getattr(config, 'frame'))
    stride  = float(getattr(config, 'stride'))
    stride_s= float(getattr(config, 'stride_s', stride/2.0))
    factor  = int(getattr(config, 'factor', 5))

    def _overlap_frac(a0, a1, b0, b1):
        inter = max(0.0, min(a1, b1) - max(a0, b0))
        denom = max(1e-12, (a1 - a0))
        return inter / denom

    def _build_intervals(events):
        """Return ictal and preictal intervals (preictal is clipped to previous ictal end)."""
        ictals = []
        preictals = []
        last_off = float('-inf')
        for (on, off) in events:
            on = max(0.0, float(on))
            off = max(on, float(off))
            ictals.append((on, off))
            pre_start = max(last_off, on - PREICTAL_SEC)
            if on > pre_start:
                preictals.append((max(0.0, pre_start), on))
            last_off = off
        return ictals, preictals

    def _tri_for_windows(seg_start, seg_stop, ictals, preictals):
        tri = []
        for s, e in zip(seg_start, seg_stop):
            # ictal first
            if any(_overlap_frac(s, e, a, b) >= boundary for (a, b) in ictals):
                tri.append(2)
                continue
            # preictal next
            if any(_overlap_frac(s, e, a, b) >= boundary for (a, b) in preictals):
                tri.append(1)
                continue
            tri.append(0)
        return np.asarray(tri, dtype=float)

    segments_S = []   # kept positives: ictal (+ preictal if mode='triple')
    segments_NS = []  # background candidates: interictal only (tri_label==0)

    for idx, f in tqdm(enumerate(recs_list)):
        annotations = Annotation.loadAnnotation(config.data_path, f)
        events = annotations.events if annotations.events else []
        ictal_intervals, preictal_intervals = _build_intervals(events)
        duration = float(annotations.rec_duration)

        if not annotations.events:
            # Entire recording is interictal -> only contributes to negative pool
            n_segs = int(np.floor((np.floor(duration) - frame)/stride))
            if n_segs > 0:
                seg_start = np.arange(0, n_segs)*stride
                seg_stop  = seg_start + frame
                tri_lab = np.zeros(n_segs, dtype=float)
                bin_lab = np.zeros(n_segs, dtype=float)
                segments_NS.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, bin_lab, tri_lab)))
            continue

        # ---- ICTAL positives with dense stride_s (kept in both modes) ----
        for ev in events:
            start_dense = ev[0] - frame * (1 - boundary)
            end_dense   = ev[1] + frame * (1 - boundary)
            total = end_dense - start_dense - frame
            n_segs = int(np.floor(total / stride_s) + 1) if total >= 0 else 0
            if n_segs <= 0:
                continue
            seg_start = np.arange(0, n_segs)*stride_s + start_dense
            seg_start = seg_start[seg_start >= 0]
            if seg_start.size == 0:
                continue
            seg_stop = seg_start + frame
            valid = seg_stop <= duration
            seg_start = seg_start[valid]; seg_stop = seg_stop[valid]
            n_segs = seg_start.size
            if n_segs == 0:
                continue
            bin_lab = np.ones(n_segs, dtype=float)    # ictal in binary
            tri_lab = np.full(n_segs, 2.0, dtype=float)
            segments_S.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, bin_lab, tri_lab)))

        # ---- PREICTAL positives with dense stride_s (only when mode='triple') ----
        if mode == 'triple':
            for e, ev in enumerate(events):
                prev_off = events[e-1][1] if e > 0 else float('-inf')
                pre_start = max(prev_off, ev[0] - PREICTAL_SEC)
                pre_end   = ev[0]
                if pre_end <= pre_start:
                    continue
                start_dense = pre_start - frame * (1 - boundary)
                end_dense   = pre_end   + frame * (1 - boundary)
                total = end_dense - start_dense - frame
                n_segs = int(np.floor(total / stride_s) + 1) if total >= 0 else 0
                if n_segs <= 0:
                    continue
                seg_start = np.arange(0, n_segs)*stride_s + start_dense
                seg_start = seg_start[seg_start >= 0]
                if seg_start.size == 0:
                    continue
                seg_stop = seg_start + frame
                valid = seg_stop <= duration
                seg_start = seg_start[valid]; seg_stop = seg_stop[valid]
                n_segs = seg_start.size
                if n_segs == 0:
                    continue

                # ensure preictal by overlap
                tri_lab = _tri_for_windows(seg_start, seg_stop, ictal_intervals, preictal_intervals)
                keep = (tri_lab == 1.0)
                if not np.any(keep):
                    continue
                seg_start = seg_start[keep]; seg_stop = seg_stop[keep]; tri_lab = tri_lab[keep]
                n_keep = seg_start.size
                bin_lab = np.zeros(n_keep, dtype=float)  # preictal is negative in binary
                segments_S.extend(np.column_stack(([idx]*n_keep, seg_start, seg_stop, bin_lab, tri_lab)))

        # ---- NEGATIVE candidates (interictal) with stride ----
        for e, ev in enumerate(events):
            if e == 0:
                # before first event
                n_segs = int(np.floor((ev[0])/stride)-1)
                n_segs = max(0, n_segs)
                if n_segs > 0:
                    seg_start = np.arange(0, n_segs)*stride
                    seg_stop  = seg_start + frame
                    tri_lab = _tri_for_windows(seg_start, seg_stop, ictal_intervals, preictal_intervals)
                    keep = (tri_lab == 0.0)
                    if np.any(keep):
                        seg_start = seg_start[keep]; seg_stop = seg_stop[keep]; tri_lab = tri_lab[keep]
                        bin_lab = np.zeros(seg_start.size, dtype=float)
                        segments_NS.extend(np.column_stack(([idx]*seg_start.size, seg_start, seg_stop, bin_lab, tri_lab)))
            else:
                # between prev and current event
                n_segs = int(np.floor((ev[0] - events[e-1][1])/stride)-1)
                n_segs = max(0, n_segs)
                if n_segs > 0:
                    seg_start = np.arange(0, n_segs)*stride + events[e-1][1]
                    seg_stop  = seg_start + frame
                    tri_lab = _tri_for_windows(seg_start, seg_stop, ictal_intervals, preictal_intervals)
                    keep = (tri_lab == 0.0)
                    if np.any(keep):
                        seg_start = seg_start[keep]; seg_stop = seg_stop[keep]; tri_lab = tri_lab[keep]
                        bin_lab = np.zeros(seg_start.size, dtype=float)
                        segments_NS.extend(np.column_stack(([idx]*seg_start.size, seg_start, seg_stop, bin_lab, tri_lab)))
            if e == len(events)-1:
                # after last event
                n_segs = int(np.floor((np.floor(duration) - ev[1])/stride)-1)
                n_segs = max(0, n_segs)
                if n_segs > 0:
                    seg_start = np.arange(0, n_segs)*stride + ev[1]
                    seg_stop  = seg_start + frame
                    tri_lab = _tri_for_windows(seg_start, seg_stop, ictal_intervals, preictal_intervals)
                    keep = (tri_lab == 0.0)
                    if np.any(keep):
                        seg_start = seg_start[keep]; seg_stop = seg_stop[keep]; tri_lab = tri_lab[keep]
                        bin_lab = np.zeros(seg_start.size, dtype=float)
                        segments_NS.extend(np.column_stack(([idx]*seg_start.size, seg_start, seg_stop, bin_lab, tri_lab)))

    # ---- Subsample negatives and shuffle ----
    if mode == 'binary':
        # positives counted by binary label == 1 (ictal only)
        n_pos = sum(1 for r in segments_S if int(r[3]) == 1)
    else:
        # positives are ictal + preictal
        n_pos = len(segments_S)

    sample_n = min(len(segments_NS), factor * n_pos)
    if sample_n > 0:
        segments_S.extend(random.sample(segments_NS, sample_n))
    random.shuffle(segments_S)

    return segments_S
