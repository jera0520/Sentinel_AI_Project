import time
import copy
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor

# -------------------------
# Utils
# -------------------------
def iou_xyxy(a, b):
    # a,b: [x1,y1,x2,y2]
    xx1 = max(a[0], b[0])
    yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2])
    yy2 = min(a[3], b[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    union = area_a + area_b - inter + 1e-6
    return inter / union

def iou_matrix_xyxy(trk_boxes, det_boxes):
    """
    trk_boxes: Nx4, det_boxes: Mx4
    반환: NxM IoU 행렬 (넘파이 브로드캐스팅 활용)
    """
    if len(trk_boxes) == 0 or len(det_boxes) == 0:
        return np.zeros((len(trk_boxes), len(det_boxes)), dtype=np.float32)
    A = np.asarray(trk_boxes, dtype=np.float32)
    B = np.asarray(det_boxes, dtype=np.float32)
    tl = np.maximum(A[:, None, :2], B[None, :, :2])
    br = np.minimum(A[:, None, 2:], B[None, :, 2:])
    wh = np.clip(br - tl, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = (A[:, 2]-A[:, 0]) * (A[:, 3]-A[:, 1])
    area_b = (B[:, 2]-B[:, 0]) * (B[:, 3]-B[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter + 1e-6
    return inter / union

# 빠른 매칭을 위한 IoU 기반 그리디 매처(사전 계산된 IoU 사용)
def greedy_match_from_iou(iou_matrix, iou_th):
    T, M = iou_matrix.shape
    if T == 0 or M == 0:
        return [], list(range(T)), list(range(M))
    mask = iou_matrix >= float(iou_th)
    ti, di = np.where(mask)
    if ti.size == 0:
        return [], list(range(T)), list(range(M))
    vals = iou_matrix[ti, di]
    order = np.argsort(-vals)
    ti = ti[order]; di = di[order]

    matched_t = np.full(T, False, dtype=bool)
    matched_d = np.full(M, False, dtype=bool)
    matches = []
    for r in range(ti.size):
        t_idx = int(ti[r]); d_idx = int(di[r])
        if matched_t[t_idx] or matched_d[d_idx]:
            continue
        matched_t[t_idx] = True
        matched_d[d_idx] = True
        matches.append((t_idx, d_idx))

    u_tracks = [i for i in range(T) if not matched_t[i]]
    u_dets = [i for i in range(M) if not matched_d[i]]
    return matches, u_tracks, u_dets

def xywh_c_to_xyxy(det):
    # det = [cx, cy, w, h]
    cx, cy, w, h = det
    x1 = int(round(cx - w/2))
    y1 = int(round(cy - h/2))
    x2 = int(round(x1 + w))
    y2 = int(round(y1 + h))
    return [x1, y1, x2, y2]

def xyxy_to_z(bbox):
    # z = [cx, cy, s, r]^T
    x1, y1, x2, y2 = bbox
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cx = x1 + w/2.0
    cy = y1 + h/2.0
    s  = w * h
    r  = w / h
    return np.array([[cx], [cy], [s], [r]], dtype=np.float32)

def x_to_bbox(x):
    # x = [cx, cy, s, r, vx, vy, vs]^T
    cx, cy, s, r = x[0,0], x[1,0], max(1.0, x[2,0]), max(1e-3, x[3,0])
    w = np.sqrt(s * r)
    h = s / w
    x1 = int(round(cx - w/2.0))
    y1 = int(round(cy - h/2.0))
    x2 = int(round(cx + w/2.0))
    y2 = int(round(cy + h/2.0))
    return [x1, y1, x2, y2]

# def rand_color_by_id(idx):
#     # 고정 색상(재현성) : 간단한 해시
#     np.random.seed((idx * 9973) % (2**32 - 1))
#     c = np.random.randint(0, 255, size=3).tolist()
#     return (int(c[0]), int(c[1]), int(c[2]))

def nms_for_yolo_results(results, iou_th=0.95):
    """
    results: [ [label, score, [cx,cy,w,h]], ... ]
    YOLOv4가 중복 박스가 많을 때의 간단 NMS 대체 (네 코드 remove_overlap과 동일 성격).
    """
    dets = []
    for i, det in enumerate(results):
        label, score, (cx,cy,w,h) = det
        box = xywh_c_to_xyxy([cx,cy,w,h])
        dets.append({'idx': i, 'label': label, 'score': score, 'box': box, 'raw': det})

    dets = sorted(dets, key=lambda d: d['score'], reverse=True)
    keep = []
    while dets:
        cur = dets.pop(0)
        keep.append(cur)
        dets = [d for d in dets if iou_xyxy(cur['box'], d['box']) < iou_th or cur['label'] != d['label']]
    return [k['raw'] for k in keep]

# -------------------------
# Minimal Kalman Filter (SORT-style)
# state: [cx, cy, s, r, vx, vy, vs]
# meas : [cx, cy, s, r]
# -------------------------
class KalmanBox:
    def __init__(self):
        self.dim_x = 7
        self.dim_z = 4

        self.x = np.zeros((self.dim_x, 1), dtype=np.float32)
        self.P = np.eye(self.dim_x, dtype=np.float32) * 10.0

        self.F = np.eye(self.dim_x, dtype=np.float32)
        # constant velocity model (dt=1)
        self.F[0,4] = 1.0  # cx += vx
        self.F[1,5] = 1.0  # cy += vy
        self.F[2,6] = 1.0  # s  += vs

        self.H = np.zeros((self.dim_z, self.dim_x), dtype=np.float32)
        self.H[0,0] = 1.0
        self.H[1,1] = 1.0
        self.H[2,2] = 1.0
        self.H[3,3] = 1.0

        self.Q = np.eye(self.dim_x, dtype=np.float32)
        self.Q[0,0] = self.Q[1,1] = self.Q[2,2] = self.Q[3,3] = 1.0
        self.Q[4,4] = self.Q[5,5] = self.Q[6,6] = 10.0

        self.R = np.eye(self.dim_z, dtype=np.float32)
        self.R[0,0] = self.R[1,1] = 1.0
        self.R[2,2] = self.R[3,3] = 10.0

    def initiate(self, z):
        # z: (4,1)
        self.x[:4] = z
        self.x[4:] = 0.0
        self.P = np.eye(self.dim_x, dtype=np.float32) * 10.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return x_to_bbox(self.x)

    def update(self, z):
        # z: (4,1)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = z - (self.H @ self.x)
        self.x = self.x + K @ y
        I = np.eye(self.dim_x, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

# -------------------------
# Track class
# -------------------------
class Track:
    def __init__(self, bbox_xyxy, label, score, tid, now_ts):
        self.kf = KalmanBox()
        self.kf.initiate(xyxy_to_z(bbox_xyxy))

        self.id = tid
        self.label = label
        self.score = float(score)

        self.age = 0
        self.hits = 1
        self.time_since_update = 0

        self.bbox = bbox_xyxy
        self.created_at = now_ts
        #self.color = rand_color_by_id(tid)
        self.confirmed = False
        self.history = []
        self._matched_this_frame = False
        self.last_matched_bbox = bbox_xyxy

    def predict(self):
        self.age += 1
        self.time_since_update += 1
        self.bbox = self.kf.predict()
        self.history.append(self.bbox)
        return self.bbox

    def update(self, bbox_xyxy, score):
        self.time_since_update = 0
        self.hits += 1
        self.score = float(score)
        self.kf.update(xyxy_to_z(bbox_xyxy))
        self.bbox = x_to_bbox(self.kf.x)
        self._matched_this_frame = True
        self.last_matched_bbox = self.bbox

# -------------------------
# ByteTrack-Lite
# -------------------------
class ByteTrackLite:
    """
    YOLOv4 친화: 2스테이지 매칭(High/Low score) + SORT 칼만 + IoU 그리디
    - 외부 ReID/Scipy 의존성 없음
    """
    def __init__(self, fps=30, track_thresh=0.5, low_thresh=0.1,
                 match_thresh=0.7, max_age=30, min_hits=3):
        self.fps = fps
        self.track_thresh = float(track_thresh)
        self.low_thresh = float(low_thresh)
        self.match_thresh = float(match_thresh)
        self.max_age = int(max_age)             # 미검출 허용 프레임
        self.min_hits = int(min_hits)           # 확정 전 최소 업데이트 수

        self.tracks = []
        self._next_id = 0
        # IoU 행렬 계산 병렬화를 위한 스레드 풀(넘파이는 GIL을 해제하므로 스레드로도 이점 有)
        self._executor = ThreadPoolExecutor(max_workers=2)

    def _split_by_score(self, results):
        hi, lo = [], []
        for det in results:
            label, score, (cx,cy,w,h) = det
            if score >= self.track_thresh:
                hi.append(det)
            elif score >= self.low_thresh:
                lo.append(det)
        return hi, lo

    def update(self, yolo_results, frame=None):
        """
        yolo_results: [ [label, score, [cx,cy,w,h]], ... ]
        """
        now = time.time()
        # 0) 중복 제거(라벨별 고득점 우선)
        yolo_results = nms_for_yolo_results(yolo_results, iou_th=0.95)

        # 1) 하이/로우 분리
        hi, lo = self._split_by_score(yolo_results)

        # 2) 예측 단계
        for t in self.tracks:
            # 새 프레임 시작 시 매칭 여부 초기화
            t._matched_this_frame = False
            t.predict()

        removed_ids = []

        # 트랙/디텍션 박스 배열 준비
        trk = np.asarray([t.bbox for t in self.tracks], dtype=np.float32) if self.tracks else np.zeros((0,4), dtype=np.float32)
        hi_boxes = np.asarray([xywh_c_to_xyxy(det[2]) for det in hi], dtype=np.float32) if hi else np.zeros((0,4), dtype=np.float32)
        lo_boxes = np.asarray([xywh_c_to_xyxy(det[2]) for det in lo], dtype=np.float32) if lo else np.zeros((0,4), dtype=np.float32)

        # 3) IoU 행렬 병렬 계산(하이/로우)
        fut_hi = self._executor.submit(iou_matrix_xyxy, trk, hi_boxes) if hi_boxes.shape[0] > 0 else None
        fut_lo = self._executor.submit(iou_matrix_xyxy, trk, lo_boxes) if lo_boxes.shape[0] > 0 else None

        # 4) 1차 매칭 (하이 점수만)
        matches1 = []
        u_t1 = list(range(len(self.tracks)))
        u_d1 = list(range(len(hi)))
        if fut_hi is not None:
            iou_hi = fut_hi.result()
            matches1, u_t1, u_d1 = greedy_match_from_iou(iou_hi, self.match_thresh)
            # 매칭된 트랙 업데이트
            for ti, di in matches1:
                det = hi[di]
                bbox = xywh_c_to_xyxy(det[2])
                self.tracks[ti].update(bbox, det[1])
                self.tracks[ti].label = det[0]

        # 5) 2차 매칭 (남은 트랙 ↔ 로우 점수)
        if fut_lo is not None and len(u_t1) > 0:
            iou_lo = fut_lo.result()
            # 남은 트랙 부분행만 사용
            iou_lo_sub = iou_lo[u_t1, :] if iou_lo.shape[0] > 0 else iou_lo
            match2, u_t2_sub, u_d2 = greedy_match_from_iou(iou_lo_sub, self.match_thresh)
            # 인덱스 보정: 부분행 -> 전체 트랙 인덱스
            for rti, ldi in match2:
                ti = u_t1[rti]
                det = lo[ldi]
                bbox = xywh_c_to_xyxy(det[2])
                self.tracks[ti].update(bbox, det[1])
                self.tracks[ti].label = det[0]
            # u_t2_sub은 부분행 기준. 필요 시 참조용으로 남겨둘 수 있으나 이후 로직에 직접 사용하지 않음.
        
        # 6) 신규 트랙 생성 (1차에서 매칭 안 된 하이 점수 검출만으로 생성)
        if len(hi) > 0 and len(u_d1) > 0:
            for di in u_d1:
                det = hi[di]
                bbox = xywh_c_to_xyxy(det[2])
                t = Track(bbox, det[0], det[1], self._next_id, now)
                self._next_id += 1
                # 생성 직후는 관측 기반이므로 매칭된 것으로 간주
                t._matched_this_frame = True
                self.tracks.append(t)

        # 7) 생존/삭제 정리
        alive_tracks = []
        for t in self.tracks:
            if t.time_since_update > self.max_age:
                removed_ids.append(t.id)
                continue
            # 확정 여부 (초기 ID 튀는 것 억제)
            if not t.confirmed and t.hits >= self.min_hits:
                t.confirmed = True
            alive_tracks.append(t)
        self.tracks = alive_tracks

        # 8) 반환 형태(네 코드와 유사한 dict)
        out_tracks = []
        for track in self.tracks:
            x1,y1,x2,y2 = track.bbox
            cx = int((x1 + x2)/2)
            cy = int((y1 + y2)/2)
            w  = x2 - x1
            h  = y2 - y1
            matched_now = bool(getattr(track, '_matched_this_frame', False))
            draw_box = track.bbox if matched_now else track.last_matched_bbox
            dx1,dy1,dx2,dy2 = draw_box
            
            # --- roi_frame 추가 ---
            roi_frame = None
            if frame is not None:
                h_img, w_img = frame.shape[:2]
                rx1 = max(0, x1)
                ry1 = max(0, y1)
                rx2 = min(w_img, x2)
                ry2 = min(h_img, y2)
                if rx2 > rx1 and ry2 > ry1:
                    roi_frame = frame[ry1:ry2, rx1:rx2].copy()

            out_tracks.append({
                'id': track.id,
                'label': track.label,
                'score': track.score,
                'bbox': [x1,y1,x2,y2],
                'disp_bbox': [dx1,dy1,dx2,dy2],
                'bbox_pos': (cx, cy),
                'size': [w, h],
                #'color': track.color,
                'confirmed': track.confirmed,
                'time_since_update': track.time_since_update,
                'hits': track.hits,
                'matched': matched_now,
                'missed': int(track.time_since_update),
                'roi_frame': roi_frame
            })

        return out_tracks, removed_ids
