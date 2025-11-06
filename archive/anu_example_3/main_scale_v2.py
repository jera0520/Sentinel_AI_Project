import cv2
import multiprocessing
import time
import numpy as np
import ffmpeg
import os
import queue as pyqueue

from lib.init import *
from lib.yolov4 import Yolo
from lib.bytracker import ByteTrackLite, xywh_c_to_xyxy
from lib.upscale_new import build_filter_graph

# --- 옵션: OpenCV/BLAS 스레드 과도 경쟁 방지 ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
cv2.setNumThreads(0)

class VideoParser(multiprocessing.Process):
    def __init__(self, video_files, queue_out):
        multiprocessing.Process.__init__(self)
        self.video_files = video_files
        self.queue_out = queue_out

    def run(self):
        for video_file in self.video_files:
            try:
                probe = ffmpeg.probe(video_file)
                video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
                if video_stream is None:
                    print(f"No video stream found in {video_file}, skipping.")
                    continue
                
                width = int(video_stream['width'])
                height = int(video_stream['height'])

                # 원본용 스트림과 업스케일용 스트림을 각각 생성
                in_stream_orig = ffmpeg.input(video_file, **{'re': None}, threads=0)
                in_stream_up = ffmpeg.input(video_file, **{'re': None}, threads=0)
                
                # 업스케일 파이프라인 구성 (build_filter_graph 사용)
                vf, out_w, out_h = build_filter_graph(in_stream_up, width, height, scale_factor=1.25, keep_ar=True, preset='balanced')
                
                process_orig = (
                    in_stream_orig
                    .output('pipe:', format='rawvideo', pix_fmt='bgr24', vsync='vfr')
                    .global_args('-nostats', '-loglevel', 'error')
                    .run_async(pipe_stdout=True, quiet=True)
                )
                process_up = (
                    vf
                    .output('pipe:', format='rawvideo', pix_fmt='bgr24', vsync='vfr')
                    .global_args('-nostats', '-loglevel', 'error')
                    .run_async(pipe_stdout=True, quiet=True)
                )

                while True:
                    in_bytes_orig = process_orig.stdout.read(width * height * 3)
                    in_bytes_up = process_up.stdout.read(out_w * out_h * 3)
                    if not in_bytes_orig or not in_bytes_up:
                        break
                    
                    frame_orig = np.frombuffer(in_bytes_orig, np.uint8).reshape(height, width, 3)
                    frame_up = np.frombuffer(in_bytes_up, np.uint8).reshape(out_h, out_w, 3)
                    try:
                        self.queue_out.put_nowait((frame_orig, frame_up))
                    except pyqueue.Full:
                        # 큐가 가득 찼으면 잠시 기다렸다가 다시 시도
                        time.sleep(0.01)
                        try:
                            self.queue_out.put_nowait((frame_orig, frame_up))
                        except pyqueue.Full:
                            pass # 그래도 실패하면 프레임 드롭
                
                process_orig.wait()
                process_up.wait()

            except Exception as e:
                print(f"Error processing video {video_file}: {e}")
                continue # 다음 영상으로 넘어감
                
        self.queue_out.put(None)


class DetectParser(multiprocessing.Process):
    def __init__(self, queue_in, queue_out, gpu_id=0):
        multiprocessing.Process.__init__(self)
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.frame_cnt = 0
        self.model_cfgs = {
            k: {'cfg': v['cfg'], 'weights': v['weights'], 'names': v['names']}
            for k, v in model_cfgs.items()
        }
        self.gpu_id = gpu_id

        
    def __init__runtime(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

        # 모델 로드 (필요한 것만)
        for key, v in self.model_cfgs.items():
            self.model_cfgs[key] = Yolo(v['cfg'], v['weights'], v['names'])

        self.tracker_orig = ByteTrackLite(fps=30, 
                                    track_thresh=0.45, 
                                    low_thresh=0.1, 
                                    match_thresh=0.6,
                                    max_age=30, 
                                    min_hits=3)

        self.tracker_up = ByteTrackLite(fps=30, 
                                    track_thresh=0.45, 
                                    low_thresh=0.1, 
                                    match_thresh=0.6,
                                    max_age=30, 
                                    min_hits=3)

    def run(self):
        self.__init__runtime()
        while True:
            frame_pair = self.queue_in.get()
            if frame_pair is None:
                self.queue_out.put(None)
                break
            
            orig_frame, up_frame = frame_pair
            self.frame_cnt += 1
            if self.frame_cnt % 3 == 0:
                # 1. 사람 검출 및 추적
                det_o = self.model_cfgs.get(DETECT_MODEL).detect(orig_frame, 0.4, 0.5) if DETECT_MODEL in self.model_cfgs else []
                tracks_o, _ = self.tracker_orig.update(det_o or [], orig_frame)

                # 2. 쓰러짐 검출
                falldown_dets_o = []
                if 'falldown_v3' in self.model_cfgs:
                    falldown_dets_o = self.model_cfgs['falldown_v3'].detect(orig_frame, 0.4, 0.5)
                
                # 3. 사람 트랙에 쓰러짐 상태 매칭
                self.match_falldown_to_tracks(tracks_o, falldown_dets_o)

                # --- 업스케일 프레임도 동일하게 처리 ---
                det_u = self.model_cfgs.get(DETECT_MODEL).detect(up_frame, 0.4, 0.5) if DETECT_MODEL in self.model_cfgs else []
                tracks_u, _ = self.tracker_up.update(det_u or [], up_frame)
                
                falldown_dets_u = []
                if 'falldown_v3' in self.model_cfgs:
                    falldown_dets_u = self.model_cfgs['falldown_v3'].detect(up_frame, 0.4, 0.5)

                self.match_falldown_to_tracks(tracks_u, falldown_dets_u)
                
                # 디스플레이로 두 프레임과 트랙 전달
                try:
                    self.queue_out.put_nowait(((orig_frame, tracks_o), (up_frame, tracks_u)))
                except pyqueue.Full:
                    pass

    def match_falldown_to_tracks(self, tracks, falldown_dets, iou_threshold=0.3):
        if not falldown_dets:
            for track in tracks:
                track['falldown_status'] = {'label': 'standing', 'score': 0.0}
            return

        # xywh_c to xyxy 변환
        det_boxes = [xywh_c_to_xyxy(det[2]) for det in falldown_dets]
        
        for track in tracks:
            track['falldown_status'] = {'label': 'standing', 'score': 0.0} # 기본값
            track_box = track['bbox']
            
            best_iou = 0
            best_det = None
            for i, det_box in enumerate(det_boxes):
                iou = self.calculate_iou(track_box, det_box)
                if iou > best_iou:
                    best_iou = iou
                    best_det = falldown_dets[i]
            
            if best_iou > iou_threshold and best_det:
                track['falldown_status'] = {'label': best_det[0], 'score': best_det[1]}
    
    def calculate_iou(self, boxA, boxB):
        # ... (IoU 계산 함수, bytracker에서 가져오거나 간단히 구현) ...
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou


class DispEvent(multiprocessing.Process):
    def __init__(self, queue):
        multiprocessing.Process.__init__(self)
        self.queue = queue
        self.enable_crop_view = False
        self.color_cfgs = color_cfgs
        self.helmet_model = None
        self.falldown_model = None
        self.scores = {}
        self.label_hits = {} # UID별 라벨 카운트를 저장할 딕셔너리

    def run(self):
        # 헬멧 모델 로드
        helmet_cfg = model_cfgs.get('helmet_resort_v2')
        if helmet_cfg:
            self.helmet_model = Yolo(helmet_cfg['cfg'], helmet_cfg['weights'], helmet_cfg['names'])

        falldown_cfg = model_cfgs.get('falldown_v3')
        if falldown_cfg:
            self.falldown_model = Yolo(falldown_cfg['cfg'], falldown_cfg['weights'], falldown_cfg['names'])

        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        while True:
            data = self.queue.get()
            if data is None:
                break

            (frame_orig, results_orig), (frame_up, results_up) = data

            if frame_orig is not None:
                for track in results_orig or []:
                    uid = track['id']
                    class_id = track['label']
                    roi_frame = track['roi_frame']
                    ori_p_color = self.color_cfgs.get('person5l')
                    
                    if track.get('matched', True):
                        xmin, ymin, xmax, ymax = track['bbox']
                    else:
                        continue

                    helmet_results = self.check_helmet(roi_frame)
                    current_label, highest_score = 'detecting_helmet', 0
                    if helmet_results:
                        current_label, highest_score, _ = max(helmet_results, key=lambda x: x[1])

                    cv2.rectangle(frame_orig, (xmin, ymin), (xmax, ymax), ori_p_color, 2)
                    txt = '{} / {}'.format(str(class_id), str(uid))
                    cv2.putText(frame_orig, txt, (xmin, ymin-16), cv2.FONT_HERSHEY_SIMPLEX, 1, ori_p_color, 2)
                    
                    h_color = (0, 255, 0) if current_label == 'helmet' else (0, 0, 255)
                    cv2.putText(frame_orig, f"{current_label} ({highest_score:.2f})", (xmin, ymin-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, h_color, 2)
                
                    # 쓰러짐 상태 표시
                    if 'falldown_status' in track:
                        fd_status = track['falldown_status']
                        fd_label = fd_status['label']
                        fd_score = fd_status['score']
                        
                        fd_color = (50, 50, 255) if fd_label == 'falldown' else (200, 200, 200)
                        fd_txt = f"Fall: {fd_label} ({fd_score:.2f})"
                        cv2.putText(frame_orig, fd_txt, (xmin, ymin - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fd_color, 2)

                cv2.putText(frame_orig, 'ORIGINAL', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

            if frame_up is not None:
                for track in results_up or []:
                    uid = track['id']
                    class_id = track['label']
                    roi_frame = track['roi_frame']
                    up_p_color = self.color_cfgs.get('person5l')

                    if track.get('matched', True):
                        xmin, ymin, xmax, ymax = track['bbox']
                    else:
                        continue
                    
                    # 헬멧 판단 수행
                    helmet_results = self.check_helmet(roi_frame)
                    dominant_label, hit_count = self.update_and_get_dominant_label(uid, helmet_results)

                    
                    # 스코어링 시스템 업데이트
                    helmet_status, score = self.calc_score(uid, dominant_label, hit_count)

                    cv2.rectangle(frame_up, (xmin, ymin), (xmax, ymax), up_p_color, 2)
                    txt = '{} / {}'.format(str(class_id), str(uid))
                    cv2.putText(frame_up, txt, (xmin, ymin-16), cv2.FONT_HERSHEY_SIMPLEX, 1, up_p_color, 2)
                    
                    color_key = 'wearing_helmet' if helmet_status == 'wearing helmet' else \
                                'detecting_helmet' if helmet_status == 'detecting helmet' else \
                                'nohelmet'
                    h_color = self.color_cfgs.get(color_key, (255, 100, 255))
                    cv2.putText(frame_up, f"{helmet_status} ({score:.0f})", (xmin, ymin-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, h_color, 2)
                
                    # 쓰러짐 상태 표시
                    if 'falldown_status' in track:
                        fd_status = track['falldown_status']
                        fd_label = fd_status['label']
                        fd_score = fd_status['score']
                        
                        fd_color = (50, 50, 255) if fd_label == 'falldown' else (200, 200, 200)
                        fd_txt = f"Fall: {fd_label} ({fd_score:.2f})"
                        cv2.putText(frame_up, fd_txt, (xmin, ymin - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fd_color, 2)

                cv2.putText(frame_up, 'UPSCALED', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

            target_h = max(frame_orig.shape[0], frame_up.shape[0])
            disp_orig = self.pad_to_height(frame_orig, target_h)
            disp_up = self.pad_to_height(frame_up, target_h)

            stacked = np.hstack((disp_orig, disp_up))
            cv2.imshow('frame', stacked)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.enable_crop_view = not self.enable_crop_view

            time.sleep(0.001)
        
        cv2.destroyAllWindows()

    def pad_to_height(self, img, target_h):
        dh = target_h - img.shape[0]
        if dh <= 0:
            return img
        top = dh // 2
        bottom = dh - top
        return cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    def check_helmet(self, roi_frame):
        if roi_frame is None or roi_frame.size == 0:
            return []
        if self.helmet_model:
            return self.helmet_model.detect(roi_frame, 0.5, 0.5)
        return []

   
    def update_and_get_dominant_label(self, uid, results):
        if uid not in self.label_hits:
            self.label_hits[uid] = {}

        # 1. 현재 프레임의 최고 점수 라벨 찾기
        current_label = 'nohelmet'
        highest_score = 0
        if results:
            try:
                # 가장 높은 신뢰도의 검출 결과를 찾음
                current_label, highest_score, _ = max(results, key=lambda x: x[1])
            except (TypeError, ValueError):
                pass
        
        # 2. 라벨 카운트 업데이트
        self.label_hits[uid][current_label] = self.label_hits[uid].get(current_label, 0) + 1

        # 3. 지배적인 라벨 찾기
        if not self.label_hits[uid]:
            return 'nohelmet', 0
            
        dominant_label = max(self.label_hits[uid], key=self.label_hits[uid].get)
        hit_count = self.label_hits[uid][dominant_label]

        return dominant_label, hit_count

    def calc_score(self, uid, dominant_label, hit_count):
        if uid not in self.scores:
            self.scores[uid] = 0

        # 점수 산정 로직 수정: 지배적인 라벨과 그 카운트를 기반으로 점수 조정
        if dominant_label == 'helmet':
            # 헬멧으로 판단된 횟수가 많을수록 점수를 더 많이 올림
            self.scores[uid] = min(100, self.scores[uid] + 5 + int(hit_count / 5)) 
        else: # nohelmet 또는 다른 라벨
            self.scores[uid] = max(-100, self.scores[uid] - 10)

        score = self.scores[uid]
        if score >= 80:
             return 'wearing helmet', score
        elif -80 < score < 80:
             return 'detecting helmet', score
        else:
             return 'nohelmet', score


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    # 멀티프로세싱 큐 생성
    q_video = multiprocessing.Queue(maxsize=10) # 큐 사이즈 약간 늘림
    q_detect = multiprocessing.Queue(maxsize=10)

    # 1. 영상 로드 프로세스
    video_folder = './videos/'
    supported_formats = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.lower().endswith(supported_formats)]
    video_files.sort() # 파일 이름 순으로 정렬

    if not video_files:
        print(f"No video files found in '{video_folder}'")
    else:
        video_loader = VideoParser(video_files, q_video)
        video_loader.start()
        
        # 2. 객체 탐지 프로세스
        detector = DetectParser(q_video, q_detect, gpu_id=0)
        detector.start()

        # 3. 결과 표시를 위한 프로세스 생성 및 시작
        displayer = DispEvent(q_detect)
        displayer.start()

        try:
            # 표시 프로세스가 종료될 때까지 대기
            displayer.join()
        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다...")
        finally:
            # 모든 프로세스 종료
            if 'detector' in locals() and detector.is_alive():
                detector.terminate()
                detector.join(timeout=2)

            if 'video_loader' in locals() and video_loader.is_alive():
                video_loader.terminate()
                video_loader.join(timeout=2)
            
            # DispEvent는 join으로 이미 기다렸거나, 여기서 확실히 종료
            if 'displayer' in locals() and displayer.is_alive():
                displayer.terminate()
                displayer.join(timeout=2)