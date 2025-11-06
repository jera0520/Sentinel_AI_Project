import cv2
import multiprocessing
import time
import numpy as np
import ffmpeg
import queue as pyqueue

from lib.yolov4 import Yolo
from lib.bytracker import ByteTrackLite
from lib.upscale_new import build_filter_graph


class VideoParser(multiprocessing.Process):
    def __init__(self, url, queue_out):
        multiprocessing.Process.__init__(self)
        self.url = url
        self.queue_out = queue_out

        self.width, self.height = 0, 0
    # 영상 반환
    def run(self):
        probe = ffmpeg.probe(self.url)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        if video_stream is None:
            print(f"No video stream found in {self.url}")
            self.queue_out.put(None)
            return
        
        self.width = int(video_stream['width'])
        self.height = int(video_stream['height'])

        # 원본용 스트림과 업스케일용 스트림을 각각 생성
        in_stream_orig = ffmpeg.input(self.url, **{'re': None}, threads=0)
        in_stream_up = ffmpeg.input(self.url, **{'re': None}, threads=0)
        
        # 업스케일 파이프라인 구성 (build_filter_graph 사용)
        vf, out_w, out_h = build_filter_graph(in_stream_up, self.width, self.height, scale_factor=1.25, keep_ar=True, preset='balanced')
        
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
            in_bytes_orig = process_orig.stdout.read(self.width * self.height * 3)
            in_bytes_up = process_up.stdout.read(out_w * out_h * 3)
            if not in_bytes_orig or not in_bytes_up:
                break
            
            frame_orig = np.frombuffer(in_bytes_orig, np.uint8).reshape(self.height, self.width, 3)
            frame_up = np.frombuffer(in_bytes_up, np.uint8).reshape(out_h, out_w, 3)
            try:
                self.queue_out.put_nowait((frame_orig, frame_up))
            except pyqueue.Full:
                continue
                
        self.queue_out.put(None)
        process_orig.wait()
        process_up.wait()


class DetectParser(multiprocessing.Process):
    def __init__(self, queue_in, queue_out):
        multiprocessing.Process.__init__(self)
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.models = {}
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
        self.add_model()
        frame_cnt = 0
        
        while True:
            frame_pair = self.queue_in.get()
            if frame_pair is None:
                self.queue_out.put(None)
                break
            
            orig_frame, up_frame = frame_pair
            frame_cnt += 1
            if frame_cnt % 3 == 0:
                # Detect only persons for tracking
                dets_p_orig = []
                if 'person5l' in self.models:
                    dets_p_orig = self.models['person5l'].detect(orig_frame, 0.5, 0.5)
                tracks_orig, del_idx_o = self.tracker_orig.update(dets_p_orig or [], orig_frame)
                del del_idx_o

                # Upscaled frame: detect only persons for tracking
                dets_p_up = []
                if 'person5l' in self.models:
                    dets_p_up = self.models['person5l'].detect(up_frame, 0.5, 0.5)
                tracks_up, del_idx_u = self.tracker_up.update(dets_p_up or [], up_frame)
                del del_idx_u
                # 디스플레이로 두 프레임과 트랙 전달
                try:
                    self.queue_out.put_nowait(((orig_frame, tracks_orig), (up_frame, tracks_up)))
                except pyqueue.Full:
                    pass
            

    def add_model(self):
        # main.py와 유사한 다중 모델 로딩 방식
        model_defs = {
            'person5l': {
                'cfg': 'model/person5l/model.cfg',
                'weights': 'model/person5l/model.weights',
                'names': 'model/person5l/model.names'
            },
            'helmet_resort_v2': {
                'cfg': 'model/helmet_resort_v2/model.cfg',
                'weights': 'model/helmet_resort_v2/model.weights',
                'names': 'model/helmet_resort_v2/model.names'
            }
        }
        self.models = {}
        for key, value in model_defs.items():
            self.models[key] = Yolo(value['cfg'], value['weights'], value['names'])
        

class DispEvent(multiprocessing.Process):
    def __init__(self, queue):
        multiprocessing.Process.__init__(self)
        self.queue = queue
        self.enable_crop_view = False

    def _crop_and_show(self, frame, bbox, win_name='crop'):
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[0], frame.shape[1]
        x1 = max(0, min(int(x1), w - 1))
        x2 = max(0, min(int(x2), w))
        y1 = max(0, min(int(y1), h - 1))
        y2 = max(0, min(int(y2), h))
        if x2 <= x1 or y2 <= y1:
            return
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.imshow(win_name, crop)

    def run(self):
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
                    color = track['color']
                    
                    if track.get('matched', True):
                        xmin, ymin, xmax, ymax = track['bbox']
                        
                    else:
                        xmin, ymin, xmax, ymax = track.get('disp_bbox', track['bbox'])
                        color = (0, 255, 255)

                    cv2.rectangle(frame_orig, (xmin, ymin), (xmax, ymax), color, 2)
                    txt = '{} / {}{}'.format(str(class_id), str(uid), '' if track.get('matched', True) else f"  miss:{track.get('missed', 0)}")
                    cv2.putText(frame_orig, txt, (xmin, ymin-16), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                cv2.putText(frame_orig, 'ORIGINAL', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

            if frame_up is not None:
                for track in results_up or []:
                    uid = track['id']
                    class_id = track['label']
                    color = track['color']
                    
                    if track.get('matched', True):
                        xmin, ymin, xmax, ymax = track['bbox']
                        
                    else:
                        xmin, ymin, xmax, ymax = track.get('disp_bbox', track['bbox'])
                        color = (0, 255, 255)

                    cv2.rectangle(frame_up, (xmin, ymin), (xmax, ymax), color, 2)
                    txt = '{} / {}{}'.format(str(class_id), str(uid), '' if track.get('matched', True) else f"  miss:{track.get('missed', 0)}")
                    cv2.putText(frame_up, txt, (xmin, ymin-16), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                cv2.putText(frame_up, 'UPSCALED', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

            # 원본 해상도를 유지하고, 높이를 맞추기 위해 패딩 적용 후 좌우 비교 출력
            target_h = max(frame_orig.shape[0], frame_up.shape[0])
            def pad_to_height(img, target_h):
                dh = target_h - img.shape[0]
                if dh <= 0:
                    return img
                top = dh // 2
                bottom = dh - top
                return cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            disp_orig = pad_to_height(frame_orig, target_h)
            disp_up = pad_to_height(frame_up, target_h)

            # 선택적으로 crop 미리보기
            if self.enable_crop_view:
                if results_orig and len(results_orig) > 0:
                    t = results_orig[0]
                    b = t['bbox'] if t.get('matched', True) else t.get('disp_bbox', t['bbox'])
                    self._crop_and_show(frame_orig, b, 'crop_orig')
                if results_up and len(results_up) > 0:
                    t = results_up[0]
                    b = t['bbox'] if t.get('matched', True) else t.get('disp_bbox', t['bbox'])
                    self._crop_and_show(frame_up, b, 'crop_up')

            stacked = np.hstack((disp_orig, disp_up))
            cv2.imshow('frame', stacked)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.enable_crop_view = not self.enable_crop_view

            time.sleep(0.001)
        
        cv2.destroyAllWindows()

# 실행
if __name__ == '__main__':
    # 멀티프로세싱 큐 생성
    q_video = multiprocessing.Queue(maxsize=2)
    q_detect = multiprocessing.Queue(maxsize=2)

    # 1. 영상 로드 프로세스
    video_loader = VideoParser('test.mp4', q_video)
    video_loader.start()

    # 2. 객체 탐지 프로세스
    detector = DetectParser(q_video, q_detect)
    detector.start()

    # 3. 결과 표시를 위한 프로세스 생성 및 시작
    displayer = DispEvent(q_detect)
    displayer.start()

    # 표시 프로세스가 종료될 때까지 대기 (사용자가 'q'를 누르거나 영상이 끝났을 때)
    displayer.join()

    # 표시 프로세스가 종료되면, 다른 프로세스도 종료
    if detector.is_alive():
        detector.terminate()
        detector.join()

    if video_loader.is_alive():
        video_loader.terminate()
        video_loader.join()


    
