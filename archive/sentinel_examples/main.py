'''
검출에 고도화를 진행하고 싶다면
메인 코드에서 후 처리 알고리즘을 고도화하는 방법도 존재하지만
추론 및 매칭하는 iou_tracker 부분도 고도화하는 방법도 있습니다.
'''
'''
1. 다음 영상 재생 check_helmet
2. roi 프레임 없을경우 예외처리 run()
3. no_helmet일 경우 한번더 다른 모델을 이용하여 검사 진행
'''

from lib.init import *
from lib.yolo import Yolo
from lib.iou_tracker import *
import threading
import cv2
import time
import os

# 지역 변수 처리(검출된 영상)
class VideoParser(threading.Thread):
    def __init__(self, video_files):  # 유알엘은 비디오소스를 지정, 기본값은 0
        threading.Thread.__init__(self)
        self.video_files = video_files
        # self.url = url
        self.cap = None
        self.frame = None
        self.frame_cnt = 0
        self.results = []
        self.unmatch_list = []
        self.tracker = ObjectTracker()
        self.scores = {}  # 각 객체의 신뢰도를 저장할 딕셔너리
        self.helmet_threshold_high = 94  # 헬멧 착용 여부를 판단하는 상한 점수 기준
        self.helmet_threshold_low = 6  # 헬멧 착용 여부를 판단하는 하한 점수 기준

    # 영상 반환
    def run(self):
        for video_file in self.video_files:
            self.cap = cv2.VideoCapture(video_file)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            fps_split = 1 / (fps + 20)
            while True:
                r, self.frame = self.cap.read()
                # 영상을 못읽을 경우 while 탈출 다음 영상을 가져오는 효과
                if not r:
                    break
                if r:
                    self.frame_cnt += 1
                    # 30 frame을 다 검사하는 경우 메모리 이슈가 발생함
                    # 전체 프레임 중 3프레임씩 끊어서 검사를 진행
                    if self.frame_cnt % 3 == 0:
                        # 사람 검출
                        results = yolo_model['person5l'].detect(self.frame, 0.5, 0.5)
                        self.results, self.unmatch_list = self.tracker.update(results)

                time.sleep(fps_split)
            self.cap.release()

    # 검사 결과에 대한 라벨, 점수 값 표출 함수
    def get_most_label(self, results):
        most_label = ''
        most_acc = 0
        try:
            for motion_track in results:
                motion_label = motion_track[0]
                if float(motion_track[1]) >= most_acc:
                    most_label = motion_label
                    most_acc = float(motion_track[1])
        except:
            pass
        return most_label, most_acc

    def check_helmet(self, roi_frame):
        if roi_frame is None or roi_frame.size == 0:
            return 'no helmet', 0  # 빈 이미지인 경우 기본값 반환
        r2 = yolo_model['helmet_resort_v2'].detect(roi_frame, 0.5, 0.5)
        return self.get_most_label(r2)
    
    def check_helmet2(self, roi_frame):
        if roi_frame is None or roi_frame.size == 0:
            return 'no helmet', 0  # 빈 이미지인 경우 기본값 반환
        r2 = yolo_model['helmet_resort_my'].detect(roi_frame, 0.5, 0.5)
        return self.get_most_label(r2)

    # 객체의 헬멧 착용 여부를 판단하는 스코어링 시스템
    def update_scores(self, uid, label, acc):
        if uid not in self.scores:
            self.scores[uid] = 50  # 처음 객체가 등장할 때 점수를 50으로 설정

        # 헬멧을 착용했다고 탐지된 경우 점수를 5점 추가
        if label == 'helmet' and acc > 0.5:
            self.scores[uid] = min(100, self.scores[uid] + 5)  # 최대 점수는 100
        else:
            self.scores[uid] = max(0, self.scores[uid] - 5)  # 헬멧이 감지되지 않으면 점수 5점 감소

        # 점수에 따른 헬멧 착용 여부 반환
        if self.scores[uid] > self.helmet_threshold_high:
            return 'helmet', self.scores[uid]
        elif self.helmet_threshold_low < self.scores[uid] < self.helmet_threshold_high:
            return 'detecting helmet', self.scores[uid]  # 점수가 6점에서 94점 사이일 때
        else:
            return 'no helmet', self.scores[uid]

# 실행
if __name__ == '__main__':
    # Yolo Model Load
    yolo_model = {}

    ''' 다중 모델 사용을 위한 for문 사용 '''
    for key, value in models.items():
        yolo_model[key] = Yolo(value['cfg'], value['weights'], value['data'])
    
    # 비디오 파일 경로
    video_folder = './video/'
    # 비디오 파일 이름(mp4 만)
    video_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder)]
    
    # 영상로드 및 메인 스레드 시작
    player = VideoParser(video_files)
    player.start()

    # 검출된 영상의 정보를 이용하여 화면에 표출
    while True:
        if player.frame is not None:
            frame = player.frame.copy()
            
            for track in player.results:
                uid = track['id']
                class_id = track['label']
                xmin, ymin, xmax, ymax = track['bbox']

                # 헬멧을 검사하기 위한 target crop image
                roi_frame = frame[ymin:ymax, xmin:xmax].copy()
                
                # 크롭된 이미지에서 헬멧 검출
                most_label, most_acc = player.check_helmet(roi_frame)

                # 스코어링 시스템 업데이트
                helmet_status, score = player.update_scores(uid, most_label, most_acc)

                # 헬멧 미착용으로 판단된 경우, 추가 검사 수행
                if helmet_status == 'no helmet':
                    most_label, most_acc = player.check_helmet2(roi_frame)
                    helmet_status, score = player.update_scores(uid, most_label, most_acc)

                # 사람 바운딩 박스 그리기
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                cv2.putText(frame, 'person', (xmin, ymin - 16), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # 헬멧 착용여부 표출
                cv2.putText(frame, str(helmet_status), (xmin, ymin - 46), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            frame = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5)
            cv2.imshow('frame', frame)
        
        cv2.waitKey(1)
        time.sleep(0.01)

    cv2.destroyAllWindows()



    
