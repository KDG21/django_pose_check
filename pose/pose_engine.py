import cv2
import os
import pandas as pd
import re
import shutil
import json


class Pose_Check:
    def __init__(self, prototext_path, caffe_path):
        # 필요 모델의 경로를 받아옵니다.
        self.protoFile = prototext_path
        self.weightsFile = caffe_path

        # 기초 설정 cv2.dnn 에 모델을 입힙니다.
        self.net = cv2.dnn.readNetFromCaffe(self.protoFile, self.weightsFile)

    # 영상 필요 파일을 저장할 디렉토리를 만들어준다.
    def set_dir(self, path):
        try:
            os.mkdir(path)
        except:
            pass

    # 영상 필요 파일을 저장한 디렉토리를 삭제해준다.
    def del_dir(self, path):
        try:
            shutil.rmtree(path)
        except:
            pass

    # 영상 파일을 삭제해준다.
    def del_file(self, path):
        try:
            os.remove(path)
        except:
            pass

    # GPU 연결
    def set_gpu(self, gpu=False):
        net = self.net
        if gpu:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # 디렉토리에 있는 폴더 중 지정한 파일이름을 통해 사진 파일만 찾아주는 메소드
    def find_name(self, tuple_type):
        if tuple_type[0].startswith(tuple_type[1]):
            return tuple_type[0]
        else:
            return None

    # 자세를 검사하기 위해 두 점의 기울기를 계산하는 코드
    def isHorizontal(self, pointA, pointB):

        # 두 점 중에 하나라도 모델에서 검출을 못했다면 None 타입 반환
        if pointA == None or pointB == None:
            return None

        # 정상적으로 모델에서 점을 찍었다면 devision by zero가 나타날 수 없지만, 인체가 아닌 다른 곳에서 점을 찍을 경우 중복된 곳이나
        # 다른 곳에 점을 찍는 경우가 생겨서 예외처리를 하고 이 경우 None을 반환하도록 하였습니다.
        try:
            horizon = (pointB[1] - pointA[1] )/ (pointB[0] - pointA[0])
        except:
            return None

        # 기울기가 0.5 이상으로 잡히는 경우 거의 대부분이 인체가 아닌 다른 곳에 점을 찍을 때 뿐이었습니다. 따라서 특정 임계값 이상의 기울기가
        # 찍히면  None를 반환하도록 하였습니다.
        if horizon >= 0.5 or horizon <= -0.5:
            return None
        elif horizon <= 0.125 and horizon >= -0.125:
            return 0
            # return horizon #디버깅용 반환 코드
        else:
            return 1


    def isVertical(self, pairA, pairB):
        # 정해진 위치에 점을 찍지 못했다면 None를 반환
        for i, j in zip(pairA, pairB):
            if i == None or j == None:
                return None
            elif None in i or None in j:
                return None

        # horizon과 같은 이유로 예외처리를 진행하였습니다.
        try:
            verticalA = (pairA[0][0] - pairA[1][0]) / (pairA[0][1] - pairA[1][1])
            verticalB = (pairB[0][0] - pairB[1][0]) / (pairB[0][1] - pairB[1][1])

        except:
            return None

        # horiznon과 같은 이유로 예외처리를 진행하였습니다.
        if verticalA >= 0.5 or verticalA <= -0.5:
            return None
        elif verticalB >= 0.5 or verticalB <= -0.5:
            return None
        elif -0.1 <= verticalA <= 0.1 and -0.1 <= verticalB <= 0.1:
            return 0
            # return (verticalA, verticalB) # 디버깅용 코드
        else:
            return 1

    def get_frame_list(self, path, file_name): # 변형 할 수도 있음

        # os.walk를 통해 상위 디렉토리와 현재디렉토리 안의 파일들을 제너레이터로 일단 반환 받습니다.
        # 리스트에 튜플 형태로 들어와 있으며, 튜플의 0번에 상위 디렉토리가 할당되어 있고, 2번에 현재 파일 목록이 할당 되어 있습니다.
        img_list_walk = list(os.walk(path))

        # file name과 상위 디렉토리를 합치기 위해, file_name의 길이를 파일 목록과 같은 길이로 바꾸어 줍니다.
        name_length = [file_name] * len(img_list_walk[0][2])

        # 정규표현식을 통해 이미지의 번호순으로 리스트를 정렬해줍니다.
        img_list_re = sorted(img_list_walk[0][2], key=lambda x: int(x[re.search('\d+', x).span()[0]: re.search('\d+', x).span()[1]]), reverse = False)

        # find_name 메소드를 통해 file_name이 있는 파일들만 반환 받습니다.
        img_list = list(map(self.find_name, zip(img_list_re, name_length)))

        # file_name과 상위 디렉토리 경로를 합쳐줍니다.
        img_list = list(map(lambda x: os.path.join(img_list_walk[0][0], x), img_list))

        # 판다스로 리스트를 바꿔준후에 혹시 있을 지도 모를 nan 값을 지워줍니다.
        img_list = pd.Series(img_list)
        img_list = img_list.dropna()
        return img_list

    def estimation(self, img):
        net = self.net

        frame = cv2.imread(img)

        # blur를 통해 프레임 이미지를 전처리
        frame = cv2.bilateralFilter(frame, 5, 75, 75)

        height, width = frame.shape[:2]

        # 모델이 집어넣기전 압축하는 과정 이미지의 크기를 이미지의 비율에 맞게 조정하여 집어넣어줘야 합니다.
        # swapRB = TRUE는 BGR을 RGB로 바꾸는 설정입니다.
        in_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (640, 360), (0, 0, 0), swapRB=True, crop=False)

        # 모델에 blob을 집어넣습니다.
        net.setInput(in_blob)

        # 1D = Image_id, 2D = Confidence Maps and Part Affinity maps, 3D = height of output map, 4D = width
        output = net.forward()

        H_out = output.shape[2]
        W_out = output.shape[3]

        points = []

        for i in range(18):

            # confidence map of corresponding body's part
            probMap = output[0, i, :, :]

            # find global maximam of the probmap
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            x = (width * point[0]) / W_out
            y = (height * point[1]) / H_out

            if prob > 0.1:
                cv2.circle(frame, (int(x), int(y)), 15, (255, 255, 255), thickness=-1, lineType=cv2.FILLED)
                # cv2.putText(frame, f'{i}', (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, lineType=cv2.LINE_AA)
                # add the point to the list if probability is greater than the threshold
                points.append((int(x), int(y)))
            else:
                points.append(None)

        POSE_PAIRS_COCO = [[1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],
                          [1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

        for pair in POSE_PAIRS_COCO:
            partA = pair[0]
            partB = pair[1]

            if points[partA] and points[partB]:
                cv2.line(frame, points[partA], points[partB], (255, 0, 0), 2)

        return points, frame

    # 엔진을 실행시키는 함수
    def pose_run(self, MEDIA_ROOT, video_file):
        pose = Pose_Check(
            '/home/ubuntu/model/pose_deploy_linevec.prototxt',
            '/home/ubuntu/model/pose_iter_440000.caffemodel',
        )

        # dnn gpu 가속 설정
        pose.set_gpu(gpu=True)

        # 초당 한 프레임만 가져와서 저장하기 위한 코드
        n = 0
        idx = 0
        video = cv2.VideoCapture(MEDIA_ROOT+video_file)
        name = 'frame'
        frame_path = MEDIA_ROOT+'frames'
        converted_path = MEDIA_ROOT+'new_img'

        # 프레임 저장할 디렉토리 설정
        pose.set_dir(frame_path)

        # 영상의 초당 프레임을 갖고옴
        FPS = int(video.get(cv2.CAP_PROP_FPS))

        # 영상을 재생하면서 저장
        while (video.isOpened()):
            ret, frame = video.read()
            if ret == False:
                break
            n += 1
            idx += 1

            # 1초의 마지막 프레임에서 프레임 저장
            if n == FPS:
                cv2.imwrite(frame_path + '/{}{}.jpg'.format(name, idx), frame)
                # 판별 코드 초기화
                n = 0
            if cv2.waitKey(1) & 0xff == 27:
                break

        video.release()
        cv2.destroyAllWindows()

        pose.set_dir(converted_path)

        frame_list = pose.get_frame_list(frame_path, name)

        sh_count = []
        eye_count = []
        pel_count = []
        n = 0

        for i in frame_list:
            n += 1
            points, frame = pose.estimation(i)
            cv2.imwrite(i.replace(frame_path, converted_path),frame)
            shoulder = pose.isHorizontal(points[2], points[5])
            eye = pose.isHorizontal(points[14], points[15])
            sh_count.append(shoulder)
            eye_count.append(eye)

            if pose.isVertical((points[8], points[9]), (points[11], points[12])) == None or pose.isHorizontal(points[8], points[11]) == None:
                pelvis = None
                pel_count.append(pelvis)
            elif pose.isVertical((points[8], points[9]), (points[11], points[12])) == 1 or pose.isHorizontal(points[8], points[11]) == 1:
                pelvis = 1
                pel_count.append(pelvis)
            else:
                pelvis = 0
                pel_count.append(pelvis)


        # 리스트를 데이터프레임으로 변환
        sh_count = pd.Series(sh_count, name='sh_count', dtype='float')
        pel_count = pd.Series(pel_count, name='pel_count', dtype='float')
        eye_count = pd.Series(eye_count, name='eye_count', dtype='float')
        all_count = pd.concat([sh_count, pel_count, eye_count], axis=1)

        pose_count = []

        for i in range(len(all_count)):
            if 1 in all_count.iloc[[i]].values:
                pose_count.append(1.0)
            elif 1 not in all_count.iloc[[i]].values and 0 in all_count.iloc[[i]].values:
                pose_count.append(0.0)
            elif all_count.iloc[[i]].all(None):
                pose_count.append(None)

        pose_count = pd.Series(pose_count, name='pose_count', dtype='float')
        all_count = pd.concat([all_count, pose_count], axis=1)

        # 값들을 json형식으로 변환
        pose_json = {'shoulder':{
                'len':len(all_count),
                'balance':len(all_count) - (all_count['sh_count'].sum() + all_count['sh_count'].isnull().sum()),
                'unbalance':all_count['sh_count'].sum(),
                'none':all_count['sh_count'].isnull().sum()
            },
            'pelvis' :{
                'len':len(all_count),
                'balance':len(all_count) - (all_count['pel_count'].sum() + all_count['pel_count'].isnull().sum()),
                'unbalance':all_count['pel_count'].sum(),
                'none':all_count['pel_count'].isnull().sum()
            },
            'eye' :{
                'len':len(all_count),
                'balance':len(all_count) - (all_count['eye_count'].sum() + all_count['eye_count'].isnull().sum()),
                'unbalance':all_count['eye_count'].sum(),
                'none':all_count['eye_count'].isnull().sum()
            },
            'all_pose':{
                'len':len(all_count),
                'balance':len(all_count) - (all_count['pose_count'].sum() + all_count['pose_count'].isnull().sum()),
                'unbalance':all_count['pose_count'].sum(),
                'none':all_count['pose_count'].isnull().sum()
            }}
        pose_json = json.dumps(str(pose_json))

        # 값을 구한 뒤 저장된 파일, 폴더 삭제
        pose.del_dir(MEDIA_ROOT)

        return pose_json