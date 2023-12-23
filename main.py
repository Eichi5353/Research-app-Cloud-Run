import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from flask import Flask, jsonify,request

#ここから下のどれかにエラー要素あり installに問題があるだろう
import cv2
import base64
import numpy as np
import itertools

import torch.nn as nn
import torch
from PIL import Image
from torchvision import transforms
import json
import requests

cred = credentials.Certificate('./research-h-firebase-adminsdk-i9dtx-12a1568005.json')
app = firebase_admin.initialize_app(cred)

# Siamse Networkモデルクラス .marでは必要ない？？
class SiameseModel(nn.Module):
    def __init__(self):
        super(SiameseModel, self).__init__()
        self.flatten = nn.Flatten()

        self.encoder = nn.Sequential(
            nn.Linear(224*224, 128),#チャンネル数に応じて変更？
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )
    def get_cnn_output_size(self, shape):
        x = torch.randn(1, *shape)
        x = self.cnn(x)
        return x.view(1, -1).size(1)   
    def forward_once(self, x):
        x = self.flatten(x)
        z = self.encoder(x)
        return z
    def forward(self, x1, x2):
        z1 = self.forward_once(x1)
        z2 = self.forward_once(x2)
        return z1, z2

model = SiameseMnistModel().to(device)

db = firestore.client()

app = Flask(__name__)

@app.route("/calculate-siamese",methods=['POST'])
def calculate_siamese():
    print("calculate siamese")
    img1 = request.form["img1"]
    img2 = request.form["img2"]
    decode_img1 =decode(img1)
    decode_img2 =decode(img2)
    # 画像を処理する前に正規化や変換が必要な場合は行います
    transform = transforms.Compose([
        transforms.Resize((224, int(224 * 4 / 3))),
        transforms.CenterCrop((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # query_img = Image.open(decode_img1)
    query_img = Image.fromarray(cv2.cvtColor(decode_img1, cv2.COLOR_BGR2RGB))
    # valid_img = Image.open(decode_img2)
    valid_img = Image.fromarray(cv2.cvtColor(decode_img2, cv2.COLOR_BGR2RGB))
    trans_query_img = transform(query_img)
    trans_valid_img = transform(valid_img)
#再びbase64に変換
    base64_query = base64.b64ecode(trans_query_img.tobytes()).decode("utf-8")
    base64_valid = base64.b64ecode(trans_valid_img.tobytes()).decode("utf-8")

    input_data = {
        "instances": [
            {"input": base64_query},
            {"input": base64_valid}
        ]
    }


    # エンドポイントのURL
    # endpoint_url = "https://REGION-aiplatform.googleapis.com/v1/projects/PROJECT_ID/locations/REGION/endpoints/ENDPOINT_ID:predict"
    endpoint_url = "https://region-aiplatform.googleapis.com/v1/projects/research-h/locations/asia-northeast1/endpoints/4393569299755696128:predict"

    # APIリクエスト
    response = requests.post(endpoint_url, json=input_data)

    if response.status_code == 200:
        result = response.json()
        print(result)
        return result
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return response.text



#internal error!
@app.route("/get-data",methods=['POST'])
def show_data():
    print("get-data")
    collection_name = request.form["collection_name"]
    ref = db.collection(collection_name)
    docs = ref.stream()
    data = []
    for doc in docs:
        try:
            user_data = {
                "name": doc.id,
                "point": doc.get("totalPoint")
            }
        except KeyError as e:
            # "totalPoint"が存在しない場合の処理
            user_data = {
                "name": doc.id,
                "point": 0  # または別のデフォルト値を設定
            }
        data.append(user_data)
        print(u'{} => {}'.format(doc.id, doc.to_dict()))
    response = jsonify({"user_data": data})
    #response.headers['Content-Type'] = 'application/json'  # Content-Typeを設定
    print("response: ",response)
    return response

@app.route("/user-data",methods=['GET'])
def show_user_data():
    print("user-data GET")
    ref = db.collection("users")
    docs = ref.stream()
    data = []
    for doc in docs:
        user_data = {
            "name": doc.id,
            "point": doc.get("totalPoint")
        }
        data.append(user_data)
        print(u'{} => {}'.format(doc.id, doc.to_dict()))
    return jsonify({"user_data": data})

@app.route('/calculate-similarity', methods=['POST'])
def extract_img():
    print(request.url)
    img1 = request.form["img1"]
    img2 = request.form["img2"]

    #ここを画像にしてみる
    #ここでは値を得るだけ
    #print("img中身は",str(img1))
    #print("img中身は",str(img2))

    print("decode")
    decode_img1 =decode(img1)
    decode_img2 =decode(img2)


    print("procedure")
    q_img=resize(decode_img1)
    t_img=resize(decode_img2)

    #AKZE
    similarity,good_matches=match(q_img,t_img)
    result=round(culcScore2(similarity))
    print("akaze:",result,"  similarity:",similarity)
    print("good matches",good_matches)
    result_matches = 0
    if(good_matches>20 and result<50):
        result+=30
    elif(good_matches>=5 and good_matches<20):
        result_matches=good_matches*3
    
    
    print("good matches result",result_matches)
    print("akaze result final",result)
    print("変えた後")

    #return str(result)

    #print("main")
    print("HSV")
    
    q_hsv,q_d_img,q_array,vq_array=decreaseColor(q_img)
    q_hist=np.histogram(q_array)

    t_hsv,t_d_img,t_array,vt_array=decreaseColor(t_img)
    t_hist=np.histogram(t_array)

    #ヒストグラム型変換->npArray
    q_array = np.array(list(q_array), np.float32)
    t_array = np.array(list(t_array), np.float32)
    
    vq_array = np.array(list(vq_array), np.float32)
    vt_array = np.array(list(vt_array), np.float32)
    
    c_result1=cv2.compareHist(q_array, t_array, cv2.HISTCMP_BHATTACHARYYA)
    #print("Hの類似度は",c_result)
    v_result=cv2.compareHist(vq_array, vt_array, cv2.HISTCMP_BHATTACHARYYA)
    #print("Vの類似度は",v_result)
    
    mix_result = (16*c_result1 +4*v_result)/20
    print(mix_result)
   
    mix_result_score =round(culcScore(0.20, mix_result))
    c_result=round(culcScore(0.2,c_result1))
    print("H得点",c_result,"Hの類似度は",c_result)

    print("総得点は",mix_result_score)

    largest = max(result,result_matches, mix_result_score,c_result)
    print("最終",largest)
    if(good_matches==0):
        return str(0)
    return str(largest)
    #img_size = test(decode_img)
    #return str(mix_result_score) #str(img);#str(img_size);
    
    
    

def decode(data1):
    #data1
    #base64型から読めるString型へ
    data1 += "" * ((4 - len(data1) % 4) % 4)  
    #print(data1)
    decode_data1= base64.urlsafe_b64decode(data1)
    #np配列に変換
    np_data1 = np.fromstring(decode_data1,np.uint8)
    #Arrayを読み込んでimgにdecode
    img1 = cv2.imdecode(np_data1,cv2.IMREAD_UNCHANGED)
    #print(img1)
    return img1

def resize(img):
    height = img.shape[0]
    width = img.shape[1]
    w=500
    h=round(w/width*height)
    dst = cv2.resize(img,dsize=(w,h))
    return dst

def decreaseColor(img):
    #hsv = rgb_to_hsv(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    
    #平滑化
    c=40
    m=128
    v = (v-np.mean(v)) / np.std(v) * c + m
    v = np.array(v, dtype=np.uint8) # 配列のdtypeをunit8に戻す
    
    idx = np.where((0<=h) & (h<8))
    h[idx] =0
    idx = np.where((8<=h) & (h<26))
    h[idx] =17
    idx = np.where((26<=h) & (h<42))
    h[idx] =34
    idx = np.where((42<=h) & (h<58))
    h[idx] =51
    idx = np.where((58<=h) & (h<74))
    h[idx] =68
    idx = np.where((74<=h) & (h<90))
    h[idx] =85
    idx = np.where((90<=h) & (h<106))
    h[idx] =102
    idx = np.where((106<=h) & (h<122))
    h[idx] =119
    idx = np.where((122<=h) & (h<138))
    h[idx] =130
    idx = np.where((138<=h) & (h<154))
    h[idx] =146
    idx = np.where((154<=h) & (h<170))
    h[idx] =162
    idx = np.where((170<=h) & (h<186))
    h[idx] =178
    idx = np.where((186<=h) & (h<202))
    h[idx] =194
    idx = np.where((202<=h) & (h<218))
    h[idx] =210
    idx = np.where((218<=h) & (h<234))
    h[idx] =226
    idx = np.where((234<=h) & (h<250))
    h[idx] =242
    idx = np.where((250<=h) & (h<255))
    h[idx] =255

    h[idx] = np.round(h[idx]*(15/256))
    idx = np.where((0<=s) & (s<43))    
    s[idx] = 0
    v[idx] = 0
    idx = np.where((43<=s) & (s<128))    
    s[idx] = 85
    v[idx] = 85
    idx = np.where((128<=s) & (s<213))    
    s[idx] = 170
    v[idx] = 170
    idx = np.where((213<=s) & (s<255))   
    s[idx] = 255
    v[idx] = 255

    hh=list(itertools.chain.from_iterable(h))

    ss=list(itertools.chain.from_iterable(s))
    vv=list(itertools.chain.from_iterable(v))

    img_size = img.shape[0]*img.shape[1]
    c_hist_array = np.zeros(256)
    v_hist_array = np.zeros(256)
    for idx in range(img_size):
        s_thre = 255 - 0.8*vv[idx]/255
        if ss[idx] >= 126:#64:
            c_hist_array[hh[idx]] = c_hist_array[hh[idx]]+1
            #print(ss[idx])
        elif ss[idx] < 126:#64:
            v_hist_array[vv[idx]] = v_hist_array[vv[idx]]+1
            #print(ss[idx2])
    hsvs=cv2.merge((h,s,v))
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return hsvs,img,c_hist_array,v_hist_array

def culcScore(color_thres:float,color_score:float):
    thres_score = 60
    zero_border = 0.45
    
    if color_score < color_thres and color_score >= 0:
        color_result = -((thres_score-100)/-color_thres**2)*color_score**2+100
    elif color_score > color_thres and color_score <= zero_border:
        color_result =(thres_score/(color_thres-zero_border)**2)* (color_score-zero_border)**2
    else:
        color_result = 0
    color_result=round(color_result)
    #char_result=round(char_result,2)
    return color_result

def culcScore2(color_score:float):
    #thres_score = 60
    #zero_border = 0.15
    
    if color_score > 0.15 and color_score <= 0.25:
        result = 900*color_score-135
    elif(color_score>0.25):
        result = 95
    else:
        result=0
    result=round(result)
    #char_result=round(char_result,2)
    return result
def match(q_img, t_img):
    
    q_img = cv2.cvtColor(q_img, cv2.COLOR_RGB2GRAY)
    t_img = cv2.cvtColor(t_img, cv2.COLOR_RGB2GRAY)
    
    # 各画像の特徴点を取る
    akaze = cv2.AKAZE_create()
    q_key_points, q_descriptions = akaze.detectAndCompute(q_img, None)
    t_key_points, t_descriptions = akaze.detectAndCompute(t_img, None)
    
    # 2つの特徴点をマッチさせる
    bf_matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, True)
    matches = bf_matcher.match(q_descriptions, t_descriptions)
    
    good_matches = [[good] for good in matches if good.distance < 75]
    print("特徴点の数",t_key_points)
    #if(len(t_key_points)>300):
     #   t_key_points=300
    similarity = len(good_matches) / len(t_key_points)
    return similarity,len(good_matches)

@app.route("/hello")
def hello_world():
    """Example Hello World route."""
    name = os.environ.get("NAME", "World")
    return f"Hello {name}!"



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
    print("main")
    #print("test")
