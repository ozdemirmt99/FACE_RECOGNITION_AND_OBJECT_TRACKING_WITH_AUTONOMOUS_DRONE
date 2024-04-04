from YuzTanima import faceCascadeWithYolo
from djitellopy import tello

drone=tello.Tello()
drone.connect()
drone.streamon()
print(drone.get_battery())


opt = {
    "weights":"best.pt",        # Ağırlıkların bulunduğu yer
    "source":"2",               # Görüntü / Görselin Alınacağı Yer ( 0 Kamera için)
    "img_size": 512,            # Görüntünün boyutları
    "conf_thres":0.8,           # Eşik değer Face nesneleri için
    "iou_thres": 0.7,           # IOU threshold for NMS
    "device": "0",              # Kullanılacak cihazın seçimi ( 0 Ekran kartı )
    "view_img": False,          # Display results
    "save_txt": False,          # save results to *.txt
    "save_conf": False,         # save confidences in --save-txt labels
    "no_save": True,            # Video /Fotoğraf Kaydet (durum==False)
    "classes": None,            # filtreleme Class göre
    "agnostic_nms": False,      # class-agnostic NMS
    "augment": False,           # augmented inference
    "update": True,             # Tüm modeller güncelle
    "project": "runs/detect",   # sonuçları kaydet
    "name": "exp",              # kaydet sonuçları project/name
    "exist_ok": True,           # existing project/name ok, do not increment
    "no_trace": True,           # don`t trace model
}
faceCascadeWithYolo(opt,drone)