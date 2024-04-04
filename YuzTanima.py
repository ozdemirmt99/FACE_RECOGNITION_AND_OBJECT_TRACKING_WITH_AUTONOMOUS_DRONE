import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import face_recognition
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from DronGoruntuAlici import LoadStreamsForTello
import threading

class MyThread(threading.Thread):
    def __init__(self, param1, param2):
        threading.Thread.__init__(self)
        self.param1 = param1
        self.param2 = param2
        self.result = False

    def run(self):
        # Yapılacak işlem fonksiyonu
        self.result = face_recognition.compare_faces(self.param1,self.param2)

    def get_result(self):
        return self.result

def faceCascadeWithYolo(opt ,drone,save_img=False):

    targetImage = face_recognition.load_image_file("Resources/target/image.jpg")
    targetEncoded = np.array(face_recognition.face_encodings(targetImage)[0])
    # parametreleri yerleştirme
    source = opt["source"]
    weights = opt["weights"]
    view_img = opt["view_img"]
    save_txt = opt["save_txt"]
    imgsz = opt["img_size"]
    trace = not opt["no_trace"]
    save_img = not opt["no_save"] and not source.endswith('.txt')
    webcam = False  #source.isnumeric() \
    # Directories
    save_dir = Path(increment_path(Path(opt["project"]) / opt["name"], exist_ok=opt["exist_ok"]))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Initialize
    set_logging()
    device = select_device(opt["device"])
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt["img_size"])

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    ## Algoritmada Sınıflandırma istenmesi durumunda bu kısım kullanılabilir.
    ## Bu program için gerekli değildir.

    #classify = False
    # if classify:
    #     modelc = load_classifier(name='resnet101', n=2)  # initialize
    #     modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    ## Yüz tespit algoritmasının dron haricinde çalıştırılması istenirse aşağıdaki kodların yorum satırını açıp
    ## fonskiyona gelen source parametresini 0 yapmalısınız.

    # if webcam:
    #     view_img = check_imshow()
    #     cudnn.benchmark = True  # set True to speed up constant image size inference
    #     dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    # elif source == "2":
    #     dataset = LoadStreamsForTello(source, img_size=imgsz, stride=stride, drone = drone)
    # else:
    #     dataset = LoadImages(source, img_size=imgsz, stride=stride)

    dataset = LoadStreamsForTello(source, img_size=imgsz, stride=stride, drone=drone)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    drone.takeoff()
    while(drone.get_distance_tof() < 165):
        time.sleep(0.1)
        drone.send_rc_control(0, 0, 20, 0)

    time.sleep(0.1)
    drone.send_rc_control(0, 0, 0, 0)


    t0 = time.time()
    t4 = time.time()  # Algoritma Sonlanma durumu için tutulan zaman başlagıcı.
    t5 = time.time()
    anyMath = False;

    for path, img, im0s, vid_cap in dataset:

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt["augment"])[0]

        # Inference
        t1 = time_synchronized()

        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt["augment"])[0]
        t2 = time_synchronized()

        # Apply NMS

        pred = non_max_suppression(pred, opt["conf_thres"], opt["iou_thres"], classes=opt["classes"], agnostic=opt["agnostic_nms"])
        t3 = time_synchronized()
        # Apply Classifier
        # if classify:
        #     pred = apply_classifier(pred, modelc, img, im0s)
         #For breaking program loop for searching faces
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if  not webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det):
                t4 = time.time()
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # if save_txt:  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))/gn ).view(-1).tolist()  # normalized xywh
                    #     print("xywh:",xywh)
                    #     line = (cls, *xywh, conf) if opt["save_conf"] else (cls, *xywh)  # label format
                    #     with open(txt_path + '.txt', 'a') as f:
                    #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # if save_img or view_img:  # Add bbox to image
                    #     label = f'{names[int(cls)]} {conf:.2f}'
                    #     plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                    label = f'{names[int(cls)]} {conf:.2f}'

                    xTop = int(xyxy[0].item())
                    xBottom = int(xyxy[2].item())
                    yTop = int(xyxy[1].item())
                    yBottom = int(xyxy[3].item())
                    vet = im0[yTop:yBottom, xTop:xBottom]

                    currentEncoded = face_recognition.face_encodings(vet)

                    if (len(currentEncoded) > 0):
                        currentEncodedFace = np.array(currentEncoded[0])

                        compareFaceThread = MyThread(currentEncoded, targetEncoded)
                        compareFaceThread.start()
                        # compareFaceThread.join()
                        control = compareFaceThread.get_result()

                        if control[0] :
                            anyMath=True;
                            print("Yüz eşleşmesi gerçekleşti")
                            if time.time() - t5 > 1:
                                t5=time.time();
                                yonlendir(drone, xTop + (xBottom-xTop)/2 , yTop + (yBottom - yTop)/2, xTop,
                                          yTop, xBottom, yBottom, True)
                            else:
                                droneBekle(drone)

                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                        else:
                            if not (anyMath):
                                yonlendir(drone, 0, 0 ,0, 0, 0, 0, yuzAlgilandi=False)
                                print("Aranan yüz değil")
            else:
                print("yüzsüz bölge desin yüz bulana kadar tarama algoritmasını çalıştırılıyor.")
                now_time = time.time()
                yonlendir(drone, 0, 0, 0, 0, 0, 0, yuzAlgilandi=False)
                if(now_time-t4 >= 30):
                    print("dron indi")
                    drone.land()
                    break;
            # Print time (inference + NMS)
            # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            cv2.imshow("Dron Goruntusu ", im0)
            cv2.waitKey(1)  # 1 millisecond

    print(f'Done. ({time.time() - t0:.3f}s)')

def yonlendir(drone, centerX, centerY, sol,
                ust, sag, alt, yuzAlgilandi = False):


    print("yüz alagılandı fonksiyonundayım")
    yukseklik=720
    genislik=920
    yukseklikYarim = yukseklik/2;
    genislikYarim = genislik/2;
    optimumFrameLineLenghth = 250;
    width = sag - sol;
    height = alt - ust;
    Up, Forward, LR, Yaw = 0, 0, 0, 0;
    speed = 25;
    """
    Send RC control via four channels. Command is sent every self.TIME_BTW_RC_CONTROL_COMMANDS seconds.
        Arguments:
            left_right_velocity: -100~100 (left/right)
            forward_backward_velocity: -100~100 (forward/backward)
            up_down_velocity: -100~100 (up/down)
            yaw_velocity: -100~100 (yaw)
    """
    if drone.get_distance_tof() <165:
        drone.send_rc_control(0, 0, 20, 0)
        time.sleep(0.1)
        droneBekle(drone)

    if yuzAlgilandi == True  :
        ## Dronnun yüksekliğini ayarlama.
        # print("CENTER Y :",centerY)
        # if((centerY > yukseklikYarim or centerY < yukseklikYarim)
        #         and abs(centerY - yukseklikYarim) > 50
        # ):
        #
        #     upDown = centerY - yukseklikYarim;
        #     if (centerY < yukseklikYarim):
        #         print("Dron yukarıya gidiyor .....")
        #         # drone.send_rc_control(0, 0, 20, 0)
        #         # time.sleep(0.2)
        #         # drone.send_rc_control(0, 0, 0, 0)
        #         Up = speed;
        #
        #
        #     # dronu yukarı yönlerdir
        #     else:
        #         print("Dron aşağıya gidiyor .....")
        #         # time.sleep(0.01)
        #         #dornu aşağıya yönlendir...
        #         # drone.send_rc_control(0,0,-20, 0)
        #         # time.sleep(0.2)
        #         # drone.send_rc_control(0, 0, 0, 0)
        #         Up = -speed;

        #Dronun görüntüsündeki yüz nesnesini rotatede ortalama (x-ekseni).
        if (centerX > genislikYarim or centerX < yukseklikYarim and abs(centerX - genislikYarim) > 10):
            rotateRL = centerX - genislikYarim
            if (rotateRL > 0):
                #dornu sağa rotate et.
                print("Dron sağa rotate gidiyor .....")
                Yaw = speed;

            else:
                #dornu sola rotate et.
                print("Dron sola roatate gidiyor .....")
                Yaw = -speed;

        # Yeterli yakınlık ayarı

        if((height > optimumFrameLineLenghth or height < optimumFrameLineLenghth)  and abs(height - optimumFrameLineLenghth) > 10):
            if(height > optimumFrameLineLenghth ):
                # Dronu uzaklaştır
                print("Dron uzaklaştır gidiyor .....")
                Forward = -speed;

            else:
                # Dronu yaklaştır.
                print("Dron yakınlaştır gidiyor .....")
                Forward = speed;

        drone.send_rc_control(LR, Forward, Up, Yaw);
        time.sleep(0.1)
    else :
        print("Yüz bulunamadı....")
        # Yüz algılanmadığı vakitte sağa rotateler devam et
        drone.send_rc_control(0, 0, 0, speed)

def droneBekle(drone):
    drone.send_rc_control(0, 0, 0, 0);

