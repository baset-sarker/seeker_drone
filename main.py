import argparse
import time,math
from math import atan2,degrees
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from helper_function import run_command,get_area,get_center,get_distance,get_angle,decide_drone_movement
import queue
from DJITelloPy.djitellopy import Tello
import threading,sys
# #import imutils
# import sys

run_process = True
drone_command = False
battery_level = 0
command_str = ""

  
def run_tello():
    global run_process,command_str,battery_level
    tello = Tello()
    tello.connect()
    tello.set_video_resolution(tello.RESOLUTION_480P)
    tello.set_video_fps(tello.FPS_30)
    tello.streamon()
    battery_level = tello.get_battery()
    tello.takeoff()
    
    while run_process:        
        try:
            run_command(command_str,tello)
        except Exception as e:
            print("Exception:",e)
        
        #time.sleep(0.05)

    try:
        tello.land()
    except:
        pass


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    global run_process,drone_command,battery_level,command_str
    count_no_obj = 0

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    n = 0
    for path, img, im0s, vid_cap in dataset:
        # n += 1
        # # run every 5th frame
        # if n % 2!= 0:
        #     continue
        # if n > 1000:
        #     n=0

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
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()


                # image center , drone middle position marking
                im_w,im_h = im0.shape[1],im0.shape[0]
                dx1,dy1 = int(im0.shape[1]/2), int(im0.shape[0])
                cv2.circle(img=im0, center = (dx1,dy1), radius =5, color =(255,255,0), thickness=5)

                # Write results
                for *xyxy, conf, cls in reversed(det):

                    xywhs = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist() 
                    center = get_center(xywhs)
                    distance = get_distance(center, (dx1,dy1))
                    angle = get_angle(center, (dx1,dy1))
                    area = get_area(xywhs)   
 

                    
                    cv2.circle(img=im0, center = get_center(xywhs), radius =10, color =(255,0,0), thickness=5)
                    cv2.line(img=im0, pt1=center, pt2=(dx1,dy1), color=colors[int(cls)], thickness=1)
                    cv2.putText(im0,f'D:{distance:.2f}  A:{angle:.2f}',org=(dx1,dy1-10),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=colors[int(cls)], thickness=1)
                    cv2.putText(im0,f'Bat:{battery_level}%',org=(im_w-75,15),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,0), thickness=1)
                    
                    mng_cmd = decide_drone_movement(distance,angle,area)
                    if mng_cmd != "":
                        command_str = mng_cmd
                
                    if save_img or view_img:  # Add bbox to image
                        label = f'{int(cls)} {names[int(cls)]} {conf:.2f} D:{distance:.2f} A:{angle:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
            else:
                pass
                # count_no_obj += 1
                # #every 5 frame if no object detected, send command rotate
                # if count_no_obj % 5 == 0:
                #     command_str = "rotate_clockwise:30"

                # if count_no_obj > 1000:
                #     count_no_obj = 0


            # Print time (inference + NMS)
            #print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
        

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                #cv2.waitKey(1)  # 1 millisecond
                key = cv2.waitKey(1) & 0xff
                if key == 27 or key == ord('q'): # ESC or q to quit
                    run_process = False
                    cv2.destroyAllWindows()
                    sys.exit()
            
                if key == ord('r'):
                    command_str = "rotate"
                    #command_data.put("rotate")
                if key == ord('l'):
                    command_str = "land"
                
                if key == ord('b'):
                    command_str = "battery"
                    #command_data.put("land")

                if key == ord('a'):
                    command_str = "move_left"
                
                if key == ord('d'):
                    command_str = "move_right"

            
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='stream.txt', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))


    with torch.no_grad():

        # if opt.update:  # update all models (to fix SourceChangeWarning)
        #     for opt.weights in ['yolov7.pt']:
        #         detect()
        #         strip_optimizer(opt.weights)
        # else:
        #     detect()

        drone = threading.Thread(target=detect)
        drone.start()
        time.sleep(2)

        run_tello()

        run_process = False
        drone.join()