import cv2
import time
import multiprocessing as mp

import sys, os
sys.path.append(os.path.join(os.getcwd(),'python/'))
import darknet as dn


def array_to_image(arr):
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = (arr/255.0).flatten()
    data = dn.c_array(dn.c_float, arr)
    im = dn.IMAGE(w,h,c,data)
    return im


def detect_loop(img_q, box_q, args):
    net = dn.load_net(args.cfg, args.weights, 0)
    meta = dn.load_meta(args.data)
    while True:
        im = img_q.get() 
        im = cv2.resize(im, (int(im.shape[1]/2), int(im.shape[0]/2)))
        im = array_to_image(im)
        dn.rgbgr_image(im)
        ret = dn.detect_np(net, meta, im)
        box_q.put(ret)


def draw_box(im, box):
    for obj in box:
        cx = obj[2][0]
        cy = obj[2][1]
        w = obj[2][2]
        h = obj[2][3]
        
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)

        im = cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return im


def main(args):
    img_q = mp.Queue()
    box_q = mp.Queue()
    detect_process = mp.Process(target=detect_loop, args=(img_q, box_q, args))
    detect_process.start()

    cv2.namedWindow('image')
    cap = cv2.VideoCapture(args.video)
    while True:
        ok, f = cap.read()
        if ok:    
            img_q.put(f)
            box = []
            try:
                box = box_q.get(False)
            except Exception as e:
                pass
            if len(box) > 0:
                f = draw_box(f, box)

            cv2.imshow("image", f)
        ch = cv2.waitKey(1)
        if ch == 27: #ord('q')
            break
    cv2.destroyAllWindows()
    detect_process.terminate()




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Yolo with multiprocess')
    parser.add_argument('--video', default='data/edifice.mp4', type=str, help='input video file mp4')
    parser.add_argument('--data', default='cfg/coco.data', type=str, help='darknet model data')
    parser.add_argument('--cfg', default='cfg/yolov3-tiny.cfg', type=str, help='model cfg')
    parser.add_argument('--weights', default='yolov3-tiny.weights', type=str, help='model weights')
    args = parser.parse_args()
    main(args)
