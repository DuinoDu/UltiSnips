import cv2
import numpy as np
import cPickle
import os
import imageio
from PIL import Image
import importlib


class Logo(object):
    def __init__(self, logofile, start_frame=0, duration=10, x=None, y=None, scale=1.0, M=None):
        png = cv2.imread(logofile, cv2.IMREAD_UNCHANGED)
        self.logo = png[:, :, :3].astype(float)
        alpha = png[:, :, 3].astype(float)/255
        alpha = alpha[:, :, np.newaxis]
        self.alpha = np.concatenate((alpha, alpha, alpha), axis=2)

        self.start_frame = start_frame
        self.end_frame = start_frame + duration 
        self.cnt = 0
        self.x = x
        self.y = y
        self.scale = scale
        self.M = M
        self.logo = cv2.resize(self.logo, (int(png.shape[1]*scale), int(png.shape[0]*scale)))
        self.alpha = cv2.resize(self.alpha, (int(png.shape[1]*scale), int(png.shape[0]*scale)))

    def draw(self, bg):
        x, y = self.x, self.y
        if self.start_frame < self.cnt < self.end_frame:
            logo = self.logo
            h, w = logo.shape[:2]
            try:
                foreground = cv2.multiply(self.alpha, logo.astype(float))
                background = cv2.multiply(1-self.alpha, bg[y:y+h, x:x+w].astype(float))
            except Exception as e:
                __import__('ipdb').set_trace()
                raise e
            out = cv2.add(foreground, background)
            bg[y:y+h, x:x+w] = out.astype(bg.dtype)
        self.cnt += 1
        return bg

    def _transform_logo(self, img, M):
        img_transformed = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
        nonzero = np.where(img_transformed[:,:,0]!=0)
        ymin = nonzero[0].min()
        ymax = nonzero[0].max()
        xmin = nonzero[1].min()
        xmax = nonzero[1].max()
        img = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), borderValue=[255,255,255])
        img = img[ymin:ymax, xmin:xmax]
        return img


def render_with_music(src_video_path, dst_video_path):
    aud_path = src_video_path[:-3] + '.m4a'
    save_video_music_path = dst_video_path[:-4] + '_music' + dst_video_path[-4:]
    if os.path.exists(save_video_music_path):
        cmd = 'rm -f %s' % save_video_music_path
        os.system(cmd)
    cmd = "ffmpeg -i %s -vn -y -acodec copy %s" % (src_video_path, aud_path)
    os.system(cmd)
    cmd = "ffmpeg -i %s -i %s -vcodec copy -acodec copy %s" % (dst_video_path, aud_path, save_video_music_path)
    os.system(cmd)
    cmd = 'rm -f %s' % aud_path
    os.system(cmd)
    cmd = 'mv %s %s' % (save_video_music_path, dst_video_path)
    os.system(cmd)


def draw_kps(im, kps, kps_thresh=-1, show_num=False):
    skeleton = np.array([[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
                         [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]) - 1
    point_color = (0, 255, 255)
    skeleton_color = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
                      [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
                      [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [0, 0, 255]]
    # kps: (num_kps * 3, )
    kps = kps.reshape((-1, 3))
    for j in range(kps.shape[0]):
        x = int(kps[j, 0] + 0.5)
        y = int(kps[j, 1] + 0.5)
        v = kps[j, 2]
        if kps_thresh < v < 3:
            if point_color is None:
                color = (rand() * 255, rand() * 255, rand() * 255)
            elif isinstance(point_color, list):
                color = point_color[j]
            else:
                color = point_color
            cv2.circle(im, (x, y), 2, color=color, thickness=2)
            if show_num:
                cv2.putText(im, '%d' % j, (x+3, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, color, 1)
    if skeleton is not None:
        for j in range(skeleton.shape[0]):
            p1 = skeleton[j, 0]
            p2 = skeleton[j, 1]
            x1 = int(kps[p1, 0] + 0.5)
            y1 = int(kps[p1, 1] + 0.5)
            x2 = int(kps[p2, 0] + 0.5)
            y2 = int(kps[p2, 1] + 0.5)
            if kps_thresh < kps[p1, 2] < 3 and kps_thresh < kps[p2, 2] < 3:
                if skeleton_color is None:
                    color = (rand() * 255, rand() * 255, rand() * 255)
                elif isinstance(skeleton_color, list):
                    color = skeleton_color[j]
                else:
                    color = skeleton_color
                cv2.line(im, (x1, y1), (x2, y2), color=color, thickness=2)
    return im


def draw_mask(im, bbox):
    bbox[0] = max(bbox[0], 0)
    bbox[1] = max(bbox[1], 0)
    bbox[2] = min(bbox[2], im.shape[1])
    bbox[3] = min(bbox[3], im.shape[1])
    h = max(0, bbox[3] - bbox[1])
    w = max(0, bbox[2] - bbox[0])

    if h == 0 or w == 0:
        return im

    mask = np.zeros((bbox[3]-bbox[1], bbox[2]-bbox[0], 3)).astype(np.float32)
    mask[:, :, 0] = 255
    mask[:, :, 2] = 255
    roi = im[bbox[1]:bbox[3], bbox[0]:bbox[2]].astype(np.float32)
    try:
        roi = cv2.addWeighted(roi, 0.8, mask, 0.2, 0)
    except Exception as e:
        __import__('ipdb').set_trace()
        raise e
    im[bbox[1]:bbox[3], bbox[0]:bbox[2]] = roi.astype(im.dtype)
    return im


def main(args):
    cfg = importlib.import_module(args.cfg[:-3]).video
    reader = cv2.VideoCapture(cfg.mp4)
    logos = []
    for logo in cfg.logos:
        mLogo = Logo(logo.img, start_frame=logo.start_frame, duration=logo.duration, 
                     x=logo.x, y=logo.y, scale=logo.scale)
        logos.append(mLogo)

    motion = [1000, 1000, 0, 0]

    if args.render:
        save_video_path = cfg.mp4[:-4] + '_render.mp4'
        fps = int(reader.get(cv2.CAP_PROP_FPS))
        writer = imageio.get_writer(save_video_path, fps=fps)

    imgdir = cfg.mp4[:-4]
    det = cfg.mp4[:-4] + '_res_kps.pkl'
    if not os.path.exists(det):
        print('please detection body first')
        return

    roidb = cPickle.load(open(det, 'r'))
    print("len of roidb: %d" % len(roidb))
    imgsrc = sorted([os.path.join(imgdir, x) for x in sorted(os.listdir(imgdir)) if x.endswith('.jpg') and 'draw' not in x])
    imgdraw = sorted([os.path.join(imgdir, x) for x in sorted(os.listdir(imgdir)) if x.endswith('.jpg') and 'draw' in x])

    for cnt, imgfile in enumerate(imgsrc):
        f = cv2.imread(imgfile)
        if not args.render:
            print(f.shape, "%s / %s" % (cnt, len(imgsrc)))
        # draw human box
        boxes = roidb[cnt]['person_boxes'][:, :4].astype(np.int)
        for box in boxes:
            if cfg.det:
                f = cv2.rectangle(f, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            motion[0] = min(box[0], motion[0])
            motion[1] = min(box[1], motion[1])
            motion[2] = max(box[2], motion[2])
            motion[3] = max(box[3], motion[3])

        # draw skeleton
        kps = roidb[cnt]['keypoints'].astype(np.int)
        if len(kps.shape) > 1:
            kps = kps[:, :51]
        for k in kps:
            if not cfg.kps:
                break
            f = draw_kps(f, k)

        # draw mask
        if cfg.mask:
            if cfg.mask_type is 'merge':
                f = draw_mask(f, motion)
            elif cfg.mask_type is 'seperate':
                for box in boxes:
                    f = draw_mask(f, box)

        # draw logo
        for logo in logos:
            f = logo.draw(f)

        if args.render:
            if cnt % 100 == 0:
                print("%s / %s" % (cnt, len(imgsrc)))
            try:
                writer.append_data(f[:, :, ::-1])
            except Exception as e:
                pass
        else:
            cv2.imshow("image", f)
            ch = cv2.waitKey(10) & 0xff
            if ch == ord('q'):
                break

    if args.render:
        writer.close()
        #render_with_music(cfg.mp4, save_video_path)
        #print("output to %s" % save_video_path)
    else:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='vis video')
    parser.add_argument('--cfg', default='', type=str, help='config file', required=True)
    parser.add_argument('--render', dest='render', action='store_true', help='render output video')
    args = parser.parse_args()
    main(args)
