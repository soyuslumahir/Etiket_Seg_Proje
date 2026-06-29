import cv2, numpy as np
from PIL import Image

CC = {0:(200,200,200),1:(0,200,0),2:(0,100,255),3:(255,165,0),4:(255,50,50),
      5:(255,150,50),6:(255,230,50),7:(50,255,50),8:(50,255,200),9:(50,200,255),
      10:(50,50,255),11:(200,50,255),12:(255,50,200),13:(180,180,100)}

samples = [
    'scene_0050', 'scene_0200_a0', 'scene_0350_a1',
    'scene_0450', 'scene_0100_a0', 'scene_0480_a1'
]

thumbs = []
for name in samples:
    img = cv2.imread(f'scenes/images/{name}.jpg')
    msk = np.array(Image.open(f'scenes/masks/{name}.png'))
    ov = img.copy()
    for cid, color in CC.items():
        c = np.array(color, dtype=np.int32)
        diff = np.abs(msk.astype(np.int32) - c).max(axis=2)
        binary = (diff < 5).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            if cv2.contourArea(cnt) < 100: continue
            cv2.drawContours(ov, [cnt], -1, (color[2], color[1], color[0]), 4)
    blended = cv2.addWeighted(img, 0.5, ov, 0.5, 0)
    thumbs.append(cv2.resize(blended, (1000, 750)))

grid = np.vstack([np.hstack(thumbs[:3]), np.hstack(thumbs[3:])])
cv2.imwrite('preview_mask_check.jpg', grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
print('ok')
