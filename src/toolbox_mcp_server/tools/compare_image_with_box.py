from skimage.metrics import structural_similarity

import imutils
import cv2
import base64
import mcp.types as types

def compare_image_with_box(srcA, srcB):
    # extract the ROI from the image
    imageA = cv2.imread(srcA)
    imageB = cv2.imread(srcB)

    # 检查图片尺寸
    if imageA is None or imageB is None:
        return [types.TextContent(type="text", text="Error: Failed to read one or both images")]
    if imageA.shape != imageB.shape:
        # 如果图片尺寸不同，按较小的图片的尺寸截取
        if imageA.shape[0] > imageB.shape[0]:
            imageA = imageA[:imageB.shape[0], :]
        else:
            imageB = imageB[:imageA.shape[0], :]
        
        if imageA.shape[1] > imageB.shape[1]:
            imageA = imageA[:, :imageB.shape[1]]
        else:
            imageB = imageB[:, :imageA.shape[1]]
        # return [types.TextContent(type="text", text="Error: Input images must have the same dimensions")]
    
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    (score, diff) = structural_similarity(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")

    print("SSIM: {}".format(score))

    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

    #返回imageB的base64编码
    _, buffer = cv2.imencode('.jpg', imageB)
    image_base64 = str(base64.b64encode(buffer), encoding='utf-8')
    return [types.ImageContent(type="image", mimeType="image/png", data=image_base64)]