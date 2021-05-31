import cv2
import numpy as np
import json
import glob
import sys

#imgs C:\Users\Marcin\Desktop\projekt\public.json C:\Users\Marcin\Desktop\projekt\results.json
imgpath = sys.argv[1]
jsonpath = sys.argv[2]
outputpath = sys.argv[3]


def write_json(data, filename=outputpath):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

red_img1 = []
blue_img1 = []
white_img1 = []
gray_img1 = []
yellow_img1 = []


red_fig1 = []
blue_fig1 = []
white_fig1 = []
gray_fig1 = []
yellow_fig1 = []

red_waga = []
blue_waga = []
white_waga = []
gray_waga = []
yellow_waga = []

roi = []




kolory = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 128, 255), (255, 255, 0), (255, 0, 255),
          (0, 255, 255), (128, 128, 128), (165, 255, 165), (19, 69, 139), (128, 0, 128), (238, 130, 238), (208, 224, 64),
          (114, 128, 250), (128, 0, 0), (130, 0, 75), (196, 228, 255), (250, 230, 230), (128, 0, 0), (0, 128, 128),
          (210, 105, 30), (0, 191, 255), (192, 192, 192)]
#kolory = [ WHITE,     BLUE, GREEN, RED, ORANGE, CYAN, MAGENTA,
#           YELLOW, GRAY, LIME, BROWN, PURPLE, VIOLET, TURQUOISE,
#           SALMON, NAVY, INDIGO, BISQUE, LAVENDER, MAROON, TEAL
#           CHOCOLATE SKYBLUE SILVER]
# Write some Text
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
lineType = 2

red_cx = []
red_cy = []
blue_cx = []
blue_cy = []
white_cx = []
white_cy = []
gray_cx = []
gray_cy = []
yellow_cx = []
yellow_cy = []

filenames = [img for img in glob.glob(imgpath + "/*.*")]
#filenames.sort()
print(filenames)
images = []
tekst = {}
write_json(tekst)

for image in filenames:
    red_img1.clear()
    blue_img1.clear()
    white_img1.clear()
    gray_img1.clear()
    yellow_img1.clear()
    red_fig1.clear()
    blue_fig1.clear()
    white_fig1.clear()
    gray_fig1.clear()
    yellow_fig1.clear()
    red_waga.clear()
    blue_waga.clear()
    white_waga.clear()
    gray_waga.clear()
    yellow_waga.clear()
    red_cx.clear()
    red_cy.clear()
    blue_cx.clear()
    blue_cy.clear()
    white_cx.clear()
    white_cy.clear()
    gray_cx.clear()
    gray_cy.clear()
    yellow_cx.clear()
    yellow_cy.clear()
    roi.clear()
    n = 0
    m = 0
    with open(jsonpath) as json_file:
        data = json.load(json_file)
        img_id = image[-11]+image[-10]+image[-9]+image[-8]+image[-7]+image[-6]+image[-5]
        print(img_id)
        for img1 in data[img_id]:
            red_img1.append(img1['red'])
            blue_img1.append(img1['blue'])
            white_img1.append(img1['white'])
            gray_img1.append(img1['grey'])
            yellow_img1.append(img1['yellow'])
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, None, fx=0.35, fy=0.35, interpolation=cv2.INTER_CUBIC)
    gray = cv2.resize(gray, None, fx=0.35, fy=0.35, interpolation=cv2.INTER_CUBIC)

    figure_number = np.zeros_like(img)
    contours_img = np.zeros_like(img)
    contours_color_img = np.zeros_like(img)

    """BACKGROUND"""
    img_filtr = cv2.medianBlur(img, 9)
    img_hsv_filtr = cv2.cvtColor(img_filtr, cv2.COLOR_BGR2HSV)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(150, 150))
    cl1 = clahe.apply(gray)
    img_clahe = cv2.cvtColor(cl1, cv2.COLOR_GRAY2BGR)
    img_hsv_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2HSV)

    bg_lower_hsv_filtr = np.array([0, 0, 145], np.uint8)#filtr medianowy
    bg_higher_hsv_filtr = np.array([180, 45, 220], np.uint8)#filtr medianowy
    bg_mask_filtr = cv2.inRange(img_hsv_filtr, bg_lower_hsv_filtr, bg_higher_hsv_filtr)
    bg_mask_inv_filtr = cv2.bitwise_not(bg_mask_filtr)
    bg_result_filtr = cv2.bitwise_and(img, img, mask=bg_mask_filtr)

    clahe_lower_hsv = np.array([0, 0, 140], np.uint8)  # CLAHE
    clahe_higher_hsv = np.array([180, 255, 255], np.uint8)  # CLAHE
    bg_mask_clahe = cv2.inRange(img_hsv_clahe, clahe_lower_hsv, clahe_higher_hsv)
    kernel_clahe = np.ones((5, 5), np.uint8)
    bg_mask_clahe = cv2.erode(bg_mask_clahe, kernel_clahe, iterations=3)  # czarny
    bg_result_clahe = cv2.bitwise_and(img, img, mask=bg_mask_clahe)

    fg_result = cv2.bitwise_xor(bg_result_filtr, img)
    fg_result2 = cv2.bitwise_xor(bg_result_clahe, img)
    full_result = cv2.bitwise_or(fg_result, fg_result2)

    """DRAW CONTOURS/FOREGROUND"""
    full_result_gray = cv2.cvtColor(full_result, cv2.COLOR_BGR2GRAY)
    _, contours_fg, _ = cv2.findContours(full_result_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours_fg)):
        cnt = contours_fg[i]
        M = cv2.moments(cnt)
        area = cv2.contourArea(cnt)
        if area > 5000:
            n += 1
            cx = (int(M['m10'] / M['m00']))
            cy = (int(M['m01'] / M['m00']))
            cv2.drawContours(contours_img, [cnt], 0, kolory[0], -1)
            #cv2.drawContours(full_result, [cnt], 0, kolory[1], 2)
            #cv2.circle(full_result, (cx, cy), 2, kolory[1], 5)

    foreground_result = cv2.bitwise_and(contours_img, img)
    img_hsv = cv2.cvtColor(foreground_result, cv2.COLOR_BGR2HSV)
    img_hsv_bit_not = cv2.cvtColor(cv2.bitwise_not(foreground_result), cv2.COLOR_BGR2HSV)

    '''RED'''
    red_lower_hsv = np.array([0, 25, 0], np.uint8)
    red_higher_hsv = np.array([10, 255, 255], np.uint8)
    red_mask1 = cv2.inRange(img_hsv, red_lower_hsv, red_higher_hsv)
    red_lower_hsv2 = np.array([166, 25, 0], np.uint8)
    red_higher_hsv2 = np.array([180, 255, 255], np.uint8)
    red_mask2 = cv2.inRange(img_hsv, red_lower_hsv2, red_higher_hsv2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    red_result = cv2.bitwise_and(img, img, mask=red_mask)
    red_result_gray = cv2.cvtColor(red_result, cv2.COLOR_BGR2GRAY)
    _, contours_red, _ = cv2.findContours(red_result_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours_red)):
        cnt = contours_red[i]
        M = cv2.moments(cnt)
        area = cv2.contourArea(cnt)
        if area > 1500:
            cx = (int(M['m10'] / M['m00']))
            cy = (int(M['m01'] / M['m00']))
            red_cx.append(cx)
            red_cy.append(cy)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            width = int(rect[1][0])
            height = int(rect[1][1])
            rozmiar = min(height, width)
            if rozmiar < 57:
                red_waga.append(1)
            elif 57 <= rozmiar < 90:
                red_waga.append(2)
            elif rozmiar >= 90:
                red_waga.append(3)
            cv2.drawContours(full_result, [cnt], 0, kolory[3], 3)
            # cv2.circle(full_result, (cx, cy), 2, kolory[3], 5)
            cv2.drawContours(red_result, [box], 0, kolory[3], 3)
            # cv2.circle(red_result, (cx, cy), 2, kolory[3], 5)
            cv2.circle(figure_number, (cx, cy), 8, kolory[3], 1)
            bottomLeftCornerOfText = (cx + 5, cy + 5)
            cv2.putText(figure_number, "({},{},{})".format(cx, cy, area),
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        kolory[3],
                        lineType)

    '''BLUE'''
    blue_lower_hsv = np.array([100, 60, 125], np.uint8)
    blue_higher_hsv = np.array([125, 255, 255], np.uint8)
    blue_mask = cv2.inRange(img_hsv, blue_lower_hsv, blue_higher_hsv)
    blue_result = cv2.bitwise_and(img, img, mask=blue_mask)
    blue_result_gray = cv2.cvtColor(blue_result, cv2.COLOR_BGR2GRAY)
    _, contours_blue, _ = cv2.findContours(blue_result_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours_blue)):
        cnt = contours_blue[i]
        M = cv2.moments(cnt)
        area = cv2.contourArea(cnt)
        if area > 1500:
            cx = (int(M['m10'] / M['m00']))
            cy = (int(M['m01'] / M['m00']))
            blue_cx.append(cx)
            blue_cy.append(cy)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            width = int(rect[1][0])
            height = int(rect[1][1])
            rozmiar = min(height, width)
            if rozmiar < 57:
                blue_waga.append(1)
            elif 57 <= rozmiar < 90:
                blue_waga.append(2)
            elif rozmiar >= 90:
                blue_waga.append(3)
            cv2.drawContours(full_result, [cnt], 0, kolory[1], 3)
            #cv2.circle(full_result, (cx, cy), 2, kolory[1], 5)
            cv2.drawContours(blue_result, [box], 0, kolory[1], 3)
            #cv2.circle(blue_result, (cx, cy), 2, kolory[1], 5)
            cv2.circle(figure_number, (cx, cy), 8, kolory[1], 1)
            bottomLeftCornerOfText = (cx+5, cy+5)
            cv2.putText(figure_number, "({},{},{})".format(cx, cy, area),
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        kolory[1],
                        lineType)

    '''WHITE'''
    white_lower_hsv = np.array([25, 0, 180], np.uint8)
    white_higher_hsv = np.array([120, 48, 255], np.uint8)
    white_mask = cv2.inRange(img_hsv, white_lower_hsv, white_higher_hsv)
    white_mask_inv = cv2.bitwise_not(white_mask)
    white_result = cv2.bitwise_and(img, img, mask=white_mask)

    white_result_gray = cv2.cvtColor(white_result, cv2.COLOR_BGR2GRAY)

    _, contours_white, _ = cv2.findContours(white_result_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours_white)):
        cnt = contours_white[i]
        M = cv2.moments(cnt)
        area = cv2.contourArea(cnt)
        if area > 1500:
            cx = (int(M['m10'] / M['m00']))
            cy = (int(M['m01'] / M['m00']))
            white_cx.append(cx)
            white_cy.append(cy)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            width = int(rect[1][0])
            height = int(rect[1][1])
            rozmiar = min(height, width)
            #print('rozmiar', rozmiar)
            if rozmiar < 60:
                white_waga.append(1)
            elif 60 <= rozmiar < 90:
                white_waga.append(2)
            elif rozmiar >= 90:
                white_waga.append(3)
            cv2.drawContours(full_result, [cnt], 0, kolory[0], 3)
            #cv2.circle(full_result, (cx, cy), 2, kolory[0], 5)
            cv2.drawContours(white_result, [box], 0, kolory[0], 3)
            #cv2.circle(white_result_final, (cx, cy), 2, kolory[0], 5)
            cv2.circle(figure_number, (cx, cy), 8, kolory[0], 1)
            bottomLeftCornerOfText = (cx+5, cy+5)
            cv2.putText(figure_number, "({},{},{})".format(cx, cy, area),
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        kolory[0],
                        lineType)
    '''GRAY'''
    gray_lower_hsv = np.array([43, 0, 0], np.uint8)
    gray_higher_hsv = np.array([105, 255, 145], np.uint8)
    gray_mask = cv2.inRange(img_hsv, gray_lower_hsv, gray_higher_hsv)
    gray_result = cv2.bitwise_and(img, img, mask=gray_mask)
    gray_result_gray = cv2.cvtColor(gray_result, cv2.COLOR_BGR2GRAY)
    _, contours_gray, _ = cv2.findContours(gray_result_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours_gray)):
        cnt = contours_gray[i]
        M = cv2.moments(cnt)
        area = cv2.contourArea(cnt)
        if area > 1500:
            cx = (int(M['m10'] / M['m00']))
            cy = (int(M['m01'] / M['m00']))
            gray_cx.append(cx)
            gray_cy.append(cy)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            width = int(rect[1][0])
            height = int(rect[1][1])
            rozmiar = min(height, width)
            if rozmiar < 57:
                gray_waga.append(1)
            elif 57 <= rozmiar < 90:
                gray_waga.append(2)
            elif rozmiar >= 90:
                gray_waga.append(3)
            cv2.drawContours(full_result, [cnt], 0, kolory[8], 3)
            # cv2.circle(full_result, (cx, cy), 2, kolory[8], 5)
            cv2.drawContours(gray_result, [box], 0, kolory[8], 3)
            # cv2.circle(gray_result, (cx, cy), 2, kolory[8], 5)
            cv2.circle(figure_number, (cx, cy), 8, kolory[8], 1)
            bottomLeftCornerOfText = (cx + 5, cy + 5)
            cv2.putText(figure_number, "({},{},{})".format(cx, cy, area),
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        kolory[8],
                        lineType)
    '''YELLOW'''
    yellow_lower_hsv = np.array([104, 105, 0], np.uint8)
    yellow_higher_hsv = np.array([180, 255, 255], np.uint8)
    yellow_mask = cv2.inRange(img_hsv_bit_not, yellow_lower_hsv, yellow_higher_hsv)
    yellow_result = cv2.bitwise_and(img, img, mask=yellow_mask)
    yellow_result_gray = cv2.cvtColor(yellow_result, cv2.COLOR_BGR2GRAY)
    _, contours_yellow, _ = cv2.findContours(yellow_result_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours_yellow)):
        cnt = contours_yellow[i]
        M = cv2.moments(cnt)
        area = cv2.contourArea(cnt)
        if area > 1500:
            cx = (int(M['m10'] / M['m00']))
            cy = (int(M['m01'] / M['m00']))
            yellow_cx.append(cx)
            yellow_cy.append(cy)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            width = int(rect[1][0])
            height = int(rect[1][1])
            rozmiar = min(height, width)
            if rozmiar < 57:
                yellow_waga.append(1)
            elif 57 <= rozmiar < 90:
                yellow_waga.append(2)
            elif rozmiar >= 90:
                yellow_waga.append(3)
            cv2.drawContours(full_result, [cnt], 0, kolory[7], 3)
            #cv2.circle(full_result, (cx, cy), 2, kolory[7], 5)
            cv2.drawContours(yellow_result, [box], 0, kolory[7], 3)
            #cv2.circle(yellow_result, (cx, cy), 2, kolory[7], 5)
            cv2.circle(figure_number, (cx, cy), 8, kolory[7], 1)
            bottomLeftCornerOfText = (cx+5, cy+5)
            cv2.putText(figure_number, "({},{},{})".format(cx, cy, area),
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        kolory[7],
                        lineType)


    """GRUPOWANIE FIGUR + WYKRYWANIE DZIUREK"""
    dziurki_img = img.copy()
    _, contours_color, _ = cv2.findContours(full_result_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    liczba_dziurek_figura = []
    for i in range(len(contours_color)):
        cnt = contours_color[i]
        M = cv2.moments(cnt)
        area = cv2.contourArea(cnt)
        if area > 5000:
            #x, y, w, h = cv2.boundingRect(cnt)
            #cv2.rectangle(contours_color_img, (x, y), (x + w, y + h), kolory[m], 2)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cx = (int(M['m10'] / M['m00']))
            cy = (int(M['m01'] / M['m00']))
            width = int(rect[1][0])
            height = int(rect[1][1])
            #print(rect)
            #print(box)
            src_pts = box.astype("float32")
            dst_pts = np.array([[0, height - 1],
                                [0, 0],
                                [width - 1, 0],
                                [width - 1, height - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(img, M, (width, height))
            warped = cv2.resize(warped, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(warped_gray, 20, 20)
            circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 50,
                                       param1=1, param2=13, minRadius=15, maxRadius=20)
            if circles is not None:
                m += 1
                roi.append(warped)
                cv2.drawContours(contours_color_img, [box], 0, kolory[m], 2)
                cv2.drawContours(contours_color_img, [cnt],  0, kolory[m], -1)
                red_fig1.append(0)
                blue_fig1.append(0)
                white_fig1.append(0)
                gray_fig1.append(0)
                yellow_fig1.append(0)
                for j in range(len(red_cx)):
                    if np.all(contours_color_img[red_cy[j], red_cx[j]] == kolory[m]):
                        red_fig1[m-1] += 1*red_waga[j]
                        cv2.circle(figure_number, (red_cx[j], red_cy[j]), 5, kolory[m], -1)
                for j in range(len(blue_cx)):
                    if np.all(contours_color_img[blue_cy[j], blue_cx[j]] == kolory[m]):
                        blue_fig1[m-1] += 1*blue_waga[j]
                        cv2.circle(figure_number, (blue_cx[j], blue_cy[j]), 5, kolory[m], -1)
                for j in range(len(white_cx)):
                    if np.all(contours_color_img[white_cy[j], white_cx[j]] == kolory[m]):
                        white_fig1[m-1] += 1*white_waga[j]
                        cv2.circle(figure_number, (white_cx[j], white_cy[j]), 5, kolory[m], -1)
                for j in range(len(gray_cx)):
                    if np.all(contours_color_img[gray_cy[j], gray_cx[j]] == kolory[m]):
                        gray_fig1[m-1] += 1*gray_waga[j]
                        cv2.circle(figure_number, (gray_cx[j], gray_cy[j]), 5, kolory[m], -1)
                for j in range(len(yellow_cx)):
                    if np.all(contours_color_img[yellow_cy[j], yellow_cx[j]] == kolory[m]):
                        yellow_fig1[m-1] += 1*yellow_waga[j]
                        cv2.circle(figure_number, (yellow_cx[j], yellow_cy[j]), 5, kolory[m], -1)
                circles = np.uint16(np.around(circles))
                for j in circles[0, :]:
                    # draw the outer circle
                    cv2.circle(warped, (j[0], j[1]), j[2], (0, 255, 0), 2)
                    # draw the center of the circle
                    cv2.circle(warped, (j[0], j[1]), 2, (0, 0, 255), 3)
                liczba_dziurek_figura.append(len(circles[0, :]))
                bottomLeftCornerOfText = (cx, cy+50)
                cv2.putText(dziurki_img, "(Ilosc dziurek: {})".format(len(circles[0, :])),
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            kolory[1],
                            lineType)
                cv2.imwrite("roi {}.jpg".format(m), warped)

    print('FIGURY WEJSCIOWE')
    print('r', red_img1)
    print('b', blue_img1)
    print('w', white_img1)
    print('g', gray_img1)
    print('y', yellow_img1)
    print('FIGURY WYJSCIOWE PRZED SORTOWANIEM')
    print('   be', 'gn', 'rd', 'oe', 'cn', 'ma', 'yw', 'gy', 'le', 'bn', 'pe', 'vt', 'tq', 'sn', 'ny', 'io')
    print('r', red_fig1)
    print('b', blue_fig1)
    print('w', white_fig1)
    print('g', gray_fig1)
    print('y', yellow_fig1)
    print(liczba_dziurek_figura)

    """ PRZYPISANIE ODPOWIEDNIEGO KONTURU DO FIGURY Z PLIKU """
    suma_1_sortowanie = []
    minimum1 = 0
    wejsciowa_figura = []
    dopasowane_klocki_red = []
    dopasowane_klocki_blue = []
    dopasowane_klocki_white = []
    dopasowane_klocki_gray = []
    dopasowane_klocki_yellow = []
    dopasowana_liczba_dziurek = []
    dopasowana_figura = []
    niedopasowana_figura_wejsciowa = []

    for i in range(0, len(red_img1)):
        suma_1_sortowanie.clear()
        wejsciowa_figura.append(i)
        for j in range(0, len(red_fig1)):
            suma_1_sortowanie.append(abs(int(red_img1[i]) - red_fig1[j]) + abs(int(blue_img1[i]) - blue_fig1[j]) +
                                     abs(int(white_img1[i]) - white_fig1[j]) + abs(int(gray_img1[i]) - gray_fig1[j]) +
                                     abs(int(yellow_img1[i]) - yellow_fig1[j]))
            minimum1 = np.argmin(suma_1_sortowanie)
        if min(suma_1_sortowanie) is 0:
            dopasowane_klocki_red.append(red_fig1[minimum1])
            dopasowane_klocki_blue.append(blue_fig1[minimum1])
            dopasowane_klocki_white.append(white_fig1[minimum1])
            dopasowane_klocki_gray.append(gray_fig1[minimum1])
            dopasowane_klocki_yellow.append(yellow_fig1[minimum1])
            dopasowana_liczba_dziurek.append(liczba_dziurek_figura[minimum1])
            dopasowana_figura.append(minimum1)
        else:
            dopasowane_klocki_red.append('x')
            dopasowane_klocki_blue.append('x')
            dopasowane_klocki_white.append('x')
            dopasowane_klocki_gray.append('x')
            dopasowane_klocki_yellow.append('x')
            dopasowana_liczba_dziurek.append('x')
            niedopasowana_figura_wejsciowa.append(i)
    niedopasowana_figura_wyjsciowa = list(set(wejsciowa_figura) - set(dopasowana_figura))
    print('FIGURY WEJSCIOWE')
    print('r', red_img1)
    print('b', blue_img1)
    print('w', white_img1)
    print('g', gray_img1)
    print('y', yellow_img1)
    print('1 SORTOWANIE')
    print('r', dopasowane_klocki_red)
    print('b', dopasowane_klocki_blue)
    print('w', dopasowane_klocki_white)
    print('g', dopasowane_klocki_gray)
    print('y', dopasowane_klocki_yellow)
    print('dziurki', dopasowana_liczba_dziurek)
    print('numer figury dopasowanej', dopasowana_figura)
    print('niedopasowano figur wejsciowych o nr', niedopasowana_figura_wejsciowa)
    print('niedopasowano figur wyjsciowych o nr', niedopasowana_figura_wyjsciowa)
    print('------------2 SORTOWANIE-----------------')
    # print('sumy', suma[i])
    suma_2_sortowanie = []
    minimum2 = 0
    for i in range(0, len(niedopasowana_figura_wejsciowa)):
        suma_2_sortowanie.clear()
        for j in range(0, len(niedopasowana_figura_wyjsciowa)):
            suma_2_sortowanie.append(
                abs(int(red_img1[niedopasowana_figura_wejsciowa[i]]) - red_fig1[niedopasowana_figura_wyjsciowa[j]]) +
                abs(int(blue_img1[niedopasowana_figura_wejsciowa[i]]) - blue_fig1[niedopasowana_figura_wyjsciowa[j]]) +
                abs(int(white_img1[niedopasowana_figura_wejsciowa[i]]) - white_fig1[
                    niedopasowana_figura_wyjsciowa[j]]) +
                abs(int(gray_img1[niedopasowana_figura_wejsciowa[i]]) - gray_fig1[niedopasowana_figura_wyjsciowa[j]]) +
                abs(int(yellow_img1[niedopasowana_figura_wejsciowa[i]]) - yellow_fig1[
                    niedopasowana_figura_wyjsciowa[j]]))
            minimum2 = np.argmin(suma_2_sortowanie)
            dopasowane_klocki_red[niedopasowana_figura_wejsciowa[i]] = red_fig1[niedopasowana_figura_wyjsciowa[minimum2]]
            dopasowane_klocki_blue[niedopasowana_figura_wejsciowa[i]] = blue_fig1[niedopasowana_figura_wyjsciowa[minimum2]]
            dopasowane_klocki_white[niedopasowana_figura_wejsciowa[i]] = white_fig1[niedopasowana_figura_wyjsciowa[minimum2]]
            dopasowane_klocki_gray[niedopasowana_figura_wejsciowa[i]] = gray_fig1[niedopasowana_figura_wyjsciowa[minimum2]]
            dopasowane_klocki_yellow[niedopasowana_figura_wejsciowa[i]] = yellow_fig1[niedopasowana_figura_wyjsciowa[minimum2]]
            dopasowana_liczba_dziurek[niedopasowana_figura_wejsciowa[i]] = liczba_dziurek_figura[
                niedopasowana_figura_wyjsciowa[minimum2]]
        print(suma_2_sortowanie)
        print(minimum2)

    print('----------FIGURY WEJSCIOWE---------------')
    print('czerwony ', red_img1)
    print('niebieski', blue_img1)
    print('bialy    ', white_img1)
    print('szary    ', gray_img1)
    print('zolty    ', yellow_img1)
    print('------------2 SORTOWANIE-----------------')
    print('czerwony ', dopasowane_klocki_red)
    print('niebieski', dopasowane_klocki_blue)
    print('bialy    ', dopasowane_klocki_white)
    print('szary    ', dopasowane_klocki_gray)
    print('zolty    ', dopasowane_klocki_yellow)
    print('dziurki  ', dopasowana_liczba_dziurek)

    with open(outputpath) as json_file:
        dataw = json.load(json_file)
        dataw[img_id] = []
        for i in range(len(dopasowana_liczba_dziurek)):
            y = dopasowana_liczba_dziurek[i]
            dataw[img_id].append(y)
    write_json(dataw)
    key_code = cv2.waitKey(10)
    if key_code == 27:
        # escape key pressed
        break
    #cv2.imshow('image', img)
    #cv2.imwrite('full_result.jpg', full_result)
    #cv2.imwrite('bg_result1.jpg', bg_result_filtr)
    #cv2.imshow('bg_result_darken', bg_result_darken)
    #cv2.imshow('bg_result_filtr', bg_result_filtr)
    #cv2.imshow('bg_result_clahe', bg_result_clahe)
    #cv2.imshow('bg result', bg_result)
    #cv2.imshow('foreground_result', foreground_result)
    #cv2.imshow('foreground_result white ', foreground_result_white)
    #cv2.imwrite('foreground_result.jpg', foreground_result)
    #cv2.imshow('fg result', fg_result)
    #cv2.imshow('fg result1', fg_result1)
    #cv2.imshow('fg result2', fg_result2)
    #cv2.imwrite('red_canny.jpg', red_result)
    #cv2.imwrite('blue_canny.jpg', blue_result)
    #cv2.imwrite('white_canny.jpg', white_result)
    #cv2.imwrite('gray_canny.jpg', gray_result)
    #cv2.imwrite('yellow_canny.jpg', yellow_result)
    #cv2.imshow('red result', red_result)
    #cv2.imshow('blue result', blue_result)
    #cv2.imshow('white result', white_result)
    #cv2.imshow('gray result', gray_result)
    #cv2.imshow('yellow result', yellow_result)
    #cv2.imshow('full result', full_result)
    #cv2.imshow('contours img', contours_color_img)
    # cv2.imshow('dziurki', dziurki_img)
    #cv2.imshow('full result1', full_result1)
    #cv2.imshow('figure number', figure_number)
cv2.destroyAllWindows()
