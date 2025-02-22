import asyncio
import cv2
import websockets
from ultralytics import YOLO
from gtts import gTTS
import os
import pygame

pygame.init()
pygame.mixer.init()
buffer = 0

model = YOLO("yolov8s.pt")
# model2 = YOLO("best_traffic_nano_yolo.pt")

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("/dev/video0")

async def text_to_speech(text, language='en'):
    tts = gTTS(text=text, lang=language)
    tts.save("output.mp3")
    pygame.mixer.music.load("output.mp3")
    pygame.mixer.music.play()

async def process():
    global buffer
    url = "ws://192.168.191.161/ws"
    async with websockets.connect(url) as websocket:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            cv2.line(frame, (140, 0), (140, 500), (255, 0, 0), 3)
            cv2.line(frame, (500, 0), (500, 500), (255, 0, 0), 3)
            
            cropped_frame = frame[0:479, 140:500]
            result = model.predict(cropped_frame, conf=0.75)[0]

            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
                width = abs(x2 - x1)
                height = abs(y2 - y1)
                area = width * height
                stage = "close" if area > 105000 else "far"

                if stage == "close":
                    name = str(int(box.cls))
                    cv2.rectangle(cropped_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(cropped_frame, name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # result2 = model2.predict(frame, conf=0.6)[0]
                       
            # for box in result2.boxes:
            #     # if int(box.cls) == 1:
            #     #     text_to_speech("red light")
            #     #     print("red light")
            #     # elif int(box.cls) == 2:
            #     #     text_to_speech("red light")
            #     #     print("red light")
            #     name = "red light"
            #     x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
            #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            #     cv2.putText(frame, name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imshow("booh", frame)
            # cv2.imshow("test", cropped_frame)

            key = cv2.waitKey(3)
            if key & 0xFF == ord('q'):
                break
            if buffer % 20 == 0:
                if 'stage' in locals():
                    if stage == "close":
                        await websocket.send("100")
                    elif stage == "far":
                        await websocket.send("50")
                    # else:
                    #     await websocket.send("0")

            buffer += 1
        cap.release()
        cv2.destroyAllWindows()

async def main():
    await asyncio.gather(process())

asyncio.run(main())

pygame.quit()