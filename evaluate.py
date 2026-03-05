from ultralytics import YOLO


model = YOLO("./runs/detect/train/weights/best.pt")

metrics = model.val(data="./yolo_rgb_dataset/data.yaml", split="test")

precision = metrics.box.p

recall = metrics.box.r

map50 = metrics.box.map50

map5095 = metrics.box.map


with open("Resultsv26.txt", "w") as f:
    f.write(f"Precision:{precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"Map50:{map50}\n")
    f.write(f"Map50-95:{map5095}\n")