from ultralytics import YOLO

model = YOLO("yolo26n.pt")  # pass any model type

results = model.train(data="./yolo_rgb_dataset/data.yaml",
                      batch=0.70,
                      epochs=100,
                      dropout=0.2,
                      patience=40)


