import os
import glob
import shutil
from PIL import Image
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Database config
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///defects.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# Database model
class Defect(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_name = db.Column(db.String(100))
    defect_type = db.Column(db.String(50))
    confidence = db.Column(db.Float)

with app.app_context():
    db.create_all()

# Load YOLO model
model = YOLO("models/best.pt")

def latest_predict_folder():
    folders = glob.glob("runs/detect/predict*")
    return max(folders, key=os.path.getmtime) if folders else None

@app.route("/", methods=["GET", "POST"])
def index():
    result_image = None
    message = ""

    if request.method == "POST":
        file = request.files.get("file")
        if file:
            filename = secure_filename(file.filename)
            upload_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(upload_path)

            # Run model without relying on saved files and get annotated image
            results = model(upload_path, conf=0.05, save=False)
            res = results[0]

            # Try to get the annotated image from the result; fall back to original
            try:
                annotated = res.plot()
            except Exception:
                try:
                    annotated = res.orig_img
                except Exception:
                    annotated = None

            # Save annotated (or original) image to results folder with same filename
            dest_path = os.path.join(RESULT_FOLDER, filename)
            if annotated is not None:
                # res.plot() returns an ndarray in RGB; save via PIL
                Image.fromarray(annotated).save(dest_path)
                result_image = f"static/results/{filename}"
            else:
                # if something went wrong, copy the uploaded file
                shutil.copy(upload_path, dest_path)
                result_image = f"static/results/{filename}"

            # Save detections to DB
            boxes = getattr(res, "boxes", [])
            if boxes is None or len(boxes) == 0:
                message = "No defects detected."
                # Record a no-defect entry so dashboard can track this upload
                db.session.add(Defect(
                    image_name=filename,
                    defect_type="no_defect",
                    confidence=0.0
                ))
                db.session.commit()
            else:
                for box in boxes:
                    # access class and confidence robustly
                    try:
                        cls = int(box.cls)
                    except Exception:
                        cls = int(box.cls[0])
                    try:
                        conf = float(box.conf)
                    except Exception:
                        conf = float(box.conf[0])

                    name = model.names.get(cls, str(cls))
                    db.session.add(Defect(
                        image_name=filename,
                        defect_type=name,
                        confidence=round(conf * 100, 2)
                    ))
                db.session.commit()

    return render_template("index.html", uploaded=bool(result_image),
                           result_image=result_image, message=message)
@app.route("/dashboard")
def dashboard():
    defects = Defect.query.all()

    # Compute image-level metrics (distinct uploads)
    total_rows = len(defects)
    image_map = {}
    stats = {}
    total_conf = 0.0

    for d in defects:
        # per-image defect presence
        if d.image_name not in image_map:
            image_map[d.image_name] = False
        if d.defect_type != "no_defect":
            image_map[d.image_name] = True

        # stats for defect types (exclude 'no_defect')
        if d.defect_type != "no_defect":
            stats[d.defect_type] = stats.get(d.defect_type, 0) + 1
            total_conf += d.confidence

    total_images = len(image_map)
    images_with_defects = sum(1 for v in image_map.values() if v)
    images_without_defects = total_images - images_with_defects

    total_defects = sum(stats.values())
    most_common = max(stats, key=stats.get) if stats else "None"
    avg_conf = round(total_conf / total_defects, 2) if total_defects else 0

    labels = list(stats.keys())
    values = list(stats.values())

    return render_template(
        "dashboard.html",
        defects=defects,
        labels=labels,
        values=values,
        total_defects=total_defects,
        most_common=most_common,
        avg_conf=avg_conf,
        total_images=total_images,
        images_with_defects=images_with_defects,
        images_without_defects=images_without_defects
    )


if __name__ == "__main__":
    app.run(debug=True)
