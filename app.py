import signal
import sys

from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.logger import logging
from src.exception import CustomizedException

app = Flask(__name__)

def graceful_shutdown(signum, frame):
    print("Received signal", signum)
    print("Shutting down gracefully...")
    # Perform any cleanup here
    # e.g., closing database connections, etc.
    sys.exit(0)
    
@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    elif request.method == "POST":
        try:
            data = CustomData(
                gender=request.form.get("gender"),
                race_ethnicity=request.form.get("ethnicity"),
                parental_level_of_education=request.form.get("parental_level_of_education"),
                lunch=request.form.get("lunch"),
                test_preparation_course=request.form.get("test_preparation_course"),
                reading_score=float(request.form.get("reading_score")),
                writing_score=float(request.form.get("writing_score"))
            )
            
            # Correct method name
            processed_df = data.get_data_as_data_frame()
            predict_pipeline = PredictPipeline()
            
            result = predict_pipeline.predict(processed_df)[0]
            
            # Pass back form data to preserve inputs
            return render_template(
                "home.html",
                result=result,
                gender=data.gender,
                ethnicity=data.race_ethnicity,
                parental_level_of_education=data.parental_level_of_education,
                lunch=data.lunch,
                test_preparation_course=data.test_preparation_course,
                reading_score=data.reading_score,
                writing_score=data.writing_score
            )
        except Exception as e:
            logging.error(f"An error occurred: {CustomizedException(e)}")
            return render_template("home.html", result="An error occurred during prediction.")


if __name__ == "__main__":
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, graceful_shutdown)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, graceful_shutdown)  # Handle termination signals

    app.run(host="0.0.0.0", port=5000,debug=False)
